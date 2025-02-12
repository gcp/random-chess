#!/usr/bin/env python3

import time
import chess
import random
import re
import math
import os
import argparse
import threading
import queue
from datetime import datetime
from typing import Tuple
import concurrent.futures

import openai  # Requires openai==0.28

# Configuration defaults
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_POSITIONS = 200  # default to 200 positions
CONFIDENCE_LEVEL = 1.96   # ~95% confidence intervals
MAX_RETRIES = 3           # we'll retry up to 3 times total


def generate_positions(num_positions: int = 200) -> list:
    """
    Generate random positions, weighting pawn moves to be 1/4 as likely as other moves.
    """
    positions = []
    while len(positions) < num_positions:
        board = chess.Board()
        num_moves = random.randint(10, 60)
        valid = True

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                valid = False
                break

            # Weight=1 if it's a pawn, else weight=4 => pawns are 1/4 as common
            weights = [
                1 if board.piece_at(move.from_square).piece_type == chess.PAWN else 4
                for move in legal_moves
            ]
            move = random.choices(legal_moves, weights=weights, k=1)[0]
            board.push(move)

            if board.is_game_over():
                valid = False
                break

        if valid and not board.is_game_over():
            fen = board.fen()
            legal_moves_sorted = sorted(move.uci() for move in board.legal_moves)
            positions.append((fen, legal_moves_sorted))

    return positions


def parse_llm_response(response: str) -> list:
    """
    Extract moves from a structured LLM response like:
      'Legal moves: e2e4,e2e3,d2d4,...'
    or possibly including brackets or quotes, e.g.:
      'Legal moves: [e2e4,e2e3,d2d4]'
      'Legal moves: "e2e4,e2e3,d2d4"'
    We'll accept any leading/trailing quotes/brackets around the comma-separated moves.
    """
    # Search for the "Legal moves:" line (case-insensitive).
    match = re.search(r"(?i)\blegal moves:\s*([^\n]+)", response)
    if not match:
        return []

    # Extract whatever follows "Legal moves:" up to the end of the line (or next newline).
    raw_str = match.group(1).strip()
    # Remove possible leading/trailing quotes or brackets.
    raw_str = re.sub(r"^[\"\[\']+", "", raw_str)
    raw_str = re.sub(r"[\"\]\']+$", "", raw_str)

    # Parse out valid UCI moves like e2e4, e7e8q, etc.
    return sorted(re.findall(r"\b([a-h][1-8][a-h][1-8][qrnb]?)\b", raw_str.lower()))


def wilson_score_interval(p: float, n: int) -> Tuple[float, float]:
    """
    Calculate the Wilson score interval for a binomial proportion.
    p = proportion of successes (0..1),
    n = total trials.
    Returns (lower_bound, upper_bound).
    """
    if n == 0:
        return (0.0, 0.0)
    z = CONFIDENCE_LEVEL
    denominator = 1 + (z**2 / n)
    centre = (p + (z**2 / (2 * n))) / denominator
    adjustment = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * (n**2)))) / denominator
    return (max(0.0, centre - adjustment), min(1.0, centre + adjustment))


def process_position(
    fen: str,
    correct_moves: list,
    results_queue: queue.Queue,
    conv_queue: queue.Queue,
    timeout: float,
    model: str,
    stream: bool,
) -> bool:
    """
    Synchronously process a single position with up to MAX_RETRIES on exceptions.
    Returns True if LLM's parsed moves == correct moves, False otherwise.

    If all attempts fail (exception), this raises an exception so we can log properly.
    """
    prompt = (
        f"Given the chess position in FEN notation: {fen}\n"
        "List all legal moves in UCI format, comma-separated.\n"
        "Respond exactly with: 'Legal moves: [move1,move2,...]' and nothing else.\n"
    )

    for attempt in range(MAX_RETRIES):
        try:
            full_response = ""
            stream_buffer = ""

            # For the first attempt, if streaming, show a heading
            if stream and attempt == 0:
                print(f"\n\033[36mPosition\033[0m | FEN: {fen}")
                print("\033[33mLLM Thinking:\033[0m ", end="", flush=True)

            # Make the request via the old Completions endpoint
            if stream:
                # We read chunk by chunk
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    stream=True,
                    request_timeout=timeout,
                )
                # This is a generator in the old library
                for chunk in response:
                    content = chunk.choices[0].text or ""
                    full_response += content
                    stream_buffer += content
                    if len(stream_buffer) > 10 or "\n" in stream_buffer:
                        print(stream_buffer, end="", flush=True)
                        stream_buffer = ""
                print()  # final flush
            else:
                # Non-streaming: fetch the entire response
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    request_timeout=timeout,
                )
                full_response = response.choices[0].text

            # Parse the moves from the LLM's response
            parsed_moves = parse_llm_response(full_response)
            is_perfect = (parsed_moves == correct_moves)

            # Results log: 5 fields = FEN, LLM_MOVES, LEGAL_MOVES, PASS_FAIL, ERROR
            results_queue.put((fen, parsed_moves, correct_moves, is_perfect, ""))

            # Conversation log: 6 fields = PROMPT, FULL_RESPONSE, PARSED_MOVES, CORRECT_MOVES, PASS_FAIL, ERROR
            conv_queue.put((prompt, full_response, parsed_moves, correct_moves, is_perfect, ""))

            return is_perfect

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)  # minor delay before retry
                continue
            else:
                # Final attempt fails => log error and raise
                results_queue.put((fen, [], correct_moves, False, last_error))
                conv_queue.put((prompt, "", [], correct_moves, False, last_error))
                raise RuntimeError(last_error)


def log_writer(file_path: str, log_queue: queue.Queue, header: str):
    """
    Dedicated log writer thread.
    Reads from the queue until it encounters None, then exits.
    """
    with open(file_path, "w") as f:
        f.write(header)
        while True:
            item = log_queue.get()
            if item is None:
                break
            f.write("|".join(str(x) for x in item) + "\n")
            f.flush()
            log_queue.task_done()


def monitor_progress(futures, start_time):
    """
    Prints progress every second until all futures are complete.
    Shows # done, # wrong, # errors, and elapsed time.

    A "wrong" result is a future that completes without exception
    but returns False. An "error" is a future that raises an exception.
    """
    total = len(futures)
    while True:
        done_count = sum(f.done() for f in futures)
        error_count = sum(f.exception() is not None for f in futures if f.done())
        wrong_count = 0
        for f in futures:
            if f.done() and f.exception() is None:
                # No exception => safe to call f.result()
                if f.result() is False:
                    wrong_count += 1

        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            f"\rProcessed: {done_count}/{total} | Wrong: {wrong_count} | Errors: {error_count} | Elapsed: {elapsed:.1f}s",
            end="",
            flush=True,
        )

        if done_count >= total:
            break

        time.sleep(1)

    # Final newline
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="API base URL (LLM_API_BASE_URL env)")
    parser.add_argument("--api-key", help="API key (LLM_API_KEY env)")
    parser.add_argument("--model", help="Model name (LLM_MODEL env)")
    parser.add_argument("--results-log", help="Results log path")
    parser.add_argument("--conversation-log", help="Conversation log path")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--positions",
        type=int,
        default=DEFAULT_POSITIONS,
        help="Number of positions to generate (default=200)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (requires concurrency=1)",
    )

    args = parser.parse_args()

    # Validate streaming vs concurrency
    if args.stream and args.concurrency != 1:
        print("Error: Streaming mode requires concurrency=1")
        return

    # Gather credentials from args or environment
    base_url = args.base_url or os.getenv("LLM_API_BASE_URL")
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    model = args.model or os.getenv("LLM_MODEL")

    if not all([base_url, api_key, model]):
        print("Error: Missing required parameters (base URL, API key, model).")
        return

    # Configure openai to use the old completions endpoint base
    openai.api_key = api_key
    openai.api_base = base_url

    # Generate positions
    positions = generate_positions(num_positions=args.positions)

    # Prepare logging queues & threads
    results_queue = queue.Queue()
    conv_queue = queue.Queue()
    loggers = []

    if args.results_log:
        t = threading.Thread(
            target=log_writer,
            args=(
                args.results_log,
                results_queue,
                "FEN|LLM_MOVES|LEGAL_MOVES|PASS_FAIL|ERROR\n",
            ),
        )
        t.start()
        loggers.append(t)

    if args.conversation_log:
        t = threading.Thread(
            target=log_writer,
            args=(
                args.conversation_log,
                conv_queue,
                "PROMPT|FULL_RESPONSE|PARSED_MOVES|CORRECT_MOVES|PASS_FAIL|ERROR\n",
            ),
        )
        t.start()
        loggers.append(t)

    start_time = datetime.now()

    # We'll store results (True/False or exceptions) so we can compute accuracy
    results_list = []

    # Streaming mode => single-threaded, just iterate over positions
    # Because concurrency=1 in streaming mode
    if args.stream:
        count = 0
        for fen, moves in positions:
            try:
                is_correct = process_position(
                    fen=fen,
                    correct_moves=moves,
                    results_queue=results_queue,
                    conv_queue=conv_queue,
                    timeout=args.request_timeout,
                    model=model,
                    stream=True,
                )
                results_list.append(is_correct)
            except Exception:
                # Count as an error
                results_list.append(RuntimeError("Position failed."))
            count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            wrong_count = sum(r is False for r in results_list if not isinstance(r, Exception))
            error_count = sum(isinstance(r, Exception) for r in results_list)
            print(
                f"\rProcessed: {count}/{len(positions)} | Wrong: {wrong_count} | Errors: {error_count} | Elapsed: {elapsed:.1f}s",
                end="",
                flush=True,
            )
        print()  # final newline
    else:
        # Non-streaming => we can use a thread pool for concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for fen, moves in positions:
                futures.append(
                    executor.submit(
                        process_position,
                        fen,
                        moves,
                        results_queue,
                        conv_queue,
                        args.request_timeout,
                        model,
                        False,  # stream=False
                    )
                )

            # Start a progress monitor in the main thread
            monitor_progress(futures, start_time)

        # Collect final results from futures
        for f in futures:
            exc = f.exception()
            if exc is not None:
                results_list.append(exc)  # an Exception
            else:
                results_list.append(f.result())  # True or False

    # Cleanly stop log threads
    results_queue.put(None)
    conv_queue.put(None)
    for t in loggers:
        t.join()

    # Compute final stats
    total = len(positions)
    correct_count = sum((r is True) for r in results_list if not isinstance(r, Exception))
    accuracy = correct_count / total if total else 0.0

    low, high = wilson_score_interval(accuracy, total)
    margin = (high - low) / 2
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\nProcessed {total} positions in {elapsed:.1f} seconds.")
    print(f"Accuracy: {accuracy:.1%} ± {margin:.1%} (95% CI)")
    print(f"Confidence interval: {low:.1%} - {high:.1%}")


if __name__ == "__main__":
    main()
