import asyncio
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
from openai import AsyncOpenAI

# Configuration defaults
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_POSITIONS = 200  # default to 200 positions
CONFIDENCE_LEVEL = 1.96  # ~95% confidence intervals


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
    # - Leading: any combo of double quote, single quote, or '['
    # - Trailing: any combo of double quote, single quote, or ']'
    raw_str = re.sub(r"^[\"\[\']+", "", raw_str)
    raw_str = re.sub(r"[\"\]\']+$", "", raw_str)

    # Now parse out moves such as "e2e4", "e7e8q", etc.
    # We'll ignore punctuation between them; re.findall captures valid UCI moves.
    return sorted(re.findall(r"\b([a-h][1-8][a-h][1-8][qrnb]?)\b", raw_str.lower()))


def wilson_score_interval(p: float, n: int) -> Tuple[float, float]:
    """
    Calculate the Wilson score interval for a binomial proportion.
    p = proportion of successes (0..1),
    n = total trials,
    Returns (lower_bound, upper_bound).
    """
    if n == 0:
        return (0.0, 0.0)
    z = CONFIDENCE_LEVEL
    denominator = 1 + (z**2 / n)
    centre = (p + (z**2 / (2 * n))) / denominator
    adjustment = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * (n**2)))) / denominator
    return (max(0.0, centre - adjustment), min(1.0, centre + adjustment))


async def process_position(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    fen: str,
    correct_moves: list,
    results_queue: queue.Queue,
    conv_queue: queue.Queue,
    timeout: float,
    model: str,
    stream: bool,
) -> bool:
    """
    Process a single position with optional streaming support.
    Returns True if the LLM's parsed moves == correct moves; else False.
    """
    prompt = (
        f"Given the chess position in FEN notation: {fen}\n"
        "List all legal moves in UCI format, comma-separated.\n"
        "Respond exactly with: 'Legal moves: [move1,move2,...]' and nothing else.\n"
    )

    try:
        async with sem:
            full_response = ""
            stream_buffer = ""

            if stream:
                # Simple heading for streaming output
                print(f"\n\033[36mPosition\033[0m | FEN: {fen}")
                print("\033[33mLLM Thinking:\033[0m ", end="", flush=True)

            # The 'timeout' here is for the OpenAI client library
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                timeout=timeout,  # Belongs to the OpenAI call
            )

            if stream:
                # If streaming, accumulate partial chunks
                async for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    stream_buffer += content

                    # Just a simple flush every so often or if newline
                    if len(stream_buffer) > 10 or "\n" in stream_buffer:
                        print(stream_buffer, end="", flush=True)
                        stream_buffer = ""
                # Final flush
                print()
            else:
                # Non-streaming: read the entire response at once
                full_response = response.choices[0].message.content

            parsed_moves = parse_llm_response(full_response)
            is_perfect = parsed_moves == correct_moves

            # Results log: 5 fields = FEN, LLM_MOVES, LEGAL_MOVES, PASS_FAIL, ERROR
            results_queue.put((fen, parsed_moves, correct_moves, is_perfect, ""))
            # Conversation log: 6 fields = PROMPT, FULL_RESPONSE, PARSED_MOVES, CORRECT_MOVES, PASS_FAIL, ERROR
            conv_queue.put(
                (prompt, full_response, parsed_moves, correct_moves, is_perfect, "")
            )

            return is_perfect

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        results_queue.put((fen, [], correct_moves, False, error_msg))
        conv_queue.put((prompt, "", [], correct_moves, False, error_msg))
        return False


def log_writer(file_path: str, log_queue: queue.Queue, header: str):
    """
    Dedicated log writer thread.
    Reads from the queue until it encounters None, then exits.

    Because we use a single thread and a queue, no additional lock is needed.
    The queue provides the necessary synchronization.
    """
    with open(file_path, "w") as f:
        f.write(header)
        while True:
            item = log_queue.get()
            if item is None:
                break
            f.write("|".join(str(x) for x in item) + "\n")
            # Write each line to disk immediately
            f.flush()
            log_queue.task_done()


def is_error(task: asyncio.Task) -> bool:
    """
    Helper function for progress monitoring: returns True if
    the task is done AND either raised an exception or returned False.
    """
    if not task.done():
        return False
    ex = task.exception()
    if ex is not None:
        return True
    return task.result() is False


async def monitor_progress(tasks: list[asyncio.Task], start_time: datetime):
    """
    Prints progress every second until all tasks have completed.
    """
    total = len(tasks)
    while True:
        done_count = sum(t.done() for t in tasks)
        error_count = sum(is_error(t) for t in tasks)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            f"\rProcessed: {done_count}/{total} | Errors: {error_count} | Elapsed: {elapsed:.1f}s",
            end="",
            flush=True,
        )

        if done_count >= total:
            break

        await asyncio.sleep(1)

    # Print a final newline for cleanliness
    print()


async def main_async(args):
    """
    Main async workflow.
    If --stream is enabled, we skip the progress display.
    Otherwise, we show progress updates.
    """

    # Validate streaming vs concurrency
    if args.stream and args.concurrency != 1:
        raise ValueError("Streaming mode requires concurrency=1")

    # Gather credentials from args or environment
    base_url = args.base_url or os.getenv("LLM_API_BASE_URL")
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    model = args.model or os.getenv("LLM_MODEL")

    if not all([base_url, api_key, model]):
        raise ValueError(
            "Missing required parameters. Need base URL, API key, and model name."
        )

    # Initialize client
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Generate positions (configurable via --positions)
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

    # Build async tasks
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            process_position(
                client=client,
                sem=sem,
                fen=fen,
                correct_moves=moves,
                results_queue=results_queue,
                conv_queue=conv_queue,
                timeout=args.request_timeout,
                model=model,
                stream=args.stream,
            )
        )
        for fen, moves in positions
    ]

    start_time = datetime.now()

    if args.stream:
        # Streaming mode: just run tasks, no progress monitor
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Non-streaming: start progress monitor and run tasks
        progress_task = asyncio.create_task(monitor_progress(tasks, start_time))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await progress_task

    # Cleanly stop log threads
    results_queue.put(None)
    conv_queue.put(None)

    for t in loggers:
        t.join()

    # Calculate final stats
    total = len(positions)
    perfect_count = sum((r is True) for r in results if not isinstance(r, Exception))
    accuracy = perfect_count / total if total else 0.0

    # Compute confidence interval
    low, high = wilson_score_interval(accuracy, total)
    margin = (high - low) / 2

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nProcessed {total} positions in {elapsed:.1f} seconds.")
    print(f"Accuracy: {accuracy:.2%} ± {margin:.3%} (95% CI)")
    print(f"Confidence interval: {low:.2%} - {high:.2%}")


if __name__ == "__main__":
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

    try:
        asyncio.run(main_async(args))
    except ValueError as e:
        print(f"Error: {str(e)}")
