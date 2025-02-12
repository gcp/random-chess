#!/usr/bin/env python3

import asyncio
import chess
import chess.engine
import random
import re
import math
import os
import argparse
import threading
import queue
from datetime import datetime
from typing import Tuple, Optional, List

# Using the OpenAI async client
from openai import AsyncOpenAI

# ------------------------------------------------------------------------------------
# CONFIG DEFAULTS
# ------------------------------------------------------------------------------------
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_MATE_POSITIONS = 100
DEFAULT_NONMATE_POSITIONS = 100
CONFIDENCE_LEVEL = 1.96  # ~95% confidence intervals
MAX_RETRIES = 3

# Adjust the path to your Stockfish binary if needed
STOCKFISH_PATH = "./stockfish"

# ------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------------


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
    return (
        max(0.0, centre - adjustment),
        min(1.0, centre + adjustment),
    )


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


def is_error(task: asyncio.Task) -> bool:
    """
    Returns True if this task ended with an exception.
    """
    if not task.done():
        return False
    return task.exception() is not None


def is_wrong(task: asyncio.Task) -> bool:
    """
    Return True if the task completed (no exception) but returned False,
    i.e. the LLM gave an answer but got it wrong.
    """
    if not task.done():
        return False
    if task.exception() is not None:
        return False
    return task.result() is False


# ------------------------------------------------------------------------------------
# RANDOM BOARD GENERATION
# ------------------------------------------------------------------------------------


def generate_random_position() -> chess.Board:
    """
    Generate a random position by starting from the initial board
    and making between 10 and 80 random moves.
    Bias the move generation a bit to get a more sensible position.
    Return a board that is not yet game-over.
    """
    board = chess.Board()
    num_moves = random.randint(10, 80)

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # Weight = 1 if from-square is a pawn, else 4
        weights = []
        for mv in legal_moves:
            piece = board.piece_at(mv.from_square)
            if piece and piece.piece_type == chess.PAWN:
                weights.append(2)
            elif piece and piece.piece_type == chess.KING:
                if mv.to_square in (
                    chess.G1,
                    chess.C1,
                    chess.G8,
                    chess.C8,
                ) and mv.from_square in (chess.E1, chess.E8):
                    weights.append(10)
                else:
                    weights.append(1)
            else:
                weights.append(5)

        move = random.choices(legal_moves, weights=weights, k=1)[0]
        board.push(move)
        if board.is_game_over():
            break

    # If it's still not game-over, we have a candidate
    return board


# ------------------------------------------------------------------------------------
# STOCKFISH EVALUATION
# ------------------------------------------------------------------------------------


def find_unique_mate_in_2(
    board: chess.Board, engine, multipv, depth
) -> Optional[chess.Move]:
    """
    Use Stockfish in multi-PV mode to see if there's exactly one forced mate in 2
    for the side to move. We require:
      1) The best line (info[0]) is mate in 2.
      2) None of the other lines is mate in 2.
    If that condition is met, return the best move from that line; else return None.
    """
    info_list = engine.analyse(
        board, limit=chess.engine.Limit(depth=depth), multipv=multipv
    )

    if not info_list:
        return None

    # Check the top line
    best_line = info_list[0]
    best_score = best_line["score"].relative

    if not best_score.is_mate() or best_score.mate() != 2:
        return None

    # We want to be sure no other line indicates mate in 2
    for i, line_info in enumerate(info_list[1:], start=1):
        score = line_info["score"].relative
        if score.is_mate() and score.mate() == 2:
            # Another line is also mate in 2 => not unique
            return None

    # If we get here, the best line is mate in 2, and no other line is
    if "pv" not in best_line or not best_line["pv"]:
        return None
    # Return the first move in that PV
    return best_line["pv"][0]


def no_mate_in_2(board: chess.Board, engine, multipv, depth) -> bool:
    """
    Check that *none* of the top multi-PV lines produce a mate in 2 for the side to move.
    Return True if we confirm no mate in 2, else False.
    """
    info_list = engine.analyse(
        board, limit=chess.engine.Limit(depth=depth), multipv=multipv
    )
    for info in info_list:
        score = info["score"].relative
        if score.is_mate() and score.mate() == 2:
            return False
    return True


# ------------------------------------------------------------------------------------
# BUILD DATASETS: MATE-IN-2 AND NON-MATE-IN-2
# ------------------------------------------------------------------------------------


def generate_unique_mate_in_2_positions(num_positions=100) -> List[tuple]:
    """
    Generate 'num_positions' boards where exactly ONE move leads to a forced mate in 2.
    Return (fen, True, move_uci).
    """
    results = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        while len(results) < num_positions:
            board = generate_random_position()
            if board.is_game_over():
                continue

            move = find_unique_mate_in_2(board, engine, multipv=256, depth=9)
            if move is not None:
                fen = board.fen()
                results.append((fen, True, move.uci()))
                print(
                    f"Mate in 2 ({len(results)}/{num_positions}): {fen} => {move.uci()}"
                )

    return results


def generate_non_mate_in_2_positions(num_positions=100) -> List[tuple]:
    """
    Generate 'num_positions' boards that do NOT have any mate in 2 among top lines.
    Return (fen, False, None).
    """
    results = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        while len(results) < num_positions:
            board = generate_random_position()
            if board.is_game_over():
                continue

            if no_mate_in_2(board, engine, multipv=256, depth=9):
                fen = board.fen()
                results.append((fen, False, None))

    return results


# ------------------------------------------------------------------------------------
# PARSING LLM RESPONSES
# ------------------------------------------------------------------------------------


def parse_llm_mate_in_2_answer(response: str) -> tuple:
    """
    Expect either:
      "Answer: No"
    or
      "Answer: Yes, d1d7"

    Return (is_yes, move_uci_or_None), e.g. (True, 'd1d7') or (False, None).
    Return (None, None) if we can't parse.
    """
    match = re.search(
        r"(?i)Answer\s*:\s*(Yes|No)\s*(?:,\s*([a-h][1-8][a-h][1-8][qrnb]?))?", response
    )
    if not match:
        return (None, None)

    yesno = match.group(1).strip().lower()
    move_uci = match.group(2)  # might be None if user said "No"
    is_yes = yesno == "yes"

    return (is_yes, move_uci)


# ------------------------------------------------------------------------------------
# ASYNC LLM PROCESS
# ------------------------------------------------------------------------------------


async def process_position(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    fen: str,
    is_mate_in_2: bool,
    best_move: Optional[str],
    results_queue: queue.Queue,
    conv_queue: queue.Queue,
    timeout: float,
    model: str,
    stream: bool,
) -> bool:
    """
    Ask the LLM if it's mate in 2. If yes, provide the best move. If no, answer "Answer: No".
    We'll do up to MAX_RETRIES for network exceptions.

    Return True if LLM's answer is exactly correct, else False.
    """
    prompt = (
        f"Given this chess position in FEN notation: {fen}\n"
        "Consider whether it is mate in 2 for the side to move. If yes, respond exactly with:\n"
        "Answer: Yes, <uci_move>\n"
        "With <uci_move> being the first move in the mating sequence, in UCI notation.\n"
        "Otherwise, respond exactly with:\n"
        "Answer: No\n"
        "Do not reply with anything else.\n"
    )

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                full_response = ""

                if stream and attempt == 0:
                    print(f"\n\033[36mPosition\033[0m | FEN: {fen}")
                    print("\033[33mLLM Thinking:\033[0m ", end="", flush=True)

                # Call OpenAI with streaming or non-streaming
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=stream,
                    timeout=timeout,
                )

                if stream:
                    # Streaming mode
                    stream_buffer = ""
                    async for chunk in response:
                        content = getattr(chunk.choices[0].delta, "content", "")
                        full_response += content
                        stream_buffer += content
                        if len(stream_buffer) > 10 or "\n" in stream_buffer:
                            print(stream_buffer, end="", flush=True)
                            stream_buffer = ""
                    print(stream_buffer, end="", flush=True)
                    print()
                else:
                    # Non-stream
                    full_response = response.choices[0].message.content

                # Parse the answer
                predicted_is_mate, predicted_move = parse_llm_mate_in_2_answer(
                    full_response
                )

                # Decide correctness
                if predicted_is_mate is None:
                    correct = False
                else:
                    if is_mate_in_2:
                        # Must say "Yes, <best_move>"
                        correct = predicted_is_mate and predicted_move == best_move
                    else:
                        # Must say "No"
                        correct = not predicted_is_mate

                # Log
                error_msg = ""
                results_queue.put(
                    (
                        fen,
                        is_mate_in_2,
                        best_move if best_move else "",
                        full_response.replace("\n", "\\n"),
                        correct,
                        error_msg,
                    )
                )
                conv_queue.put(
                    (
                        prompt.replace("\n", "\\n"),
                        full_response.replace("\n", "\\n"),
                        str(predicted_is_mate),
                        predicted_move if predicted_move else "",
                        str(is_mate_in_2),
                        best_move if best_move else "",
                        correct,
                        error_msg,
                    )
                )

                return correct

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    # Log final failure
                    results_queue.put(
                        (
                            fen,
                            is_mate_in_2,
                            best_move if best_move else "",
                            "",
                            False,
                            last_error,
                        )
                    )
                    conv_queue.put(
                        (
                            prompt.replace("\n", "\\n"),
                            "",
                            "",
                            "",
                            str(is_mate_in_2),
                            best_move if best_move else "",
                            False,
                            last_error,
                        )
                    )
                    raise RuntimeError(last_error)


# ------------------------------------------------------------------------------------
# PROGRESS MONITOR
# ------------------------------------------------------------------------------------


async def monitor_progress(tasks: List[asyncio.Task], start_time: datetime):
    """
    Prints progress every second until all tasks have completed.
    """
    total = len(tasks)
    while True:
        done_count = sum(t.done() for t in tasks)
        error_count = sum(is_error(t) for t in tasks)
        wrong_count = sum(is_wrong(t) for t in tasks)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            f"\rProcessed: {done_count}/{total} | Wrong: {wrong_count} | Errors: {error_count} | Elapsed: {elapsed:.1f}s",
            end="",
            flush=True,
        )

        if done_count >= total:
            break
        await asyncio.sleep(1)

    print()  # final newline


# ------------------------------------------------------------------------------------
# MAIN ASYNC
# ------------------------------------------------------------------------------------


async def main_async(args):
    """
    Main async workflow:
      1) Generate 100 unique mate-in-2 positions, 100 non-mate
      2) For each position, prompt LLM
      3) Log, measure accuracy
    """
    # Validate streaming vs concurrency
    if args.stream and args.concurrency != 1:
        raise ValueError("Streaming mode requires concurrency=1")

    # Gather credentials from args or environment
    base_url = args.base_url or os.getenv("LLM_API_BASE_URL")
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    model = args.model or os.getenv("LLM_MODEL")

    if not all([base_url, api_key, model]):
        raise ValueError("Missing required parameters (base_url, api_key, model).")

    # 1) Generate positions
    print("Generating unique mate-in-2 positions...")
    mate_positions = generate_unique_mate_in_2_positions(
        num_positions=DEFAULT_MATE_POSITIONS
    )
    print(f"  Found {len(mate_positions)} mate-in-2 positions.")

    print("Generating non-mate-in-2 positions...")
    nonmate_positions = generate_non_mate_in_2_positions(
        num_positions=DEFAULT_NONMATE_POSITIONS
    )
    print(f"  Found {len(nonmate_positions)} non-mate-in-2 positions.")

    dataset = mate_positions + nonmate_positions
    random.shuffle(dataset)

    # 2) Prepare logging
    results_queue = queue.Queue()
    conv_queue = queue.Queue()
    loggers = []

    if args.results_log:
        t = threading.Thread(
            target=log_writer,
            args=(
                args.results_log,
                results_queue,
                "FEN|IS_MATE_IN_2|BEST_MOVE|LLM_RESPONSE|CORRECT|ERROR\n",
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
                "PROMPT|LLM_RESPONSE|PREDICTED_IS_MATE|PREDICTED_MOVE|TRUE_IS_MATE|TRUE_MOVE|CORRECT|ERROR\n",
            ),
        )
        t.start()
        loggers.append(t)

    # 3) Build the OpenAI async client
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for fen, is_mate, move_uci in dataset:
        tasks.append(
            asyncio.create_task(
                process_position(
                    client=client,
                    sem=sem,
                    fen=fen,
                    is_mate_in_2=is_mate,
                    best_move=move_uci,
                    results_queue=results_queue,
                    conv_queue=conv_queue,
                    timeout=args.request_timeout,
                    model=model,
                    stream=args.stream,
                )
            )
        )

    # 4) Run tasks
    start_time = datetime.now()
    if args.stream:
        # Streaming => no progress monitor
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Non-stream => show progress
        progress_task = asyncio.create_task(monitor_progress(tasks, start_time))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await progress_task

    # Stop log threads
    results_queue.put(None)
    conv_queue.put(None)
    for t in loggers:
        t.join()

    # 5) Compute accuracy
    total = len(dataset)
    correct_count = sum((r is True) for r in results if not isinstance(r, Exception))
    accuracy = correct_count / total if total else 0.0

    low, high = wilson_score_interval(accuracy, total)
    margin = (high - low) / 2
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\nProcessed {total} positions in {elapsed:.1f} seconds.")
    print(f"Accuracy: {accuracy:.1%} ± {margin:.1%} (95% CI)")
    print(f"Confidence interval: {low:.1%} - {high:.1%}")


# ------------------------------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------------------------------

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
        "--stream",
        action="store_true",
        help="Enable streaming mode (requires concurrency=1)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except ValueError as e:
        print(f"Error: {str(e)}")
