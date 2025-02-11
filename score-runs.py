#!/usr/bin/env python3

"""
score_runs.py

Given one or more 'results' log files from the LLM chess script,
compute the macro-average F1 score (precision/recall on moves)
and rank them. Tie-break by total exact matches.

Now also compute a 95% confidence interval on the macro-average F1
using a normal approximation: mean +/- 1.96*(std/sqrt(n)).

Usage:
  python score_runs.py --files run1.results run2.results ...

Logs are expected to have lines like:
  FEN|LLM_MOVES|LEGAL_MOVES|PASS_FAIL|ERROR

Where:
  - LLM_MOVES: a string representation of the list of moves returned by the LLM
  - LEGAL_MOVES: a string representation of the list of ground-truth moves
  - We treat these as sets to compute TP, FP, FN.

HTML formula reference (pseudo-code):
  TP = |M_pred ∩ M_gt|
  FP = |M_pred - M_gt|
  FN = |M_gt - M_pred|
  precision = TP / (TP+FP)
  recall    = TP / (TP+FN)
  F1 = 2 * precision*recall / (precision + recall)

We compute these per position, then average (macro-average) across all positions
in each log. The best LLM is the one with the highest F1. Ties are broken by
who has more exact matches (M_pred == M_gt).

Now also computing:
  mean_F1 ± margin,  where margin = 1.96 * (std_F1 / sqrt(n))
"""

import argparse
import sys
import os
import math

Z_VALUE = 1.96  # for ~95% confidence

def parse_moves(moves_str):
    """
    Convert a string like "['e2e4', 'd2d4']" or "['e2e4','d2d4']" into a Python set.
    We'll do a naive parse by stripping brackets and splitting on commas.
    Adapt as needed if your logs have a different format.
    """
    moves_str = moves_str.strip()
    # Remove leading/trailing brackets
    if moves_str.startswith("["):
        moves_str = moves_str[1:]
    if moves_str.endswith("]"):
        moves_str = moves_str[:-1]

    # If it's empty or something, return empty set
    if not moves_str.strip():
        return set()

    # Now split on commas
    raw_moves = moves_str.split(",")
    # Strip quotes/spaces from each
    moves = []
    for m in raw_moves:
        m_clean = m.strip().strip("'").strip('"').strip()
        if m_clean:
            moves.append(m_clean)

    return set(moves)


def compute_f1(m_pred, m_gt):
    """
    Given two sets of moves (m_pred, m_gt), compute precision, recall, F1.
    Return (precision, recall, f1, exact_match_bool).
    """
    tp = len(m_pred.intersection(m_gt))
    fp = len(m_pred - m_gt)
    fn = len(m_gt - m_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 0.0
    if (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    exact_match = (m_pred == m_gt)
    return (precision, recall, f1, exact_match)


def process_log_file(fpath):
    """
    Parse the results log:
      FEN|LLM_MOVES|LEGAL_MOVES|PASS_FAIL|ERROR

    Compute per-position F1, then return (avg_f1, count, exact_count, f1_list).
      - avg_f1 = macro-average F1
      - count = number of positions processed
      - exact_count = how many times M_pred == M_gt
      - f1_list = list of each position's F1 (for confidence calc)
    """
    if not os.path.isfile(fpath):
        print(f"Warning: file not found: {fpath}", file=sys.stderr)
        return (0.0, 0, 0, [])

    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header line (assuming the first line is the header)
    if not lines:
        print(f"Warning: empty file: {fpath}", file=sys.stderr)
        return (0.0, 0, 0, [])

    # Typically: FEN|LLM_MOVES|LEGAL_MOVES|PASS_FAIL|ERROR
    # But the 1st line might be the column header
    data_lines = lines[1:]  # everything after the header

    f1_values = []
    exact_count = 0

    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 5:
            # Possibly an incomplete line?
            continue

        # FEN    = parts[0]
        llm_str = parts[1].strip()  # LLM_MOVES
        gt_str  = parts[2].strip()  # LEGAL_MOVES
        pass_fail = parts[3].strip()
        error_str = parts[4].strip()

        # Optionally skip lines that have an error
        # if error_str:
        #     continue

        m_pred = parse_moves(llm_str)
        m_gt   = parse_moves(gt_str)

        precision, recall, f1, exact_match = compute_f1(m_pred, m_gt)

        f1_values.append(f1)
        if exact_match:
            exact_count += 1

    count = len(f1_values)
    if count == 0:
        return (0.0, 0, 0, [])

    avg_f1 = sum(f1_values) / count
    return (avg_f1, count, exact_count, f1_values)


def f1_confidence_interval(f1_values):
    """
    Compute a 95% confidence interval for the mean of a list of F1 values
    using the normal approximation: mean +/- 1.96 * (std / sqrt(n)).

    Returns (lower, upper).
    """
    n = len(f1_values)
    if n <= 1:
        if n == 0:
            return (0.0, 0.0)
        # If there's exactly 1 F1 value, we have no variation
        val = f1_values[0]
        return (val, val)

    mean_f1 = sum(f1_values) / n
    # sample standard deviation
    sq_diffs = [(x - mean_f1)**2 for x in f1_values]
    var_f1 = sum(sq_diffs) / (n - 1)
    std_f1 = math.sqrt(var_f1)

    margin = Z_VALUE * (std_f1 / math.sqrt(n))
    lower = max(0.0, mean_f1 - margin)
    upper = min(1.0, mean_f1 + margin)
    return (lower, upper)


def main():
    parser = argparse.ArgumentParser(
        description="Compute macro-average F1 from one or more results log files."
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        required=True,
        help="Paths to results log files from the chess-moves script.",
    )
    args = parser.parse_args()

    # We'll compute (avg_f1, position_count, exact_matches, f1_values) for each file
    # Then rank them by F1, tie-break by exact matches
    results = []
    for fpath in args.files:
        avg_f1, count, exact_count, f1_list = process_log_file(fpath)
        (ci_low, ci_high) = f1_confidence_interval(f1_list)
        results.append(
            {
                "file": fpath,
                "avg_f1": avg_f1,
                "positions": count,
                "exact_matches": exact_count,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    # Sort by descending F1, then descending exact_matches
    # (If you wanted to incorporate the interval, you'd have to define more logic)
    results.sort(key=lambda r: (r["avg_f1"], r["exact_matches"]), reverse=True)

    print()
    print("Scoring Summary (macro-average F1, tie-break by exact matches)")
    print("(95% CI computed by normal approximation)")
    print()
    print("| Rank | File                     | #Pos |    F1(%)         | Exact% |")
    print("|------|--------------------------|------|------------------|--------|")

    rank = 1
    for r in results:
        file_name = os.path.basename(r["file"])
        avg_f1_pct = r["avg_f1"] * 100
        ci_low_pct = r["ci_low"] * 100
        ci_high_pct = r["ci_high"] * 100
        ci_plusminus = (ci_high_pct - ci_low_pct) / 2.0
        exact_pct = r["exact_matches"] / r["positions"] * 100

        # We'll display: F1%  [L% - U%]
        # with 1 decimal place each
        f1_str = f"{avg_f1_pct:>5.1f}% ± {ci_plusminus:>4.1f}%"

        print(
            f"| {rank:>4} "
            f"| {file_name:<24} "
            f"| {r['positions']:<4} "
            f"| {f1_str:<16} "
            f"| {exact_pct:>5.1f}% |"
        )
        rank += 1

    print()


if __name__ == "__main__":
    main()
