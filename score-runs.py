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
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib to use non-GUI backend if needed
plt.switch_backend("Agg")  # Remove this if you want interactive plots

Z_VALUE = 1.96  # for ~95% confidence

# Plot styling
plt.style.use("ggplot")
# COLOR_EXACT = "#2ca02c"
# COLOR_F1 = "#1f77b4"
# ERROR_BAR_COLOR = "#d62728"
ERROR_BAR_WIDTH = 0.8

COLOR_F1 = "#1f77b4"  # Example color for F1 bars
COLOR_EXACT = "#ff7f0e"  # Example color for Exact Match bars
ERROR_BAR_COLOR = "black"


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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 0.0
    if (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    exact_match = m_pred == m_gt
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
        gt_str = parts[2].strip()  # LEGAL_MOVES
        pass_fail = parts[3].strip()
        error_str = parts[4].strip()

        # Optionally skip lines that have an error
        # if error_str:
        #     continue

        m_pred = parse_moves(llm_str)
        m_gt = parse_moves(gt_str)

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
    sq_diffs = [(x - mean_f1) ** 2 for x in f1_values]
    var_f1 = sum(sq_diffs) / (n - 1)
    std_f1 = math.sqrt(var_f1)

    margin = Z_VALUE * (std_f1 / math.sqrt(n))
    lower = max(0.0, mean_f1 - margin)
    upper = min(1.0, mean_f1 + margin)
    return (lower, upper)


def plot_leaderboard(results, output_file="leaderboard.png"):
    """Generate a horizontal bar chart visualization of the results."""
    # Sort results by F1 score (descending)
    sorted_results = sorted(results, key=lambda x: x["avg_f1"], reverse=True)

    # Prepare data for plotting
    labels = [
        os.path.basename(r["file"]).replace(".results", "") for r in sorted_results
    ]
    f1_scores = [r["avg_f1"] * 100 for r in sorted_results]
    ci_lower = [(r["avg_f1"] - r["ci_low"]) * 100 for r in sorted_results]
    ci_upper = [(r["ci_high"] - r["avg_f1"]) * 100 for r in sorted_results]
    exact_pct = [r["exact_matches"] / r["positions"] * 100 for r in sorted_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Positions along the y-axis
    y = np.arange(len(labels))
    height = 0.4  # Thickness of each group (F1 bar + Exact bar)

    # Plot F1 scores with error bars (horizontal bars)
    rects1 = ax.barh(
        y=y - (height / 4),  # shift upward for side-by-side grouping
        width=f1_scores,
        height=height / 2,
        label="Macro-average F1",
        color=COLOR_F1,
        xerr=[ci_lower, ci_upper],  # horizontal error bars
        error_kw={"ecolor": ERROR_BAR_COLOR, "capsize": 4},
    )

    # Plot exact match percentages (horizontal bars)
    rects2 = ax.barh(
        y=y + (height / 4),  # shift downward for side-by-side grouping
        width=exact_pct,
        height=height / 2,
        label="Exact Match %",
        color=COLOR_EXACT,
    )

    # Add labels, title, and custom y-axis tick labels
    ax.set_xlabel("Percentage")
    ax.set_title("Random-Chess LLM Leaderboard")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()

    # Invert y-axis so that the best scorer is at the top
    ax.invert_yaxis()

    # Function to add labels to the right of each bar, avoiding overlap with error bars
    def autolabel_horizontal(rects):
        for rect in rects:
            width = rect.get_width()
            y_pos = rect.get_y() + rect.get_height() / 2
            # Determine offset based on data series
            if "Macro-average F1" in str(rect.get_label()):  # F1 bars
                offset_x = 15
                offset_y = 3  # Shift F1 labels slightly up
            else:  # Exact match bars
                offset_x = 5
                offset_y = -3  # Shift exact match labels slightly down

            ax.annotate(
                f"{width:.1f}%",
                xy=(width, y_pos),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
            )

    autolabel_horizontal(rects1)
    autolabel_horizontal(rects2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nSaved leaderboard plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute scores and plot leaderboard from results logs."
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        required=True,
        help="Paths to results log files",
    )
    parser.add_argument(
        "--plot",
        "-p",
        default="leaderboard.png",
        help="Output filename for leaderboard plot",
    )
    args = parser.parse_args()

    # Process files
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

    # Generate text report
    results.sort(key=lambda r: (r["avg_f1"], r["exact_matches"]), reverse=True)

    print("\nScoring Summary (macro-average F1, tie-break by exact matches)")
    print("(95% CI computed by normal approximation)")
    print()
    print("| Rank | File                     | #Pos |    F1(%)         | Exact% |")
    print("|------|--------------------------|------|------------------|--------|")

    for rank, r in enumerate(results, 1):
        file_name = os.path.basename(r["file"])
        avg_f1_pct = r["avg_f1"] * 100
        ci_low_pct = r["ci_low"] * 100
        ci_high_pct = r["ci_high"] * 100
        ci_plusminus = (ci_high_pct - ci_low_pct) / 2.0
        exact_pct = r["exact_matches"] / r["positions"] * 100

        f1_str = f"{avg_f1_pct:>5.1f}% ± {ci_plusminus:>4.1f}%"
        print(
            f"| {rank:>4} "
            f"| {file_name:<24} "
            f"| {r['positions']:<4} "
            f"| {f1_str:<16} "
            f"| {exact_pct:>5.1f}% |"
        )

    # Generate plot
    plot_leaderboard(results, args.plot)


if __name__ == "__main__":
    main()
