#!/usr/bin/env python3

"""
score_runs.py

Given log files in the form:
  FEN|IS_MATE_IN_2|BEST_MOVE|LLM_RESPONSE|CORRECT|ERROR

We compute:
  - Mate Accuracy       = # correct on mate positions / total mate positions
  - Non-mate Accuracy   = # correct on non-mate positions / total non-mate
  - Overall Accuracy    = (# correct overall) / (total positions)
  - Hallucination Rate  = # times LLM said "Yes" on a non-mate / total non-mate

We then:
  1) Print a text table showing each file's #mates, #nonmates, mate_acc, etc.
  2) Produce a horizontal bar chart with:
        - Mate Accuracy (%)
        - Hallucination Rate (%)
     side by side, reusing the old F1+Exact horizontal layout.

Usage:
  python score_runs.py --files run1.results run2.results ...
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# If no GUI is available, you can force a non-interactive backend:
plt.switch_backend("Agg")
plt.style.use("ggplot")

# Colors reused from old script for a nostalgic match:
COLOR_MATE = "#1f77b4"
COLOR_HALLUC = "#ff7f0e"


def parse_line(line):
    """
    Given a line like:
      FEN|IS_MATE_IN_2|BEST_MOVE|LLM_RESPONSE|CORRECT|ERROR
    Return a dict or None if parse fails.
    """
    parts = line.strip().split("|")
    if len(parts) < 6:
        return None

    fen = parts[0].strip()
    is_mate_str = parts[1].strip()
    best_move = parts[2].strip()
    llm_resp = parts[3].strip()  # e.g. "Answer: Yes, d4c5" or "Answer: No"
    correct_str = parts[4].strip()
    error_str = parts[5].strip()

    # Convert is_mate_str => bool
    if is_mate_str.lower() == "true":
        is_mate_in_2 = True
    elif is_mate_str.lower() == "false":
        is_mate_in_2 = False
    else:
        return None

    # Convert correct_str => bool
    if correct_str.lower() == "true":
        correct = True
    elif correct_str.lower() == "false":
        correct = False
    else:
        # If blank or unrecognized, treat as false
        correct = False

    return {
        "fen": fen,
        "is_mate_in_2": is_mate_in_2,
        "best_move": best_move,
        "llm_response": llm_resp,
        "correct": correct,
        "error": error_str,
    }


def said_yes(llm_response: str) -> bool:
    """
    Return True if the LLM response starts with "Answer: Yes"
    ignoring case.
    """
    return llm_response.lower().startswith("answer: yes")


def process_log_file(fpath):
    """
    Parse the file, compute:
      - mate_count
      - mate_correct_count
      - nonmate_count
      - nonmate_correct_count
      - halluc_count (times we saw "Answer: Yes" on a non-mate)
    Return a dict with these plus derived stats.
    """
    if not os.path.isfile(fpath):
        print(f"Warning: file not found: {fpath}")
        return None

    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        print(f"Warning: empty file: {fpath}")
        return None

    # Skip header
    data_lines = lines[1:]

    mate_count = 0
    mate_correct_count = 0
    nonmate_count = 0
    nonmate_correct_count = 0
    halluc_count = 0
    total = 0

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        rec = parse_line(line)
        if not rec:
            continue

        total += 1
        is_mate = rec["is_mate_in_2"]
        correct = rec["correct"]
        llm_resp = rec["llm_response"]
        # If there's an error, you could decide to skip or count it as incorrect, etc.

        if is_mate:
            mate_count += 1
            if correct:
                mate_correct_count += 1
        else:
            nonmate_count += 1
            if correct:
                nonmate_correct_count += 1
            if said_yes(llm_resp):
                halluc_count += 1

    mate_acc = mate_correct_count / mate_count if mate_count else 0.0
    nonmate_acc = nonmate_correct_count / nonmate_count if nonmate_count else 0.0
    overall_acc = (
        (mate_correct_count + nonmate_correct_count) / (mate_count + nonmate_count)
        if (mate_count + nonmate_count)
        else 0.0
    )
    halluc_rate = halluc_count / nonmate_count if nonmate_count else 0.0

    # Combined F1-style score: harmonic mean of mate accuracy and anti-hallucination
    anti_halluc = 1 - halluc_rate if halluc_rate else 1.0  # Prevent division by zero
    f1_score = (
        (2 * mate_acc * anti_halluc) / (mate_acc + anti_halluc)
        if (mate_acc + anti_halluc)
        else 0.0
    )

    return {
        "file": fpath,
        "mate_count": mate_count,
        "mate_correct": mate_correct_count,
        "nonmate_count": nonmate_count,
        "nonmate_correct": nonmate_correct_count,
        "mate_acc": mate_acc,  # fraction
        "nonmate_acc": nonmate_acc,  # fraction
        "overall_acc": overall_acc,  # fraction
        "halluc_rate": halluc_rate,  # fraction
        "halluc_count": halluc_count,
        "f1_score": f1_score,
        "total": total,
    }


def plot_leaderboard(results, output_file="leaderboard.png"):
    """
    Create a horizontal bar chart with two bars per run:
      1) Mate Accuracy (%)
      2) Hallucination Rate (%)

    Reusing the old layout approach (F1 vs Exact) but replaced with
    MateAcc vs Halluc%.
    """

    # Sort results by F1 score descending
    sorted_results = sorted(results, key=lambda r: r["f1_score"], reverse=True)

    labels = [
        os.path.basename(r["file"]).replace("-mates.results", "") for r in sorted_results
    ]
    f1_scores = [r["f1_score"] * 100 for r in sorted_results]

    y = np.arange(len(labels))
    height = 0.8

    fig, ax = plt.subplots(figsize=(10, 6))

    # Single bar for combined F1 score
    rects = ax.barh(
        y=y,
        width=f1_scores,
        height=height,
        label="F1 Score: Mate Accuracy vs Anti-Hallucination",
        color=COLOR_MATE,
    )

    ax.set_xlabel("Percentage")
    ax.set_title("Random-Chess-Mates LLM Leaderboard")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)  # Fixed x-axis from 0% to 100%
    ax.legend()

    # Invert y-axis so best is at the top
    ax.invert_yaxis()

    # Reuse old auto-label logic
    def autolabel_horizontal(rects, x_offset, y_offset):
        """Attach a text label to the right of each bar."""
        for rect in rects:
            width = rect.get_width()
            y_center = rect.get_y() + rect.get_height() / 2
            ax.annotate(
                f"{width:.1f}%",
                xy=(width, y_center),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
            )

    # Label each bar with F1 score
    autolabel_horizontal(rects, x_offset=5, y_offset=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nSaved leaderboard plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Score mate vs. non-mate logs, produce summary + horizontal chart."
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        required=True,
        help="Paths to .results files with lines: FEN|IS_MATE_IN_2|BEST_MOVE|LLM_RESPONSE|CORRECT|ERROR",
    )
    parser.add_argument(
        "--plot",
        "-p",
        default="leaderboard.png",
        help="Output filename for the horizontal bar chart",
    )
    args = parser.parse_args()

    all_results = []
    for fpath in args.files:
        stats = process_log_file(fpath)
        if stats:
            all_results.append(stats)

    if not all_results:
        print("No valid data found.")
        return

    # Sort by mate accuracy descending
    all_results.sort(key=lambda r: r["mate_acc"], reverse=True)

    print()
    print("Scoring Summary (Mate vs. Non-mate):")
    print()
    print("| Rank | File                     | #Total | MateAcc% | Halluc% |  F1(%) |")
    print("|------|--------------------------|--------|----------|---------|--------|")

    for idx, r in enumerate(all_results, 1):
        name = os.path.basename(r["file"])
        print(
            f"| {idx:^4} "
            f"| {name:<24} "
            f"| {r['mate_count'] + r['nonmate_count']:^6d} "
            f"| {r['mate_acc'] * 100:7.1f}% "
            f"| {r['halluc_rate'] * 100:6.1f}% "
            f"| {r['f1_score'] * 100:5.1f}% |"
        )

    # Produce a horizontal bar chart for each run
    plot_leaderboard(all_results, args.plot)


if __name__ == "__main__":
    main()
