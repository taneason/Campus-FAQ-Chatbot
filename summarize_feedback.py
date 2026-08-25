"""
Summarize collected user feedback from data/feedback.csv.

Usage:
    python summarize_feedback.py

Writes:
    data/feedback_summary.csv
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.csv"
SUMMARY_PATH = BASE_DIR / "data" / "feedback_summary.csv"


def main():
    if not FEEDBACK_PATH.exists():
        print(f"No feedback collected yet: {FEEDBACK_PATH} not found.")
        return

    df = pd.read_csv(FEEDBACK_PATH)
    if df.empty:
        print("Feedback file is empty.")
        return

    rows = []
    for method, group in df.groupby("method"):
        total = len(group)
        helpful_count = (group["helpful"] == "Yes").sum()
        helpful_pct = 100 * helpful_count / total
        rows.append(
            {
                "method": method,
                "total_responses": total,
                "helpful_count": helpful_count,
                "helpful_pct": round(helpful_pct, 1),
            }
        )
        print(f"{method}: {helpful_count}/{total} helpful ({helpful_pct:.1f}%)")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
