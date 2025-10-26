#!/usr/bin/env python3
"""
Merging Scores from Multiple BoE Sources
----------------------------------------
Combines hawkish/dovish monthly scores from:
  - speeches
  - minutes
  - publications

Aligns them by date, handles missing data, and computes an aggregated score
(mean of available sources for that month).

Output:
    data/analysis_results/boe_combined_monthly_scores.csv
"""

import pandas as pd
from pathlib import Path

def load_monthly_scores(file_path, source_name):
    """Load a monthly score CSV and rename the column to the source name."""
    if not Path(file_path).exists():
        print(f"⚠️ File not found for {source_name}: {file_path}")
        return pd.DataFrame(columns=["month", source_name])

    df = pd.read_csv(file_path)
    # Normalize month column
    if "month" not in df.columns:
        raise ValueError(f"❌ 'month' column missing in {file_path}")

    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    score_col = [c for c in df.columns if "hawk" in c.lower() or "score" in c.lower()]
    if not score_col:
        raise ValueError(f"❌ No score column found in {file_path}")
    df.rename(columns={score_col[0]: source_name}, inplace=True)
    return df[["month", source_name]]


def merge_scores():
    print("🔗 Loading monthly score files...")

    base = Path("data/analysis_results")
    speech_path = base / "speech_monthly_scores.csv"
    minutes_path = base / "minutes_monthly_scores.csv"
    publication_path = base / "publication_monthly_scores.csv"

    df_speech = load_monthly_scores(speech_path, "speeches")
    df_minutes = load_monthly_scores(minutes_path, "minutes")
    df_public = load_monthly_scores(publication_path, "publications")

    print("🧩 Merging datasets by month...")
    merged = pd.merge(df_speech, df_minutes, on="month", how="outer")
    merged = pd.merge(merged, df_public, on="month", how="outer")

    merged.sort_values("month", inplace=True)
    merged.set_index("month", inplace=True)

    # Compute aggregated mean (ignore NaNs)
    merged["aggregated_score"] = merged[["speeches", "minutes", "publications"]].mean(axis=1, skipna=True)

    # Convert NaN to None for better readability in CSV
    merged = merged.where(pd.notnull(merged), None)

    out_file = base / "boe_combined_monthly_scores.csv"
    merged.to_csv(out_file, index=True)
    print(f"💾 Saved merged score table → {out_file}")

    # Print preview
    print("\n📊 Combined Score Preview:")
    print(merged.tail(10))
    return merged


if __name__ == "__main__":
    merge_scores()
