#!/usr/bin/env python3
"""
Ablation analysis for Bank of England MPC Minutes:
Compare stance scores (Roberta-based) for full vs summary texts
against reference monthly sentiment scores.

Inputs:
 - minutes_boe_full.json  (from meetings_full_boe.py)
 - boe_reference_scores.csv  (columns: 'Hawkishness Date','Hawkishness')

Model:
 - gtfintechlab/model_bank_of_england_stance_label

Outputs:
 - /data/plots/ablation_minutes_plot.png
 - /data/raw/ablation_minutes_plot_results.csv
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === Config defaults ===
RAW_BASE = "/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/raw"
PLOTS_DIR = "/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MINUTES_JSON = os.path.join(RAW_BASE, "minutes_boe_full.json")
REFERENCE_CSV = os.path.join(RAW_BASE, "boe_reference_scores.csv")
MODEL_NAME = "gtfintechlab/model_bank_of_england_stance_label"


# --- Helpers ---

def load_minutes(path):
    """Load full minutes JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for date_str, entry in data.items():
        if not entry.get("full_text"):
            continue
        rows.append({
            "date": pd.to_datetime(date_str),
            "full_text": entry.get("full_text", ""),
            "summary_text": entry.get("summary_text", ""),
        })
    df = pd.DataFrame(rows).sort_values("date")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def load_reference(path):
    """Load reference scores using actual BoE column names."""
    ref = pd.read_csv(path)
    if "Hawkishness Date" not in ref.columns or "Hawkishness" not in ref.columns:
        raise ValueError("Expected columns: 'Hawkishness Date' and 'Hawkishness'")
    ref["month"] = pd.to_datetime(ref["Hawkishness Date"]).dt.to_period("M").astype(str)
    ref = ref.rename(columns={"Hawkishness": "reference_score"})
    return ref[["month", "reference_score"]]


def get_roberta_score(model, tokenizer, text, device="cpu"):
    """Return stance (hawkishness) score from the Roberta stance model."""
    if not text or len(text.strip()) < 20:
        return np.nan
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    # assuming labels: [dovish, neutral, hawkish]
    hawkishness = probs[2] - probs[0]
    return float(hawkishness)


def compute_monthly_scores(df, tokenizer, model, device="cpu"):
    """Compute monthly averages for full vs summary texts."""
    recs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring minutes"):
        fscore = get_roberta_score(model, tokenizer, row["full_text"], device)
        sscore = get_roberta_score(model, tokenizer, row["summary_text"], device)
        recs.append({
            "date": row["date"],
            "month": row["month"],
            "full_score": fscore,
            "summary_score": sscore,
        })
    df_s = pd.DataFrame(recs)
    monthly = df_s.groupby("month")[["full_score", "summary_score"]].mean().reset_index()
    return monthly


def plot_comparison(df_merge, out_path):
    """Plot reference vs full vs summary scores."""
    plt.figure(figsize=(12,6))
    plt.plot(df_merge["month"], df_merge["reference_score"], label="Reference", color="black", linewidth=2.5)
    plt.plot(df_merge["month"], df_merge["full_score"], label="Full minutes", linestyle="--", color="tab:blue")
    plt.plot(df_merge["month"], df_merge["summary_score"], label="Summary", linestyle=":", color="tab:orange")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.title("BoE MPC Minutes: Roberta Stance Score Comparison")
    plt.xlabel("Month")
    plt.ylabel("Hawkishness Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"📊 Saved plot -> {out_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Ablation comparison for BoE minutes.")
    parser.add_argument("--minutes-json", type=str, default=MINUTES_JSON)
    parser.add_argument("--reference-csv", type=str, default=REFERENCE_CSV)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-plot", type=str, default=os.path.join(PLOTS_DIR, "ablation_minutes_plot.png"))
    args = parser.parse_args()

    print(f"📂 Loading minutes from {args.minutes_json}")
    df_minutes = load_minutes(args.minutes_json)
    print(f"✅ Loaded {len(df_minutes)} entries ({df_minutes['month'].nunique()} months)")

    print(f"📂 Loading reference from {args.reference_csv}")
    ref = load_reference(args.reference_csv)
    print(f"✅ Loaded {len(ref)} reference months")

    print(f"🤖 Loading Roberta model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()

    print("⚙️  Scoring full & summary texts...")
    monthly_scores = compute_monthly_scores(df_minutes, tokenizer, model, args.device)

    merged = pd.merge(ref, monthly_scores, on="month", how="inner").sort_values("month")
    out_csv = os.path.join(RAW_BASE, "ablation_minutes_plot_results.csv")
    merged.to_csv(out_csv, index=False)
    print(f"💾 Saved merged scores -> {out_csv}")

    # Optional correlation check
    corr_full = merged["reference_score"].corr(merged["full_score"])
    corr_summary = merged["reference_score"].corr(merged["summary_score"])
    print(f"📈 Correlation: full={corr_full:.3f}, summary={corr_summary:.3f}")

    plot_comparison(merged, args.out_plot)


if __name__ == "__main__":
    main()
