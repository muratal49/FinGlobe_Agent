#!/usr/bin/env python3
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
import re

# =====================================================
#        MODEL DEFINITIONS — OPTION 2 (NOW)
# =====================================================
MODELS = {
    "BOC_BANK": "gtfintechlab/model_bank_of_england_stance_label"
}

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/model_bank_of_england_stance_label")
# model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/model_bank_of_england_stance_label")


BASE = "/Users/murat/Desktop/Capstone/FinGlobe_Agent"

# =====================================================
#      PATHS FOR ALL MONTHLY CORPORA (JSON)
# =====================================================
BOC_MIN = f"{BASE}/data/raw/minutes_boc_monthly.json"
BOC_SPE = f"{BASE}/data/raw/speeches_boc_monthly.json"
BOC_MER = f"{BASE}/data/raw/merged_boc_monthly.json"
BOC_REF = f"{BASE}/data/raw/reference_boc_monthly.json"

OUTPUT_DIR = Path(f"{BASE}/output_boc_only")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =====================================================
#            LOAD PUBLIC FALLBACK CLASSIFIER
# =====================================================
def load_classifier(model_name):
    print(f"\n🧠 Loading classifier: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True
    )

# =====================================================
#            SENTENCE SPLITTING
# =====================================================
def split_sentences(text):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]

# =====================================================
#            DOCUMENT → LABELS
# =====================================================
def classify_document(text, classifier):
    sentences = split_sentences(text)
    labels = []

    for s in sentences:
        # SAFE truncation for long sentences
        s = s[:500]

        try:
            out = classifier(s)[0]
            best = max(out, key=lambda x: x["score"])
            labels.append(best["label"])
        except Exception:
            continue

    return labels

# =====================================================
#            LABELS → SCORE
# =====================================================
def stance_score(labels):
    hawk = labels.count("LABEL_1")
    dove = labels.count("LABEL_2")
    neutral = labels.count("LABEL_0")
    irr = labels.count("LABEL_3")

    total = hawk + dove + neutral + irr
    if total == 0:
        return 0.0

    return (hawk - dove) / total

# =====================================================
#            LOAD JSON CORPUS
# =====================================================
def load_reference_scores(path):
    d = json.loads(Path(path).read_text())
    return d  # dict(month → score)

# =====================================================
#      SCORE CORPUS → DICT(month → score)
# =====================================================
def score_corpus(path, classifier):
    corpus = json.loads(Path(path).read_text())
    out = {}

    for month, text in corpus.items():
        labels = classify_document(text, classifier)
        out[month] = stance_score(labels)

    return out

# =====================================================
#            SAVE CSV + PLOT
# =====================================================
def save_results_and_plot(prefix, model_scores, ref_scores):
    rows = []
    for m, sc in model_scores.items():
        if m in ref_scores:
            rows.append({
                "month": m,
                "model_score": sc,
                "reference_score": ref_scores[m]
            })

    df = pd.DataFrame(rows).sort_values("month")

    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✔ Saved CSV: {csv_path}")

    # Plot
    plt.figure(figsize=(13,5))
    plt.plot(df["month"], df["reference_score"], color="black", linewidth=3, label="Reference")
    plt.plot(df["month"], df["model_score"], "b--", linewidth=2, label="Predicted")
    plt.xticks(rotation=60)
    plt.title(prefix)
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"{prefix}.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"✔ Saved plot: {png_path}")

    mse = ((df["model_score"] - df["reference_score"])**2).mean()
    print(f"🏆 MSE for {prefix}: {mse:.6f}")

    return mse

# =====================================================
#                 MAIN BOC-ONLY LOOP
# =====================================================
def run_boc_only():
    ref_scores = load_reference_scores(BOC_REF)

    classifier = load_classifier(MODELS["BOC_BANK"])

    results = {}

    datasets = {
        "BOC_MIN": BOC_MIN,
        "BOC_SPE": BOC_SPE,
        "BOC_MER": BOC_MER
    }

    print("\n==============================")
    print("     SCORING BOC ONLY")
    print("==============================\n")

    for ds_key, ds_path in datasets.items():
        print(f"→ Scoring {ds_key}")
        model_scores = score_corpus(ds_path, classifier)

        mse = save_results_and_plot(
            prefix=f"BOC_BANK_{ds_key}",
            model_scores=model_scores,
            ref_scores=ref_scores
        )

        results[ds_key] = mse

    print("\n🎉 BOC scoring completed.\n")
    print(results)
    return results

# =====================================================
#                     RUN
# =====================================================
if __name__ == "__main__":
    run_boc_only()
