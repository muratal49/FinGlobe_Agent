#!/usr/bin/env python3
import json
import re
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

# =====================================================
#        MODEL (ONLY BOE MODEL, USED FOR BOTH BANKS)
# =====================================================
MODELS = {
    "BOE": "gtfintechlab/model_bank_of_england_stance_label",
    "BOC": "gtfintechlab/model_bank_of_england_stance_label"
}

BASE = "/Users/murat/Desktop/Capstone/FinGlobe_Agent"

# -----------------------------------------------------
# PATHS FOR BOTH BANKS (JSON)
# -----------------------------------------------------
BOE_MIN = f"{BASE}/data/raw/minutes_boe_monthly.json"
BOE_SPE = f"{BASE}/data/raw/speeches_boe_monthly.json"
BOE_MER = f"{BASE}/data/raw/merged_boe_monthly.json"
BOE_REF = f"{BASE}/data/raw/reference_boe_monthly.json"

BOC_MIN = f"{BASE}/data/raw/minutes_boc_monthly.json"
BOC_SPE = f"{BASE}/data/raw/speeches_boc_monthly.json"
BOC_MER = f"{BASE}/data/raw/merged_boc_monthly.json"
BOC_REF = f"{BASE}/data/raw/reference_boc_monthly.json"

OUTPUT_DIR = Path(f"{BASE}/output_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
#            NATURAL NAMES FOR TITLES
# =====================================================
BANK_NAMES = {
    "BOE": "Bank of England",
    "BOC": "Bank of Canada"
}

DOC_NAMES = {
    "MIN": "Minutes",
    "SPE": "Speeches",
    "MER": "Merged Documents"
}

# =====================================================
#            LOAD STANCE CLASSIFIER
# =====================================================
def load_classifier(model_name):
    print(f"\n🧠 Loading classifier: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tok,
        return_all_scores=True,
        truncation=True,
        max_length=512
    )
    return pipe

# =====================================================
#            SENTENCE SPLITTING
# =====================================================
def split_sentences(text):
    text = text.replace("\n", " ")
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 3]

# =====================================================
#            TRUNCATE EXCESSIVE SPEECH TEXT
# =====================================================
def truncate_text(text, max_words=700):
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

# =====================================================
#            DOCUMENT → LABELS
# =====================================================
def classify_document(text, classifier):
    # -------------------------------
    # FIX: handle dict inputs safely
    # -------------------------------
    if isinstance(text, dict):
        text = (
            text.get("text")
            or text.get("full_text")
            or text.get("summary")
            or ""
        )

    if not isinstance(text, str) or not text.strip():
        return []

    text = truncate_text(text, 700)
    sentences = split_sentences(text)
    labels = []

    for s in sentences:
        s = s[:500]
        try:
            out = classifier(s)
            best = max(out[0], key=lambda x: x["score"])
            labels.append(best["label"])
        except Exception:
            continue

    return labels

# =====================================================
#            LABELS → STANCE SCORE
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
#            LOADERS
# =====================================================
def load_corpus(path):
    return json.loads(Path(path).read_text())

def load_reference(path):
    return json.loads(Path(path).read_text())

# =====================================================
#           SCORE A SINGLE CORPUS
# =====================================================
def score_corpus(path, classifier):
    data = load_corpus(path)
    out = {}
    for month, text in data.items():
        labels = classify_document(text, classifier)
        out[month] = stance_score(labels)
    return out

# =====================================================
#   PLOTTING (UNCHANGED)
# =====================================================
def plot_csv(csv_path, out_path):
    df = pd.read_csv(csv_path)

    if not {"month", "model_score", "reference_score"} <= set(df.columns):
        print(f"⚠ Skipping {csv_path.name}")
        return

    parts = csv_path.stem.split("_")
    bank = parts[2]
    doc  = parts[3]

    bank_name = BANK_NAMES[bank]
    doc_name = DOC_NAMES[doc]

    mse = float(((df["model_score"] - df["reference_score"])**2).mean())

    plt.figure(figsize=(14,6))
    plt.plot(df["month"], df["reference_score"], "k-", linewidth=3, label="Reference")
    plt.plot(df["month"], df["model_score"], "b-", linewidth=2, label="Model Score")

    plt.xticks(rotation=60)
    plt.title(f"{bank_name} – {doc_name}", fontsize=16)
    plt.grid(alpha=0.3)

    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8)

    plt.annotate(
        f"MSE = {mse:.4f}",
        xy=(0.02,0.94),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✔ Saved plot: {out_path.name}")

# =====================================================
#        MAIN SCORING FUNCTION
# =====================================================
def run_all():
    banks = {
        "BOE": {
            "MIN": BOE_MIN,
            "SPE": BOE_SPE,
            "MER": BOE_MER,
            "REF": BOE_REF
        },
        "BOC": {
            "MIN": BOC_MIN,
            "SPE": BOC_SPE,
            "MER": BOC_MER,
            "REF": BOC_REF
        }
    }

    for bank, info in banks.items():
        print(f"\n==========================")
        print(f"  PROCESSING {BANK_NAMES[bank]}")
        print(f"==========================")

        clf = load_classifier(MODELS[bank])
        refs = load_reference(info["REF"])

        for doc in ["MIN", "SPE", "MER"]:
            print(f"→ Scoring {BANK_NAMES[bank]} – {DOC_NAMES[doc]}")

            scores = score_corpus(info[doc], clf)

            rows = []
            for m, s in scores.items():
                if m in refs:
                    rows.append({
                        "month": m,
                        "model_score": s,
                        "reference_score": refs[m]
                    })

            df = pd.DataFrame(rows).sort_values("month")
            out_csv = OUTPUT_DIR / f"{bank}_BANK_{bank}_{doc}.csv"
            df.to_csv(out_csv, index=False)
            print(f"✔ Saved {out_csv.name}")

            out_png = OUTPUT_DIR / f"{bank}_BANK_{bank}_{doc}.png"
            plot_csv(out_csv, out_png)

    print("\n🎉 ALL DONE — FULL BOE + BOC RUN COMPLETE\n")

# =====================================================
#                RUN
# =====================================================
if __name__ == "__main__":
    run_all()
