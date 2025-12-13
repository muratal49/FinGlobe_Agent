#!/usr/bin/env python3
import json, re, torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

# =====================================================
#  MODEL DEFINITIONS (BOE / BOC / WCB)
# =====================================================

MODELS = {
    "BOE_WCB":  "gtfintechlab/model_WCB_stance_label",
    "BOE_BANK": "gtfintechlab/model_bank_of_england_stance_label",

    "BOC_WCB":  "gtfintechlab/model_WCB_stance_label",
    "BOC_BANK": "gtfintechlab/bank_of_canada_stance_label"
}

# =====================================================
#  PATHS
# =====================================================

BASE = "/Users/murat/Desktop/Capstone/FinGlobe_Agent"

BOE_MIN  = f"{BASE}/data/raw/minutes_boe_monthly.json"
BOE_SPE  = f"{BASE}/data/raw/speeches_boe_monthly.json"
BOE_MER  = f"{BASE}/data/raw/merged_boe_monthly.json"
BOE_REF  = f"{BASE}/data/raw/reference_boe_monthly.json"

BOC_MIN  = f"{BASE}/data/raw/minutes_boc_monthly.json"
BOC_SPE  = f"{BASE}/data/raw/speeches_boc_monthly.json"
BOC_MER  = f"{BASE}/data/raw/merged_boc_monthly.json"
BOC_REF  = f"{BASE}/data/raw/reference_boc_monthly.json"

# =====================================================
#  LOADING JSON CORPORA
# =====================================================

def load_json_dict(path):
    d = json.loads(Path(path).read_text())
    df = pd.DataFrame({"month": list(d.keys()), "text": list(d.values())})
    df = df.sort_values("month").reset_index(drop=True)
    return df

def load_reference_scores(path):
    d = json.loads(Path(path).read_text())
    df = pd.DataFrame({"month": list(d.keys()), "score": list(d.values())})
    df = df.sort_values("month").reset_index(drop=True)
    return df

def align(df_text, df_score):
    return pd.merge(df_text, df_score, on="month", how="inner")

# =====================================================
#  STANCE CLASSIFIER
# =====================================================

def load_classifier(model_name):
    print(f"🧠 Loading stance model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, do_basic_tokenize=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    return pipe

def split_sentences(text):
    text = text.replace("\n", " ")
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 3]

def classify_document(text, classifier):
    sentences = split_sentences(text)
    labels = []
    for s in sentences:
        out = classifier(s)[0]
        best = max(out, key=lambda x: x['score'])
        labels.append(best['label'])
    return labels

# =====================================================
#  LABEL FREQUENCY → SCORE
# =====================================================

def stance_score(labels):
    hawk = labels.count("LABEL_1")
    dove = labels.count("LABEL_2")
    neutral = labels.count("LABEL_0")

    denom = hawk + dove + neutral
    if denom == 0:
        return 0.0

    return (hawk - dove) / denom

def score_corpus(json_path, classifier):
    data = json.loads(Path(json_path).read_text())
    results = {}
    for month, text in data.items():
        labels = classify_document(text, classifier)
        results[month] = stance_score(labels)
    return results

# =====================================================
#  MASTER RUNNER — 12 SCORING RUNS
# =====================================================

def run_all_models():

    datasets = {
        "BOE_MIN": BOE_MIN,
        "BOE_SPE": BOE_SPE,
        "BOE_MER": BOE_MER,

        "BOC_MIN": BOC_MIN,
        "BOC_SPE": BOC_SPE,
        "BOC_MER": BOC_MER,
    }

    reference_scores = {
        "BOE": json.loads(Path(BOE_REF).read_text()),
        "BOC": json.loads(Path(BOC_REF).read_text()),
    }

    outdir = Path(f"{BASE}/output")
    outdir.mkdir(exist_ok=True)

    results_all = {}

    for model_key, model_name in MODELS.items():

        print(f"\n🧠 Running model {model_key}")
        clf = load_classifier(model_name)

        bank = "BOE" if model_key.startswith("BOE") else "BOC"

        for ds_key, json_path in datasets.items():

            if not ds_key.startswith(bank):
                continue

            print(f"  → Scoring {ds_key}")

            scores = score_corpus(json_path, clf)
            results_all[f"{model_key}_{ds_key}"] = scores

            # save CSV
            df = pd.DataFrame.from_dict(scores, orient="index", columns=["model_score"]).sort_index()
            df_path = outdir / f"{model_key}_{ds_key}.csv"
            df.to_csv(df_path)

            # MSE
            ref = reference_scores[bank]
            aligned = [(scores[m], ref[m]) for m in scores if m in ref]

            mse = sum((m - r)**2 for m, r in aligned) / len(aligned) if len(aligned)>3 else None
            print(f"     MSE = {mse}")

            (outdir / f"{model_key}_{ds_key}_mse.txt").write_text(str(mse))

    print("\n✔ ALL MODEL RUNS COMPLETED\n")
    return results_all

# =====================================================
#  ANALYTICS
# =====================================================

def combined_plot_bank(bank):
    outdir = Path(f"{BASE}/output")

    files = [
        f"{bank}_minutes_wcb.csv", f"{bank}_speeches_wcb.csv", f"{bank}_merged_wcb.csv",
        f"{bank}_minutes_bank.csv", f"{bank}_speeches_bank.csv", f"{bank}_merged_bank.csv"
    ]

    dfs = {f: pd.read_csv(outdir / f) for f in files}

    months = dfs[f"{bank}_minutes_wcb.csv"]["month"]
    ref = dfs[f"{bank}_minutes_wcb.csv"]["score"]

    plt.figure(figsize=(14,6))
    plt.plot(months, ref, color="black", linewidth=3, label="Reference")

    plt.plot(months, dfs[f"{bank}_minutes_wcb.csv"]["model_score"], "b--", label="Minutes WCB")
    plt.plot(months, dfs[f"{bank}_minutes_bank.csv"]["model_score"], "b-",  label="Minutes Bank")

    plt.plot(months, dfs[f"{bank}_speeches_wcb.csv"]["model_score"], "g--", label="Speeches WCB")
    plt.plot(months, dfs[f"{bank}_speeches_bank.csv"]["model_score"], "g-",  label="Speeches Bank")

    plt.plot(months, dfs[f"{bank}_merged_wcb.csv"]["model_score"], "r--", label="Merged WCB")
    plt.plot(months, dfs[f"{bank}_merged_bank.csv"]["model_score"], "r-",  label="Merged Bank")

    plt.xticks(rotation=60)
    plt.title(f"{bank.upper()} – Minutes vs Speeches vs Merged")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{bank}_combined.png", dpi=200)
    plt.close()

def wcb_vs_bank(bank, dataset):
    outdir = Path(f"{BASE}/output")

    df_w = pd.read_csv(outdir / f"{bank}_{dataset}_wcb.csv")
    df_b = pd.read_csv(outdir / f"{bank}_{dataset}_bank.csv")

    months = df_w["month"]
    ref = df_w["score"]

    plt.figure(figsize=(12,5))
    plt.plot(months, ref, color="black", linewidth=3, label="Reference")
    plt.plot(months, df_w["model_score"], "b--", linewidth=2, label="WCB")
    plt.plot(months, df_b["model_score"], "r-", linewidth=3, label="Bank model")

    plt.xticks(rotation=60)
    plt.title(f"{bank.upper()} – {dataset.capitalize()} – WCB vs Bank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{bank}_{dataset}_compare.png", dpi=200)
    plt.close()

def run_analytics():
    combined_plot_bank("BOE")
    combined_plot_bank("BOC")

    for bank in ["BOE", "BOC"]:
        for dataset in ["minutes", "speeches", "merged"]:
            wcb_vs_bank(bank, dataset)

    print("🎉 All analytics generated.")

# =====================================================
#  MAIN
# =====================================================

if __name__ == "__main__":
    run_all_models()
    run_analytics()
