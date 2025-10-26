#!/usr/bin/env python3
"""
Scores a combined text source derived from merging monthly Minutes and Speeches text.
"""
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt", quiet=True)

# --- Configuration ---
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
MINUTES_PATH = BASE_PATH / "data/raw/minutes_boe_clean.json"
SPEECHES_PATH = BASE_PATH / "data/raw/speeches_boe_clean.json"
OUTPUT_CSV = BASE_PATH / "data/analysis_results/scored_merged_text.csv"

# --- Model Loading (Needed for scoring) ---
def load_model():
    """Load stance detection model."""
    model_name = "gtfintechlab/model_bank_of_england_stance_label"
    print(f"🧩 Loading Hugging Face model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        config=config,
        framework="pt",
        truncation="only_first",
        batch_size=32,
    )
    return clf

# --- Scoring Core Logic (Copied from main scoring tool) ---
def compute_hawkishness(sentences, clf):
    """Compute hawkishness ratio for a list of sentences."""
    preds = clf(sentences, truncation=True)
    counts = {"hawkish": 0, "dovish": 0, "neutral": 0, "irrelevant": 0}
    for p in preds:
        label = p["label"].upper()
        if label == "LABEL_1": counts["hawkish"] += 1
        elif label == "LABEL_2": counts["dovish"] += 1
        elif label == "LABEL_0": counts["neutral"] += 1
        elif label == "LABEL_3": counts["irrelevant"] += 1

    total = len(preds)
    denom = max(total - counts["irrelevant"], 1)
    hawkishness = (counts["hawkish"] - counts["dovish"]) / denom
    return round(hawkishness, 4), counts

def score_text_source(data, source_name, clf):
    """Compute hawkishness scores for a dictionary of {YYYY-MM: aggregated_text}."""
    results = []
    print(f"Scoring merged text for {len(data)} months...")
    for month_str, text in tqdm(data.items()):
        text = str(text)
        sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]
        score = None
        if sentences:
            try:
                score, _ = compute_hawkishness(sentences, clf)
            except Exception as e:
                print(f"⚠️ Error processing {month_str}: {e}")
        
        results.append({"month": month_str, "scores_merged": score})
    return pd.DataFrame(results)

def main():
    try:
        clf = load_model()
    except Exception as e:
        print(f"❌ Error loading model. Cannot proceed: {e}")
        return

    # --- 1. Load Cleaned JSON Files ---
    print("1. Merging text from Minutes and Speeches...")
    try:
        with MINUTES_PATH.open("r", encoding="utf-8") as f:
            minutes_data = json.load(f)
        with SPEECHES_PATH.open("r", encoding="utf-8") as f:
            speeches_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON files: {e}")
        return
    
    # --- 2. Combine Text by Month ---
    merged_monthly_texts = {}
    all_months = set(minutes_data.keys()) | set(speeches_data.keys())
    
    separator = "\n\n\n[NEW SOURCE TEXT]\n\n\n" # Clear demarcation between the two source texts
    
    for month in sorted(list(all_months)):
        minutes_text = minutes_data.get(month, "").strip()
        speeches_text = speeches_data.get(month, "").strip()
        
        combined_text = ""
        
        if minutes_text:
            combined_text += minutes_text
        
        if speeches_text:
            if combined_text:
                combined_text += separator
            combined_text += speeches_text
            
        if combined_text:
            merged_monthly_texts[month] = combined_text
        
    # --- 3. Score the Merged Text ---
    df_merged_scores = score_text_source(merged_monthly_texts, 'merged', clf)
    
    # --- 4. Save Final CSV ---
    df_merged_scores = df_merged_scores.rename(columns={'scores_merged': 'scores_merged_text'})
    df_merged_scores['month_period'] = df_merged_scores['month'].str.replace('-', ',')
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged_scores[['month_period', 'scores_merged_text']].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved merged scores → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()