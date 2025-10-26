#!/usr/bin/env python3
"""
Combined Hawkishness Scoring and Analysis Script
------------------------------------------------
1. Scores all text sources (aggregated monthly) using the provided JSON files.
2. Loads and aggregates reference scores monthly.
3. Merges all time series into a single monthly CSV, keeping all months.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt", quiet=True)


# --- Configuration (UPDATED FOR CLEANED JSON FILES) ---
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")

REFERENCE_PATH = BASE_PATH / "data/raw/boe_reference_scores.csv"
# Updated to use the cleaned, monthly-aggregated JSON files:
MINUTES_PATH = BASE_PATH / "data/raw/minutes_boe_clean.json"    
SPEECHES_PATH = BASE_PATH / "data/raw/speeches_boe_clean.json"
PUBLICATIONS_PATH = BASE_PATH / "data/raw/publications_boe.json"
OUTPUT_CSV = BASE_PATH / "data/analysis_results/combined_monthly_scores.csv"

# Column names for Reference data (based on previous samples)
REF_DATE_COL = 'Hawkishness Date'
REF_SCORE_COL = 'Hawkishness'


# --- Model Loading ---
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
    print("✅ Model ready.\n")
    return clf


# --- Scoring Core Logic ---
def compute_hawkishness(sentences, clf):
    """Compute hawkishness ratio for a list of sentences."""
    preds = clf(sentences, truncation=True)

    counts = {"hawkish": 0, "dovish": 0, "neutral": 0, "irrelevant": 0}
    for p in preds:
        label = p["label"].upper()
        if label == "LABEL_1":
            counts["hawkish"] += 1
        elif label == "LABEL_2":
            counts["dovish"] += 1
        elif label == "LABEL_0":
            counts["neutral"] += 1
        elif label == "LABEL_3":
            counts["irrelevant"] += 1

    total = len(preds)
    denom = max(total - counts["irrelevant"], 1)
    hawkishness = (counts["hawkish"] - counts["dovish"]) / denom
    return round(hawkishness, 4), counts


def score_text_source(data, source_name, clf):
    """Compute hawkishness scores for a dictionary of {YYYY-MM: aggregated_text}."""
    results = []

    print(f"Processing {source_name} ({len(data)} months/documents)...")
    for month_str, text in tqdm(data.items()):
        text = str(text)
        
        # Sentence segmentation (only sentences with > 3 words)
        sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]
        
        if not sentences:
            results.append({"month": month_str, f"scores_{source_name}": None})
            continue

        try:
            score, _ = compute_hawkishness(sentences, clf)
        except Exception as e:
            print(f"⚠️ Error processing {month_str} for {source_name}: {e}") 
            score = None

        results.append({
            "month": month_str,
            f"scores_{source_name}": score
        })

    return pd.DataFrame(results)



def load_and_aggregate_reference(ref_path):
    """Loads reference scores, aggregates to monthly mean, and formats."""
    print("Loading and aggregating reference scores...")
    try:
        df_ref = pd.read_csv(ref_path)
    except FileNotFoundError:
        print(f"❌ Error: Reference file not found at {ref_path}")
        return pd.DataFrame(columns=['month', 'scores_reference'])
        
    df_ref = df_ref.rename(columns={
        REF_DATE_COL: 'date', 
        REF_SCORE_COL: 'scores_reference'
    })

    df_ref['date'] = pd.to_datetime(df_ref['date'], errors='coerce')
    df_ref = df_ref.dropna(subset=['date'])

    # Aggregate by month
    df_ref['month'] = df_ref['date'].dt.to_period('M')
    df_ref_agg = (
        df_ref.groupby('month')['scores_reference']
        .mean()
        .reset_index()
    )
    
    # Format month column as string 'YYYY-MM'
    df_ref_agg['month'] = df_ref_agg['month'].astype(str)
    
    return df_ref_agg


def main():
    # --- 0. Load Model ---
    try:
        clf = load_model()
    except Exception as e:
        print(f"❌ Error loading model. Cannot proceed with scoring: {e}")
        return
        
    # --- Load all four JSON data sources ---
    def load_json_data(path):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON from {path}: {e}")
            return {}

    # 1. Load data
    minutes_data = load_json_data(MINUTES_PATH)
    speeches_data = load_json_data(SPEECHES_PATH)
    publications_data = load_json_data(PUBLICATIONS_PATH)
    
    # 2. Score all sources (Data is already MONTHLY aggregated here)
    df_minutes = score_text_source(minutes_data, 'minutes', clf)
    df_speeches = score_text_source(speeches_data, 'speeches', clf)
    df_publications = score_text_source(publications_data, 'publications', clf)
        
    # 3. Load and Aggregate Reference Scores
    df_reference = load_and_aggregate_reference(REFERENCE_PATH)
    
    
    # --- 4. Merge DataFrames (Using Outer Join to keep all months) ---
    print("Merging all dataframes using outer joins...")
    
    # Start with a base DataFrame containing all unique months from all three scored sources
    all_months = pd.concat([
        df_minutes['month'], 
        df_speeches['month'], 
        df_publications['month']
    ]).drop_duplicates().to_frame()
    all_months = all_months.rename(columns={all_months.columns[0]: 'month'})
    
    df_combined = all_months
    
    # Merge all scores onto the consolidated month list
    df_combined = df_combined.merge(df_minutes, on='month', how='outer')
    df_combined = df_combined.merge(df_speeches, on='month', how='outer')
    df_combined = df_combined.merge(df_publications, on='month', how='outer')
    df_combined = df_combined.merge(df_reference, on='month', how='outer')
    
    # --- 5. Final Formatting and Cleanup ---
    
    # Convert 'month' column to datetime for sorting (YYYY-MM format is clean)
    df_combined['month_dt'] = pd.to_datetime(df_combined['month'], format='%Y-%m', errors='coerce')
    
    # Drop rows where month parsing failed
    df_combined = df_combined.dropna(subset=['month_dt'])
    
    # Final formatting and sorting
    df_combined = df_combined.sort_values('month_dt')
    df_combined['month_period'] = df_combined['month_dt'].dt.strftime('%Y,%m')
    
    # Final CSV structure
    df_combined = df_combined[[
        'month_period', 
        'scores_minutes', 
        'scores_speeches', 
        'scores_publications', 
        'scores_reference'
    ]].reset_index(drop=True)
    
    # --- 6. Save Final CSV ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved final combined monthly scores → {OUTPUT_CSV}")
    print(df_combined.head().to_markdown(index=False))


if __name__ == "__main__":
    main()