#!/usr/bin/env python3
"""
Scores all text sources (aggregated monthly) using the provided JSON files, 
merges results, and saves the final score CSVs with a date prefix.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from nltk.tokenize import sent_tokenize
import nltk
import argparse
from datetime import datetime

nltk.download("punkt", quiet=True)


# --- Configuration ---
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")

# Input Paths (from Step 2 prep)
REFERENCE_JSON_PATH = BASE_PATH / "data/raw/reference_boe_monthly.json"
MINUTES_JSON_PATH = BASE_PATH / "data/raw/minutes_boe_monthly.json"    
SPEECHES_JSON_PATH = BASE_PATH / "data/raw/speeches_boe_monthly.json"
PUBLICATIONS_JSON_PATH = BASE_PATH / "data/raw/publications_boe_monthly.json"

# Output Path Templates
OUTPUT_COMBINED_CSV_TEMPLATE = BASE_PATH / "data/analysis_results/combined_monthly_scores.csv"
OUTPUT_MERGED_CSV_TEMPLATE = BASE_PATH / "data/analysis_results/scored_merged_text.csv" 

# Column definitions
REF_SCORE_COL = 'scores_reference'
SCORED_COLS = ['scores_minutes', 'scores_speeches', 'scores_publications']
NEW_MERGED_COL = 'scores_merged_text'


# --- Model Loading (for local execution) ---
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


# --- Scoring Core Logic (for Roberta) ---
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
    print(f"Processing {source_name} ({len(data)} months/documents)...")
    for month_str, text in tqdm(data.items()):
        text = str(text)
        sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]
        score = None
        if sentences:
            try:
                score, _ = compute_hawkishness(sentences, clf)
            except Exception as e:
                print(f"⚠️ Error processing {month_str} for {source_name}: {e}") 
            score = score

        results.append({
            "month": month_str,
            f"scores_{source_name}": score
        })

    return pd.DataFrame(results)


def merge_and_score_policy_text(minutes_data, speeches_data, clf):
    """
    Applies the Minutes-available filter, merges Minutes + Speeches text, scores the block, 
    and returns the DataFrame for the NEW_MERGED_COL.
    """
    print("--- Generating and Scoring Merged Text (Minutes Filter Applied) ---")
    
    # 1. Identify Anchor Months (Months with Minutes Data)
    minutes_months = set(minutes_data.keys())
    
    final_merged_text_blocks = {}
    separator = "\n\n\n[NEW SOURCE TEXT]\n\n\n" 

    print(f"Filtering: ONLY using {len(minutes_months)} months where Minutes text is available.")
    
    for month in minutes_months:
        minutes_text = minutes_data.get(month, "").strip()
        speeches_text = speeches_data.get(month, "").strip()
        
        # ⚠️ CRITICAL FILTER: Only proceed if Minutes text exists
        if minutes_text:
            combined_text = minutes_text
            
            # Add speeches text if available for this specific month
            if speeches_text:
                combined_text += separator + speeches_text
            
            final_merged_text_blocks[month] = combined_text
    
    # 2. Score the fully merged text blocks
    df_merged_scores = score_text_source(final_merged_text_blocks, 'merged_text', clf)
    df_merged_scores = df_merged_scores.rename(columns={'scores_merged_text': NEW_MERGED_COL})
    
    return df_merged_scores


def main():
    parser = argparse.ArgumentParser(description="Roberta scoring script for all sources.")
    parser.add_argument("--start-date", type=str, required=True, help="The date (YYYY-MM-DD) used for file naming prefix.")
    args = parser.parse_args()
    
    # Generate prefix
    date_prefix = args.start_date.replace('-', '')
    
    # Finalize output paths with prefix
    final_combined_output = OUTPUT_COMBINED_CSV_TEMPLATE.parent / (date_prefix + "_" + OUTPUT_COMBINED_CSV_TEMPLATE.name)
    final_merged_output = OUTPUT_MERGED_CSV_TEMPLATE.parent / (date_prefix + "_" + OUTPUT_MERGED_CSV_TEMPLATE.name)
    
    try:
        clf = load_model()
    except Exception as e:
        print(f"❌ Error loading model. Cannot proceed with scoring: {e}")
        return
    
    # --- 1. Load Data ---
    minutes_data = json.load(MINUTES_JSON_PATH.open("r", encoding="utf-8"))
    speeches_data = json.load(SPEECHES_JSON_PATH.open("r", encoding="utf-8"))
    publications_data = json.load(PUBLICATIONS_JSON_PATH.open("r", encoding="utf-8"))
    df_reference = pd.read_json(REFERENCE_JSON_PATH, orient='index', dtype={0: float}).reset_index().rename(columns={'index': 'month', 0: REF_SCORE_COL})

    # --- 2. Score Sources ---
    df_minutes = score_text_source(minutes_data, 'minutes', clf)
    df_speeches = score_text_source(speeches_data, 'speeches', clf)
    df_publications = score_text_source(publications_data, 'publications', clf)
    df_merged = merge_and_score_policy_text(minutes_data, speeches_data, clf) # NEW MERGED SCORE

    # --- 3. Merge DataFrames ---
    print("Merging all dataframes using outer joins...")
    
    # Start with a base DataFrame containing all unique months
    all_months = pd.concat([
        df_minutes['month'], df_speeches['month'], df_publications['month'], df_reference['month']
    ]).drop_duplicates().to_frame().rename(columns={0: 'month'})
    
    # Merge all scores onto the consolidated month list
    df_combined = all_months
    df_combined = df_combined.merge(df_minutes, on='month', how='outer')
    df_combined = df_combined.merge(df_speeches, on='month', how='outer')
    df_combined = df_combined.merge(df_publications, on='month', how='outer')
    df_combined = df_combined.merge(df_reference, on='month', how='outer')
    df_combined = df_combined.merge(df_merged, on='month', how='outer') # Add NEW merged score

    # --- 4. Final Formatting and Cleanup ---
    df_combined['month_dt'] = pd.to_datetime(df_combined['month'], format='%Y-%m', errors='coerce')
    df_combined = df_combined.dropna(subset=['month_dt'])
    df_combined = df_combined.sort_values('month_dt')
    df_combined['month_period'] = df_combined['month_dt'].dt.strftime('%Y,%m')
    
    # Final CSV structure
    df_combined = df_combined[['month_period', 'scores_minutes', 'scores_speeches', 
                               'scores_publications', NEW_MERGED_COL, REF_SCORE_COL]].reset_index(drop=True)
    
    # --- 5. Save Final CSVs ---
    
    # Save 5a. The complete combined scores CSV (for external analysis)
    final_combined_output.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(final_combined_output, index=False)
    print(f"\n💾 Saved complete combined monthly scores → {final_combined_output.resolve()}")
    
    # Save 5b. The critical merged score CSV (used by the evaluation/plotting tool)
    df_merged_output = df_combined[['month_period', NEW_MERGED_COL]].copy().dropna()
    final_merged_output.parent.mkdir(parents=True, exist_ok=True)
    df_merged_output.to_csv(final_merged_output, index=False)
    print(f"💾 Saved critical merged score CSV → {final_merged_output.resolve()}")

if __name__ == "__main__":
    main()