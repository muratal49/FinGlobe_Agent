#!/usr/bin/env python3
"""
Scores the merged Minutes+Speeches text using the OpenAI API (GPT-4) 
and generates a structured justification for each monthly score.
"""
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import re
import argparse # Added argparse

# --- Configuration ---
load_dotenv()
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    exit()

# File paths
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
MINUTES_CLEAN_PATH = BASE_PATH / "data/raw/minutes_boe_clean.json"
SPEECHES_CLEAN_PATH = BASE_PATH / "data/raw/speeches_boe_clean.json"
MONTHS_REFERENCE_CSV = BASE_PATH / "data/analysis_results/scored_merged_text.csv"
# Output Path Template
OUTPUT_CSV_PATH_TEMPLATE = BASE_PATH / "data/analysis_results/{prefix}_openai_merged_scores_analysis.csv"


# Model and System Prompt Definition
LLM_MODEL = "gpt-4o-mini" 
MAX_WORDS_JUSTIFICATION = 200

# Scoring Criteria (as provided by the user)
SCORING_CRITERIA = {
    "panel_A1": ['inflation expectation', 'interest rate', 'bank rate', 'fund rate', 'price',
                 'economic activity', 'inflation', 'employment'],
    "panel_B1": ['unemployment', 'growth', 'exchange rate', 'productivity', 'deficit',
                 'demand', 'job market', 'monetary policy'],
    "panel_A2": ['anchor', 'cut', 'subdue', 'decline', 'decrease', 'reduce', 'low', 'drop',
                 'fall', 'fell', 'decelerate', 'slow', 'pause', 'pausing', 'stable', 'nonaccelerating',
                 'downward', 'tighten'],
    "panel_B2": ['ease', 'easing', 'rise', 'rising', 'increase', 'expand', 'improve', 'strong',
                 'upward', 'raise', 'high', 'rapid']
}

# --- System Prompt for LLM Scoring ---
SYSTEM_PROMPT = f"""
You are an expert economic and monetary policy analyst focused on the Bank of England (BoE).
Your task is to analyze concatenated text from official BoE Minutes and Speeches for a single month.

Analyze the text for hawkishness (tightening bias) or dovishness (easing bias) based on the following framework:
... (System Prompt remains the same) ...
"""

# --- TEXT LOADING AND FILTERING LOGIC (The New Requirement) ---

def load_merged_text_data(months_to_analyze):
    """
    Loads text sources, applies the Minutes-only month filter, and aggregates text.
    The rule is: ONLY combine Speeches text for months where Minutes are available.
    """
    print("1. Loading Minutes and Speeches text data...")
    try:
        with MINUTES_CLEAN_PATH.open("r", encoding="utf-8") as f:
            minutes_data = json.load(f)
        with SPEECHES_CLEAN_PATH.open("r", encoding="utf-8") as f:
            speeches_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading cleaned JSON files: {e}")
        return {} 

    # --- Filtering and Aggregation ---
    final_monthly_text = {}
    separator = "\n\n\n[NEW SOURCE TEXT]\n\n\n" 

    print("2. Filtering texts: ONLY merging for months where Minutes data exists...")
    
    for month_key in months_to_analyze:
        minutes_text = minutes_data.get(month_key, "").strip()
        speeches_text = speeches_data.get(month_key, "").strip()
        
        # ⚠️ CRITICAL FILTER: Only process the month if it has Minutes data
        if minutes_text:
            combined_text = minutes_text
            
            # Add speeches text if available for this specific month
            if speeches_text:
                combined_text += separator + speeches_text
            
            final_monthly_text[month_key] = combined_text
        
    print(f"✅ Prepared merged text blocks for {len(final_monthly_text)} months (Minutes filter enforced).")
    return final_monthly_text


# --- SCORING AND MAIN EXECUTION ---

def generate_llm_response(month, text, client):
    # ... (function remains the same) ...
    if not text or len(text) < 50:
        return None, "Text too short or empty for analysis."

    user_prompt = f"Analyze the following aggregated text block for the month of {month} and provide the Hawkishness Score and Justification:\n\n---\n{text}"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        score = float(result.get('score', 0.0))
        justification = result.get('justification', 'No justification provided.')
        
        return score, justification
        
    except Exception as e:
        print(f"Error during API call for {month}: {e}")
        return None, f"API Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="OpenAI scoring and justification script.")
    parser.add_argument("--start-date", type=str, required=True, help="The date (YYYY-MM-DD) used for file naming prefix.")
    args = parser.parse_args()
    
    print("--- Running OpenAI Scoring and Justification Pipeline ---")
    
    # 1. Generate prefix and finalize output path
    date_prefix = args.start_date.replace('-', '')
    final_output_csv_path = OUTPUT_CSV_PATH_TEMPLATE.parent / (date_prefix + "_" + OUTPUT_CSV_PATH_TEMPLATE.name)
    
    # 2. Get List of Months to Analyze from the Scored Merged CSV
    try:
        df_ref = pd.read_csv(MONTHS_REFERENCE_CSV)
        # Extract unique month strings (YYYY,MM) and convert to YYYY-MM format
        months_to_analyze_list = df_ref['month_period'].str.replace(',', '-').unique().tolist()
        print(f"Loaded {len(months_to_analyze_list)} months from scoring base CSV for analysis.")
    except Exception as e:
        print(f"❌ Error loading months from reference CSV: {e}")
        return
        
    # 3. Load and Prepare Merged Text Data (Applies new Minutes filter)
    merged_texts = load_merged_text_data(months_to_analyze_list)
    if not merged_texts:
        return

    # 4. Initialize OpenAI Client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    analysis_records = []
    
    # 5. Process and Score Each Month
    print(f"Processing {len(merged_texts)} months with LLM...")
    for month, text in tqdm(merged_texts.items(), desc="Scoring Months"):
        score, justification = generate_llm_response(month, text, client)
        
        analysis_records.append({
            'month_period': month.replace('-', ','),
            'scores_openai_merged': score,
            'openai_justification': justification
        })

    # 6. Save Results
    df_results = pd.DataFrame(analysis_records)
    df_results = df_results.sort_values('month_period')
    
    final_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(final_output_csv_path, index=False)
    
    print(f"\n✅ Analysis complete. Results saved to {final_output_csv_path.resolve()}")

if __name__ == "__main__":
    main()