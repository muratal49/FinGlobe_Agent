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

# --- Configuration ---
load_dotenv() # Load variables from .env file
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    exit()

# File paths
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")
# Inputs: 
MINUTES_CLEAN_PATH = BASE_PATH / "data/raw/minutes_boe_clean.json"
SPEECHES_CLEAN_PATH = BASE_PATH / "data/raw/speeches_boe_clean.json"
# The CSV containing the final list of months from the previous merging step (used to define the date range)
MONTHS_REFERENCE_CSV = BASE_PATH / "data/analysis_results/scored_merged_text.csv"
# Output:
OUTPUT_CSV_PATH = BASE_PATH / "data/analysis_results/openai_merged_scores_analysis.csv"


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
1. Identify the frequency and context of terms from Panel A (Hawkish/Tightening bias) versus Panel B (Dovish/Easing bias).
2. Panel A terms (A1 and A2) suggest a move toward or maintenance of restrictive policy.
3. Panel B terms (B1 and B2) suggest a move toward or maintenance of accommodative policy.

CRITICAL REQUIREMENT: Your justification MUST explicitly mention and contextualize at least three specific terms found in the provided text that align with your analysis (e.g., mention both a term from Panel A and a term from Panel B to demonstrate balance).

Scoring Guidelines:
- **Score:** A single floating-point number between -1.0 (Extreme Dovishness) and +1.0 (Extreme Hawkishness). A score of 0.0 is Neutral.
- **Justification:** A professional, clear, and concise justification based on the identified keywords and monetary policy context, strictly limited to {MAX_WORDS_JUSTIFICATION} words.

Scoring Panels:
{json.dumps(SCORING_CRITERIA, indent=4)}

You MUST return your answer as a single, valid JSON object with the keys 'score' and 'justification'.
Example Output:
{{
  "score": 0.45,
  "justification": "The analysis revealed a moderately hawkish intent, primarily driven by persistent concerns over inflation expectation (Panel A1). The text explicitly discussed tightening (Panel A2) financial conditions, noting that while economic growth (Panel B1) was soft, the need to anchor (Panel A2) price stability remained paramount for the MPC's mandate."
}}
"""
# --- End of System Prompt ---

def load_all_monthly_text(months_to_analyze):
    """Loads text from JSON files and performs the forward-looking cumulative merge for specified months."""
    print("1. Loading all necessary text sources for merging...")
    try:
        with MINUTES_CLEAN_PATH.open("r", encoding="utf-8") as f:
            minutes_data_daily = json.load(f)
        with SPEECHES_CLEAN_PATH.open("r", encoding="utf-8") as f:
            speeches_data_daily = json.load(f)
    except Exception as e:
        print(f"❌ Error loading cleaned JSON files: {e}")
        return {} 

    # Convert keys to YYYY-MM-DD datetime objects for correct chronological comparison
    minutes_dt = {pd.to_datetime(k): v for k, v in minutes_data_daily.items() if v}
    speeches_dt = {pd.to_datetime(k): v for k, v in speeches_data_daily.items() if v}
    
    # Identify policy anchors (Minutes dates)
    policy_anchors = sorted(list(minutes_dt.keys()))
    
    final_monthly_text = {}
    separator = "\n\n\n[NEW SOURCE TEXT]\n\n\n" 

    print("2. Re-merging text blocks based on Minutes anchors...")

    # Logic to aggregate speeches until the next Minutes date
    for i, anchor_date in enumerate(policy_anchors):
        month_key = anchor_date.strftime('%Y-%m')
        
        # We only need to generate the text if the month is in our required list
        if month_key not in months_to_analyze:
            continue
            
        # Determine the window start date (day after the previous anchor)
        if i == 0:
            accumulation_start_date = pd.to_datetime(min(speeches_dt.keys(), default=anchor_date)) - pd.Timedelta(days=1)
        else:
            accumulation_start_date = policy_anchors[i-1]
        
        # 2b. Start with the Minutes text for this month
        current_block = minutes_dt[anchor_date]
        
        # 2c. Accumulate all speech texts that fall within the current window:
        speeches_to_merge = []
        for speech_date, text in sorted(speeches_dt.items()):
            # Accumulate speeches released (Start date EXCLUSIVE) < Speech Date <= Current Anchor Date
            if accumulation_start_date < speech_date <= anchor_date:
                speeches_to_merge.append(text)
        
        # 2d. Finalize the month's text block (Minutes + Speeches)
        if speeches_to_merge:
            speeches_text = separator.join(speeches_to_merge)
            current_block += separator + speeches_text
        
        # Assign the fully merged text block to the month key
        final_monthly_text[month_key] = current_block
    
    return final_monthly_text


def generate_llm_response(month, text, client):
    """Sends prompt to OpenAI API and returns parsed score and justification."""
    
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
        
        # Parse the JSON response
        result = json.loads(content)
        
        score = float(result.get('score', 0.0))
        justification = result.get('justification', 'No justification provided.')
        
        return score, justification
        
    except Exception as e:
        print(f"Error during API call for {month}: {e}")
        return None, f"API Error: {str(e)}"


def main():
    print("--- Running OpenAI Scoring and Justification Pipeline ---")
    
    # 1. Get List of Months to Analyze from the Scored Merged CSV
    try:
        df_ref = pd.read_csv(MONTHS_REFERENCE_CSV)
        # Extract unique month strings (YYYY,MM) and convert to YYYY-MM format
        months_to_analyze_list = df_ref['month_period'].str.replace(',', '-').unique().tolist()
        print(f"Loaded {len(months_to_analyze_list)} months from reference CSV for analysis.")
    except Exception as e:
        print(f"❌ Error loading months from reference CSV: {e}")
        return
        
    # 2. Load and Prepare Merged Text Data for those months
    merged_texts = load_all_monthly_text(months_to_analyze_list)
    if not merged_texts:
        print("❌ Pipeline halted: Could not generate merged text blocks.")
        return

    # 3. Initialize OpenAI Client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    analysis_records = []
    
    # 4. Process and Score Each Month
    print(f"Processing {len(merged_texts)} months with LLM...")
    for month, text in tqdm(merged_texts.items(), desc="Scoring Months"):
        score, justification = generate_llm_response(month, text, client)
        
        analysis_records.append({
            'month_period': month.replace('-', ','),
            'scores_openai_merged': score,
            'openai_justification': justification
        })

    # 5. Save Results
    df_results = pd.DataFrame(analysis_records)
    
    # Final cleanup and sorting
    df_results = df_results.sort_values('month_period')
    
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n✅ Analysis complete. Results saved to {OUTPUT_CSV_PATH.resolve()}")
    print(df_results.head().to_markdown(index=False))

if __name__ == "__main__":
    main()