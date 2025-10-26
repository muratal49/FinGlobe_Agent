import pandas as pd
import json
from pathlib import Path
import re

# --- CONSOLIDATED CONFIGURATION ---
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")

# INPUT PATHS
PUBLICATIONS_CSV_INPUT_PATH = BASE_PATH / "data/raw/boe_publications.csv"
MINUTES_JSON_INPUT_PATH = BASE_PATH / "data/raw/minutes_boe.json"
SPEECHES_CSV_INPUT_PATH = BASE_PATH / "data/raw/boe_filtered_speeches_conclusion.csv"
REFERENCE_CSV_INPUT_PATH = BASE_PATH / "data/raw/boe_reference_scores.csv" 
# OUTPUT PATHS (All are now monthly aggregated JSONs)
PUBLICATIONS_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/publications_boe_clean.json"
MINUTES_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/minutes_boe_clean.json"
SPEECHES_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/speeches_boe_clean.json"
REFERENCE_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/reference_boe_monthly.json" 

# Column names and cleaning artifact
PUB_DATE_COLUMN_NAME = 'published'
SPEECH_DATE_COLUMN_NAME = 'date'
SPEECH_TEXT_COLUMN_NAME = 'conclusion_text'
PUB_SUMMARY_COLUMN_NAME = 'summary'
REF_DATE_COLUMN_NAME = 'Hawkishness Date' # Reference CSV Date Column
REF_SCORE_COLUMN_NAME = 'Hawkishness'    # Reference CSV Score Column
CHARACTERS_TO_CLEAN = "Home/\n " 
# ----------------------------------


def aggregate_publications_monthly():
    """Loads publications CSV, aggregates summaries by month, cleans text, and saves to JSON."""
    print(f"--- Running: Aggregate Publications Monthly (YYYY-MM) ---")
    print(f"1. Loading CSV from: {PUBLICATIONS_CSV_INPUT_PATH}")
    
    try:
        df = pd.read_csv(PUBLICATIONS_CSV_INPUT_PATH)
    except Exception as e:
        print(f"❌ Error loading Publications CSV: {e}")
        return

    required_cols = [PUB_DATE_COLUMN_NAME, PUB_SUMMARY_COLUMN_NAME]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Publications CSV must contain '{PUB_DATE_COLUMN_NAME}' and '{PUB_SUMMARY_COLUMN_NAME}' columns.")
        return

    # --- Data Cleaning and Aggregation ---
    df['Date_DT'] = pd.to_datetime(df[PUB_DATE_COLUMN_NAME], errors='coerce', utc=True)
    df = df.dropna(subset=['Date_DT'])

    df['summary_clean'] = df[PUB_SUMMARY_COLUMN_NAME].astype(str).apply(lambda x: x.lstrip(CHARACTERS_TO_CLEAN))
    
    # Create the monthly grouping key (YYYY-MM)
    df['Month'] = df['Date_DT'].dt.strftime('%Y-%m')

    # Aggregate: Group by Month and join all clean summaries
    print("2. Aggregating all summaries within each month...")
    df_monthly = df.groupby('Month')['summary_clean'].apply(lambda x: '\n\n\n'.join(x)).reset_index()
    
    # Format for JSON Output: {"YYYY-MM": "aggregated_summary_text"}
    json_output = df_monthly.set_index('Month')['summary_clean'].to_dict()
    
    # --- Save JSON ---
    print(f"3. Saving processed monthly data to JSON: {PUBLICATIONS_JSON_OUTPUT_PATH}")
    
    try:
        PUBLICATIONS_JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PUBLICATIONS_JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        
        print(f"✅ Publications JSON generated with {len(json_output)} monthly entries.")
        
    except Exception as e:
        print(f"❌ Error saving Publications JSON: {e}")


def clean_minutes_json():
    """Reads the minutes JSON, aggregates all entries by month (YYYY-MM), cleans text, and saves."""
    print(f"\n--- Running: Clean Minutes JSON (YYYY-MM Aggregation) ---")
    print(f"1. Loading JSON from: {MINUTES_JSON_INPUT_PATH}")
    
    try:
        with MINUTES_JSON_INPUT_PATH.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading Minutes JSON: {e}")
        return

    monthly_data = {}
    
    print(f"2. Processing {len(raw_data)} entries: reformatting to YYYY-MM and aggregating text...")
    
    for iso_date_str, text_value in raw_data.items():
        if not iso_date_str:
            continue
            
        try:
            # --- Date Formatting (ISO 8601 to YYYY-MM) ---
            dt_obj = pd.to_datetime(iso_date_str)
            month_key = dt_obj.strftime('%Y-%m')
            
            # --- Text Cleaning ---
            text_value = str(text_value)
            cleaned_text = text_value.lstrip(CHARACTERS_TO_CLEAN) if text_value else ""
            
            # --- Aggregate by Month ---
            if month_key in monthly_data:
                monthly_data[month_key] += "\n\n\n" + cleaned_text
            else:
                monthly_data[month_key] = cleaned_text
                
        except ValueError:
            print(f"⚠️ Skipping entry due to invalid date string: {iso_date_str}")
            continue

    # --- Save JSON ---
    print(f"3. Saving cleaned monthly JSON file to: {MINUTES_JSON_OUTPUT_PATH}")
    
    try:
        MINUTES_JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MINUTES_JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(monthly_data, f, indent=4)
        
        print(f"✅ Minutes JSON cleaned and saved with {len(monthly_data)} monthly entries.")
        
    except Exception as e:
        print(f"❌ Error saving Minutes JSON: {e}")
        

def process_speeches_csv():
    """Loads speeches CSV, aggregates all speech text by month (YYYY-MM), cleans text, and saves."""
    print(f"\n--- Running: Process Speeches CSV (YYYY-MM Aggregation) ---")
    print(f"1. Loading CSV from: {SPEECHES_CSV_INPUT_PATH}")
    
    try:
        df = pd.read_csv(SPEECHES_CSV_INPUT_PATH)
    except Exception as e:
        print(f"❌ Error loading Speeches CSV: {e}")
        return

    if not all(col in df.columns for col in [SPEECH_DATE_COLUMN_NAME, SPEECH_TEXT_COLUMN_NAME]):
        print(f"❌ Error: Speeches CSV must contain '{SPEECH_DATE_COLUMN_NAME}' and '{SPEECH_TEXT_COLUMN_NAME}' columns.")
        return

    print(f"2. Processing {len(df)} speeches: cleaning text and aggregating by month...")
    
    # --- Data Cleaning and Aggregation ---
    df['Date_DT'] = pd.to_datetime(df[SPEECH_DATE_COLUMN_NAME], errors='coerce')
    df = df.dropna(subset=['Date_DT'])
    
    df['text_clean'] = df[SPEECH_TEXT_COLUMN_NAME].astype(str).apply(lambda x: x.lstrip(CHARACTERS_TO_CLEAN))
    
    # Create the monthly grouping key (YYYY-MM)
    df['Month'] = df['Date_DT'].dt.strftime('%Y-%m')

    # Aggregate: Group by Month and join all clean texts
    df_monthly = df.groupby('Month')['text_clean'].apply(lambda x: '\n\n\n'.join(x)).reset_index()
    
    # Format for JSON Output: {"YYYY-MM": "aggregated_speech_text"}
    json_output = df_monthly.set_index('Month')['text_clean'].to_dict()

    # --- Save JSON ---
    print(f"3. Saving cleaned monthly JSON file to: {SPEECHES_JSON_OUTPUT_PATH}")
    
    try:
        SPEECHES_JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SPEECHES_JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        
        print(f"✅ Speeches JSON created with {len(json_output)} monthly entries.")
        
    except Exception as e:
        print(f"❌ Error saving Speeches JSON: {e}")


def aggregate_reference_scores_monthly():
    """Loads reference scores CSV, aggregates to monthly mean, and saves to JSON."""
    print(f"\n--- Running: Aggregate Reference Scores Monthly (YYYY-MM) ---")
    print(f"1. Loading CSV from: {REFERENCE_CSV_INPUT_PATH}")
    
    try:
        df = pd.read_csv(REFERENCE_CSV_INPUT_PATH)
    except Exception as e:
        print(f"❌ Error loading Reference CSV: {e}")
        return

    if not all(col in df.columns for col in [REF_DATE_COLUMN_NAME, REF_SCORE_COLUMN_NAME]):
        print(f"❌ Error: Reference CSV must contain '{REF_DATE_COLUMN_NAME}' and '{REF_SCORE_COLUMN_NAME}' columns.")
        return

    # --- Data Preparation and Aggregation ---
    df['Date_DT'] = pd.to_datetime(df[REF_DATE_COLUMN_NAME], errors='coerce', utc=True)
    df = df.dropna(subset=['Date_DT'])
    
    # Create the monthly grouping key (YYYY-MM)
    df['Month'] = df['Date_DT'].dt.strftime('%Y-%m')

    # Aggregate: Group by Month and calculate the MEAN score
    print("2. Aggregating scores by month (calculating mean)...")
    df_monthly = df.groupby('Month')[REF_SCORE_COLUMN_NAME].mean().reset_index()
    
    # Format for JSON Output: {"YYYY-MM": score_mean}
    # Convert NumPy float to native Python float before saving to JSON (good practice)
    json_output = df_monthly.set_index('Month')[REF_SCORE_COLUMN_NAME].apply(lambda x: round(float(x), 6)).to_dict()

    # --- Save JSON ---
    print(f"3. Saving processed monthly scores to JSON: {REFERENCE_JSON_OUTPUT_PATH}")
    
    try:
        REFERENCE_JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REFERENCE_JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        
        print(f"✅ Reference JSON created with {len(json_output)} monthly entries.")
        
    except Exception as e:
        print(f"❌ Error saving Reference JSON: {e}")


def main():
    """Execute all four data preparation tasks."""
    aggregate_publications_monthly()
    clean_minutes_json()
    process_speeches_csv()
    aggregate_reference_scores_monthly() # RUN NEW FUNCTION


if __name__ == "__main__":
    main()