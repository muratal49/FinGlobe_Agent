import pandas as pd
import json
from pathlib import Path
import re
import argparse
from datetime import datetime

# --- CONSOLIDATED CONFIGURATION ---
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent")

# INPUT PATHS
PUBLICATIONS_CSV_INPUT_PATH = BASE_PATH / "boe_publications.csv"
MINUTES_JSON_INPUT_PATH = BASE_PATH / "tools/mpc_minutes_boe.json"
SPEECHES_CSV_INPUT_PATH = BASE_PATH / "data/raw/boe_filtered_speeches_conclusion.csv"
REFERENCE_CSV_INPUT_PATH = BASE_PATH / "data/raw/boe_reference_scores.csv"

# OUTPUT PATHS (All are now simple monthly aggregated JSONs)
PUBLICATIONS_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/publications_boe_monthly.json"
MINUTES_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/minutes_boe_monthly.json"
SPEECHES_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/speeches_boe_monthly.json"
REFERENCE_JSON_OUTPUT_PATH = BASE_PATH / "data/raw/reference_boe_monthly.json"

# Column names and cleaning artifact
PUB_DATE_COLUMN_NAME = 'published'
SPEECH_DATE_COLUMN_NAME = 'date'
SPEECH_TEXT_COLUMN_NAME = 'conclusion_text'
PUB_SUMMARY_COLUMN_NAME = 'summary'
REF_DATE_COLUMN_NAME = 'Hawkishness Date'
REF_SCORE_COLUMN_NAME = 'Hawkishness'
CHARACTERS_TO_CLEAN = "Home/\n " 
# ----------------------------------


def aggregate_and_filter_data(df, date_col, text_col, output_path, start_date_filter_dt=None, aggregation_func='text_join'):
    """Generic function to load, clean, aggregate, filter by date, and save data to JSON."""
    
    # Check for required columns
    required_cols = [date_col, text_col]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Input CSV must contain '{date_col}' and '{text_col}' columns.")
        return

    # --- Data Cleaning and Date Conversion ---
    df['Date_DT'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    df = df.dropna(subset=['Date_DT'])
    
    if text_col:
        # Clean individual texts/summaries
        df['value_clean'] = df[text_col].astype(str).apply(lambda x: x.lstrip(CHARACTERS_TO_CLEAN))
    else:
        # For numerical scores, use the score column directly
        df['value_clean'] = df[text_col]
    
    # --- Date Filtering ---
    if start_date_filter_dt:
        df = df[df['Date_DT'] >= start_date_filter_dt].copy()
        
    if df.empty:
        print("⚠️ Warning: No data remains after date filtering.")
        return

    # Create the monthly grouping key (YYYY-MM)
    df['Month'] = df['Date_DT'].dt.strftime('%Y-%m')

    # --- Aggregation ---
    if aggregation_func == 'text_join':
        # Aggregation for text data: join all texts with separator
        df_monthly = df.groupby('Month')['value_clean'].apply(lambda x: '\n\n\n'.join(x)).reset_index()
    elif aggregation_func == 'mean':
        # Aggregation for numerical data: calculate the mean
        df_monthly = df.groupby('Month')['value_clean'].mean().reset_index()
        df_monthly['value_clean'] = df_monthly['value_clean'].apply(lambda x: round(float(x), 6))

    # --- Format for JSON Output ---
    json_output = df_monthly.set_index('Month')['value_clean'].to_dict()
    
    # --- Save JSON ---
    print(f"3. Saving processed monthly data to JSON: {output_path}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=4)
        
        print(f"✅ JSON generated with {len(json_output)} monthly entries.")
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def aggregate_publications_monthly(start_date_dt):
    """Handles publications CSV loading and aggregation."""
    print(f"--- Running: Aggregate Publications Monthly ---")
    try:
        df = pd.read_csv(PUBLICATIONS_CSV_INPUT_PATH)
        aggregate_and_filter_data(
            df, PUB_DATE_COLUMN_NAME, PUB_SUMMARY_COLUMN_NAME, PUBLICATIONS_JSON_OUTPUT_PATH, start_date_dt, 'text_join'
        )
    except Exception as e:
        print(f"❌ Error in Publications pipeline: {e}")


def process_speeches_csv(start_date_dt):
    """Handles speeches CSV loading and aggregation."""
    print(f"\n--- Running: Aggregate Speeches Monthly ---")
    try:
        df = pd.read_csv(SPEECHES_CSV_INPUT_PATH)
        aggregate_and_filter_data(
            df, SPEECH_DATE_COLUMN_NAME, SPEECH_TEXT_COLUMN_NAME, SPEECHES_JSON_OUTPUT_PATH, start_date_dt, 'text_join'
        )
    except Exception as e:
        print(f"❌ Error in Speeches pipeline: {e}")


def aggregate_reference_scores_monthly(start_date_dt):
    """Handles reference scores CSV loading and aggregation."""
    print(f"\n--- Running: Aggregate Reference Scores Monthly ---")
    try:
        df = pd.read_csv(REFERENCE_CSV_INPUT_PATH)
        df = df.rename(columns={REF_SCORE_COLUMN_NAME: 'Hawkishness Score (Value)'}) # Rename score column for generic use
        aggregate_and_filter_data(
            df, REF_DATE_COLUMN_NAME, 'Hawkishness Score (Value)', REFERENCE_JSON_OUTPUT_PATH, start_date_dt, 'mean'
        )
    except Exception as e:
        print(f"❌ Error in Reference Scores pipeline: {e}")


def clean_minutes_json(start_date_dt):
    """Handles minutes JSON loading, cleaning, and aggregation."""
    print(f"\n--- Running: Aggregate Minutes Monthly ---")
    try:
        with MINUTES_JSON_INPUT_PATH.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading Minutes JSON: {e}")
        return

    # Convert the JSON dictionary back to a DataFrame for easier processing
    df = pd.DataFrame(list(raw_data.items()), columns=['date', 'text'])
    
    # Process Minutes: Date formatting is complex (ISO 8601 keys), so we run conversion here
    df = df.rename(columns={'date': 'minutes_date', 'text': 'minutes_text'})
    
    aggregate_and_filter_data(
        df, 'minutes_date', 'minutes_text', MINUTES_JSON_OUTPUT_PATH, start_date_dt, 'text_join'
    )


def main():
    parser = argparse.ArgumentParser(description="Data preparation tool: Cleans, filters, and aggregates all BoE source documents.")
    parser.add_argument(
        "--start-date", 
        type=str, 
        required=True, 
        help="Start date (YYYY-MM-DD) to filter all data sources from."
    )
    args = parser.parse_args()
    
    try:
        # Convert start date string to datetime object for filtering
        start_date_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Fatal Error: Invalid start date format. Must be YYYY-MM-DD.")
        sys.exit(1)

    print(f"Starting Data Preparation Pipeline, filtering all data from: {args.start_date}")

    aggregate_publications_monthly(start_date_dt)
    process_speeches_csv(start_date_dt)
    aggregate_reference_scores_monthly(start_date_dt)
    clean_minutes_json(start_date_dt)


if __name__ == "__main__":
    main()