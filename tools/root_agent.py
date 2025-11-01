#!/usr/bin/env python3
"""
ROOT AGENT PIPELINE: Orchestrates the entire data ingestion, scoring, and analysis workflow.
Ensures date prefixing across all outputs.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# --- Configuration ---
TOOL_DIR = Path(__file__).resolve().parent

# Define the relative paths to your existing tools
SPEECHES_SCRAPER_TOOL = TOOL_DIR / "scrape_boe_speeches.py"
MINUTES_SCRAPER_TOOL = TOOL_DIR / "meeting_scraper.py"
PREP_TOOL = TOOL_DIR / "preparing_scraped_docs.py"
SCORING_TOOL = TOOL_DIR / "speech_scoring.py" 
EVAL_PLOT_TOOL = TOOL_DIR / "roberta_merged_score_evaluate.py"
OPENAI_TOOL = TOOL_DIR / "openai_merge_score_justify.py"

DEFAULT_KEYWORDS_FOR_LOG = "Monetary Policy Committee, MPC, inflation (Default)"

def run_command(command, step_name):
    """Executes a command and checks for errors."""
    print(f"\n=======================================================")
    print(f"🚀 STEP: {step_name}")
    print(f"EXECUTING: {' '.join(command)}")
    print(f"=======================================================")
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        print("✅ SUCCESS.")
        print('\n'.join(result.stdout.split('\n')[:5]))
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {step_name}: Command failed.")
        print(f"--- Stdout ---\n{e.stdout}")
        print(f"--- Stderr ---\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERROR in {step_name}: Tool not found. Check path: {command[1]}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Root Agent Pipeline for BoE Monetary Policy Analysis."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for scraping and filtering (YYYY-MM-DD)."
    )
    args = parser.parse_args()

    date_prefix = args.start_date.replace('-', '')
    print(f"Starting pipeline. Date Prefix: {date_prefix}. Keywords assumed: {DEFAULT_KEYWORDS_FOR_LOG}")
    
    # --- 1. DATA INGESTION ---
    # 1A. Scrape Minutes Data
    run_command([
        sys.executable, str(MINUTES_SCRAPER_TOOL),
        "--start-date", args.start_date
    ], "1A. SCRAPE: Ingest Raw Minutes Data")

    # 1B. Scrape Speeches Data
    run_command([
        sys.executable, str(SPEECHES_SCRAPER_TOOL),
        "--start-date", args.start_date
    ], "1B. SCRAPE: Ingest Raw Speeches Data")

    # --- 2. CONTEXT CLEANSING & TRANSFORMATION (Creates the 4 monthly JSONs) ---
    run_command([
        sys.executable, str(PREP_TOOL),
        "--start-date", args.start_date 
    ], "2. PREPARE: Clean & Aggregate All Documents")
    
    # --- 3. CONTEXT ENRICHMENT (SCORING) ---
    
    # 3A. Roberta Scoring (Creates all base scores and the critical merged score CSVs)
    run_command([
        sys.executable, str(SCORING_TOOL),
        "--start-date", args.start_date 
    ], "3A. SCORE: Generate All Monthly Scores (Roberta)")

    # 3B. Roberta Evaluation & Plotting (Loads the CSVs created in 3A and generates the final plot/CSV)
    run_command([
        sys.executable, str(EVAL_PLOT_TOOL),
        "--start-date", args.start_date 
    ], "3B. EVAL/PLOT: Final Analysis and Plotting")


    # 3C. OpenAI Scoring & Justification
    run_command([
        sys.executable, str(OPENAI_TOOL),
        "--start-date", args.start_date 
    ], "3C. SCORE/JUSTIFY: OpenAI LLM Analysis")
    
    print("\n\n✅ AGENT PIPELINE COMPLETE.")
    print(f"Final outputs are prefixed with: {date_prefix}_")

if __name__ == "__main__":
    main()