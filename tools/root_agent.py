#!/usr/bin/env python3
"""
ROOT AGENT PIPELINE: Orchestrates the entire data ingestion, scoring, and analysis workflow.
This version integrates the corrected tool paths for Roberta and OpenAI evaluation.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# --- Configuration ---
TOOL_DIR = Path(__file__).resolve().parent

# Define the relative paths to your existing tools
SCRAPER_TOOL = TOOL_DIR / "scrape_boe_speeches.py"
PREP_TOOL = TOOL_DIR / "preparing_scraped_docs.py"
# UPDATED TOOL PATHS
ROBERTA_SCORE_TOOL = TOOL_DIR / "roberta_merged_score_evaluate.py"
OPENAI_TOOL = TOOL_DIR / "openai_merge_score_justify.py" 

# NOTE: The default keywords will now be read from inside the scrape_boe_speeches.py file.
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
        # Print only a small snippet of stdout for success confirmation
        print(result.stdout.split('\n')[:5])
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
    # The script now ONLY accepts the start date
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for scraping (YYYY-MM-DD)."
    )
    args = parser.parse_args()

    print(f"Starting pipeline. Keywords assumed: {DEFAULT_KEYWORDS_FOR_LOG}")
    
    # --- 1. DATA INGESTION ---
    run_command([
        sys.executable, 
        str(SCRAPER_TOOL),
        "--start-date", args.start_date
    ], "1. SCRAPE: Ingest Raw Speeches Data")

    # --- 2. CONTEXT CLEANSING & TRANSFORMATION ---
    run_command([sys.executable, str(PREP_TOOL)], "2. PREPARE: Clean & Aggregate All Documents (MCP Context Setup)")
    
    # --- 3. CONTEXT ENRICHMENT (SCORING) ---
    
    # 3A. Roberta Scoring & Evaluation (NOW USING CORRECTED PATH)
    run_command([
        sys.executable, 
        str(ROBERTA_SCORE_TOOL),
        "--start-date", args.start_date
    ], "3A. SCORE/EVAL: Roberta Model Scoring & Final Plotting")

    # 3B. OpenAI Scoring & Justification (PATH IS CORRECTED)
    run_command([
        sys.executable, 
        str(OPENAI_TOOL),
        "--start-date", args.start_date
    ], "3B. SCORE/JUSTIFY: OpenAI LLM Analysis")
    
    print("\n\n✅ AGENT PIPELINE COMPLETE.")
    print("Final analysis CSV and PNG plot are in the 'data/analysis_results/' directory.")

if __name__ == "__main__":
    main()