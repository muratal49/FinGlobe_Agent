import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import mean_squared_error

# --- Configuration ---
# File paths for all inputs/outputs
INPUT_CSV = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/combined_monthly_scores.csv")
MERGED_CSV = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/scored_merged_text.csv") 
PLOT_OUTPUT_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/final_reference_vs_merged_plot.png")

REFERENCE_COL = 'scores_reference'
NEW_MERGED_COL = 'scores_merged_text'
# --- CRITICAL FIX: Only plot these two columns ---
PLOTTED_COLS = [NEW_MERGED_COL, REFERENCE_COL]
# --- END CRITICAL FIX ---

# Columns required for the start date filter and general MSE check
SCORED_COLS_FOR_FILTER = ['scores_minutes', 'scores_speeches', 'scores_publications', NEW_MERGED_COL]


def calculate_all_mses(df_filtered):
    """Calculates the MSE for the Merged Score against the Reference Score."""
    mse_results = {}
    
    if REFERENCE_COL not in df_filtered.columns or NEW_MERGED_COL not in df_filtered.columns:
        print(f"❌ Error: Required columns for MSE calculation missing.")
        return None
        
    # Filter to months where BOTH the Merged Score and Reference have non-NaN values
    df_match = df_filtered.dropna(subset=[NEW_MERGED_COL, REFERENCE_COL]).copy()
    
    if len(df_match) > 0:
        mse = mean_squared_error(
            df_match[REFERENCE_COL],
            df_match[NEW_MERGED_COL]
        )
        print(f"\n📊 Calculated MSE for Merged vs. Reference (Matching Months: {len(df_match)}): {mse:.4f}")
        # The function must return a dictionary matching the name expected in the plot insert.
        mse_results[NEW_MERGED_COL] = mse 
        return mse_results
    else:
        print("⚠️ Warning: No overlapping months found for Merged Text vs. Reference data.")
        return None


def analyze_and_plot_combined_scores(start_date_filter_str):
    """Loads, filters, calculates MSE, and plots only the Merged and Reference scores."""
    print(f"1. Loading base data from: {INPUT_CSV}")
    print(f"2. Loading merged text scores from: {MERGED_CSV}")
    
    try:
        df_base = pd.read_csv(INPUT_CSV)
        df_merged_new = pd.read_csv(MERGED_CSV)
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found. Ensure scoring/prep scripts ran successfully. {e}")
        return

    # --- Combine DataFrames ---
    df_base['Date'] = pd.to_datetime(df_base['month_period'], format='%Y,%m', errors='coerce')
    df_merged_new['Date'] = pd.to_datetime(df_merged_new['month_period'], format='%Y,%m', errors='coerce')

    df = df_base.merge(df_merged_new[['Date', NEW_MERGED_COL]], on='Date', how='outer')
    
    # --- Determine Start Date and Filter ---
    try:
        start_date_manual = pd.to_datetime(start_date_filter_str)
    except ValueError:
        print(f"❌ Error: Invalid start date format provided: {start_date_filter_str}. Using auto-start.")
        df_scores_only = df.dropna(subset=SCORED_COLS_FOR_FILTER, how='all').copy()
        start_date = df_scores_only['Date'].min()
    else:
        start_date = start_date_manual
    
    # Filter data to start from the calculated or manual date
    df_filtered = df[df['Date'] >= start_date].copy()
    
    # Drop rows where BOTH PLOTTED columns are NaN (ensures cleaner axes)
    df_filtered = df_filtered.dropna(subset=PLOTTED_COLS, how='all')
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    
    if df_filtered.empty:
        print("❌ Error: No data points available after filtering. Plotting skipped.")
        return
        
    print(f"3. Plotting data filtered to start from: {start_date.strftime('%Y-%m')}")
    
    # 4. Calculate MSE (returns a dict: {NEW_MERGED_COL: mse_value})
    mse_values = calculate_all_mses(df_filtered)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    labels = {
        NEW_MERGED_COL: 'Merged Minutes & Speeches',
        REFERENCE_COL: 'Reference Score'
    }

    print("4. Generating plot with MSE insert box...")

    # Plot ONLY the two required series (PLOTTED_COLS)
    for col in PLOTTED_COLS:
        is_merged = (col == NEW_MERGED_COL)
        
        ax.plot(df_filtered['Date'], df_filtered[col], 
                label=labels[col], 
                marker='*' if is_merged else 'X', 
                linestyle='-' if is_merged else '--', 
                linewidth=3.5 if is_merged else 2, 
                alpha=0.9,
                zorder=10 if is_merged else 5 
        )

    # --- Customizations and MSE Box ---
    ax.set_title(f'Hawkishness Comparison: Merged Score vs. Reference (Starting {start_date.strftime("%Y-%m")})', fontsize=16)
    ax.set_xlabel('Date (Year-Month)', fontsize=12)
    ax.set_ylabel('Mean Monthly Hawkishness Score', fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis ticks to Year-Month (YYYY-MM)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # --- INSERT MSE BOX ---
    if mse_values and mse_values.get(NEW_MERGED_COL) is not None:
        mse_value = mse_values[NEW_MERGED_COL]
        mse_text = f"MSE vs. Reference:\n{mse_value:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        
        ax.text(0.02, 0.95, mse_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=props)
    
    # --- Save and Display ---
    PLOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"\n5. Saved final plot to {PLOT_OUTPUT_PATH.resolve()}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analysis and plotting script for combined monthly scores.")
    parser.add_argument("--start-date", type=str, required=True, help="The date (YYYY-MM-DD) from which the analysis and plot should start.")
    args = parser.parse_args()
    
    analyze_and_plot_combined_scores(args.start_date)

if __name__ == "__main__":
    main()