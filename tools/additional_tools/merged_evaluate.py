import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# --- Configuration ---
# File paths for all inputs/outputs
INPUT_CSV = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/combined_monthly_scores.csv")
MERGED_CSV = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/scored_merged_text.csv") 
PLOT_OUTPUT_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/final_hawkishness_analysis.png")

REFERENCE_COL = 'scores_reference'
SCORED_COLS = ['scores_minutes', 'scores_speeches', 'scores_publications']
NEW_MERGED_COL = 'scores_merged_text'
ALL_PLOT_COLS = SCORED_COLS + [NEW_MERGED_COL, REFERENCE_COL]


def calculate_all_mses(df_filtered):
    """Calculates the MSE for each scored source against the reference score."""
    mse_results = {}
    
    if REFERENCE_COL not in df_filtered.columns:
        print(f"❌ Error: Reference column '{REFERENCE_COL}' not found in data.")
        return mse_results
        
    print("\nCalculating MSEs against Reference Score:")
    for col in SCORED_COLS + [NEW_MERGED_COL]:
        # Filter to months where BOTH the scored source and reference have non-NaN values
        df_match = df_filtered.dropna(subset=[col, REFERENCE_COL]).copy()
        
        if len(df_match) > 0:
            mse = mean_squared_error(
                df_match[REFERENCE_COL],
                df_match[col]
            )
            mse_results[col] = mse
            print(f"  {col} (Matching Months: {len(df_match)}): {mse:.4f}")
        else:
            mse_results[col] = None
            print(f"  {col}: N/A (No matching months with reference data).")
            
    return mse_results

def analyze_and_plot_combined_scores():
    """Loads, filters, calculates MSE, and plots the combined scores."""
    print(f"1. Loading base combined data from: {INPUT_CSV}")
    print(f"2. Loading merged text scores from: {MERGED_CSV}")
    
    try:
        df_base = pd.read_csv(INPUT_CSV)
        df_merged_new = pd.read_csv(MERGED_CSV)
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found. Ensure all scoring/prep scripts ran successfully. {e}")
        return

    # --- Combine DataFrames (Base + New Merged Scores) ---
    df_base['Date'] = pd.to_datetime(df_base['month_period'], format='%Y,%m')
    df_merged_new['Date'] = pd.to_datetime(df_merged_new['month_period'], format='%Y,%m')

    # Merge the new merged scores into the base table (Outer join on Date)
    df = df_base.merge(
        df_merged_new[['Date', NEW_MERGED_COL]], 
        on='Date', 
        how='outer'
    )
    
    # --- Determine Start Date and Filter ---
    # Define all non-reference columns for finding the start date
    all_scored_cols_for_filter = SCORED_COLS + [NEW_MERGED_COL]
    
    df_scores_only = df.dropna(subset=all_scored_cols_for_filter, how='all').copy()

    if df_scores_only.empty:
        print("❌ Error: No months found with scores in any source.")
        return
        
    start_date = df_scores_only['Date'].min()
    
    # Filter the original DataFrame to start from this date
    df_filtered = df[df['Date'] >= start_date].copy()
    
    # Drop final rows where all relevant columns are NaN (to clean up the end/gaps)
    df_filtered = df_filtered.dropna(subset=ALL_PLOT_COLS, how='all')
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
    
    print(f"3. Data filtered to start from the earliest scored month: {start_date.strftime('%Y-%m')}")
    
    # 4. Calculate MSEs
    mse_values = calculate_all_mses(df_filtered)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define labels and plot each series
    labels = {
        'scores_minutes': 'Minutes Score',
        'scores_speeches': 'Speeches Score',
        'scores_publications': 'Publications Score',
        'scores_merged_text': 'Merged Minutes & Speeches', # NEW LABEL
        'scores_reference': 'Reference Score'
    }

    print("4. Generating plot with MSE insert box...")

    for col in ALL_PLOT_COLS:
        is_merged = (col == NEW_MERGED_COL)
        
        # Plotting logic for all lines
        ax.plot(df_filtered['Date'], df_filtered[col], 
                label=labels[col], 
                marker='o' if col not in [NEW_MERGED_COL, REFERENCE_COL] else ('*' if is_merged else 'X'), 
                linestyle='-' if col != REFERENCE_COL else '--', 
                # Bolder and thicker line for the merged score
                linewidth=3.5 if is_merged else 2, 
                alpha=0.9 if is_merged else 0.7,
                # Set zorder to ensure the merged line is on top
                zorder=10 if is_merged else (1 if col == REFERENCE_COL else 5)
        )

    # --- Customizations ---
    ax.set_title(f'Hawkishness Comparison (Starting {start_date.strftime("%Y-%m")})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Mean Monthly Hawkishness Score', fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis ticks
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # --- INSERT MSE BOX ---
    mse_text = "MSE vs. Reference:\n"
    # Order the display: Merged first, then Minutes, Speeches, Pubs
    display_order = [NEW_MERGED_COL] + SCORED_COLS
    
    for col in display_order:
        source_name = labels[col].split(' ')[0] 
        mse = mse_values.get(col)
        
        if mse is not None:
             mse_text += f"  {source_name}: {mse:.4f}\n"
        else:
             mse_text += f"  {source_name}: N/A\n"
             
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    # Position the box in the upper left corner
    ax.text(0.02, 0.98, mse_text.strip(), transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=props)
    
    # --- Save and Display ---
    PLOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"\n5. Saved final plot to {PLOT_OUTPUT_PATH.resolve()}")
    plt.show()

def main():
    """Executes the final combined analysis and plotting routine."""
    analyze_and_plot_combined_scores()


if __name__ == "__main__":
    main()