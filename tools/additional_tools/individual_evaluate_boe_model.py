import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# --- Configuration ---
# Use the output path from the combined scoring script
INPUT_CSV = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/combined_monthly_scores.csv")
PLOT_OUTPUT_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent/data/analysis_results/final_hawkishness_analysis.png")

REFERENCE_COL = 'scores_reference'
SCORED_COLS = ['scores_minutes', 'scores_speeches', 'scores_publications']
ALL_SCORE_COLS = SCORED_COLS + [REFERENCE_COL]

def calculate_all_mses(df_filtered):
    """Calculates the MSE for each scored source against the reference score."""
    mse_results = {}
    
    if REFERENCE_COL not in df_filtered.columns:
        print(f"❌ Error: Reference column '{REFERENCE_COL}' not found in data.")
        return mse_results
        
    print("\nCalculating MSEs against Reference Score:")
    for col in SCORED_COLS:
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
    """Loads, filters to the first non-NA month, calculates MSE, and plots."""
    print(f"1. Loading combined data from: {INPUT_CSV}")
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: Combined scores file not found at {INPUT_CSV}. Cannot proceed.")
        return

    # --- Data Preparation and Filtering ---
    try:
        # Convert 'YYYY,MM' to datetime objects
        df['Date'] = pd.to_datetime(df['month_period'], format='%Y,%m')
    except Exception as e:
        print(f"❌ Error converting 'month_period' to datetime: {e}")
        return

    # 1. Determine the earliest non-NA starting month (for ANY of the non-reference scores)
    df_scores_only = df.dropna(subset=SCORED_COLS, how='all').copy()

    if df_scores_only.empty:
        print("❌ Error: No months found with scores in Minutes, Speeches, or Publications.")
        return
        
    # Get the minimum date where at least one score exists
    start_date = df_scores_only['Date'].min()
    
    # Filter the original DataFrame to start from this date
    df_filtered = df[df['Date'] >= start_date].copy()
    
    # Drop final rows where all relevant columns are NaN (to clean up the end/gaps)
    df_filtered = df_filtered.dropna(subset=ALL_SCORE_COLS, how='all')
    df_filtered = df_filtered.sort_values('Date')
    
    print(f"2. Data filtered to start from the earliest scored month: {start_date.strftime('%Y-%m')}")
    
    # 3. Calculate MSEs
    mse_values = calculate_all_mses(df_filtered)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define labels and plot each series
    labels = {
        'scores_minutes': 'Minutes Score',
        'scores_speeches': 'Speeches Score (Aggregated)',
        'scores_publications': 'Publications Score',
        'scores_reference': 'Reference Score'
    }

    print("3. Generating plot with MSE insert box...")

    # Plot all series in the filtered data
    for col in ALL_SCORE_COLS:
        ax.plot(df_filtered['Date'], df_filtered[col], 
                label=labels[col], 
                marker='o' if col != REFERENCE_COL else 'X', 
                linestyle='-' if col != REFERENCE_COL else '--', 
                linewidth=2, markersize=5, alpha=0.8)

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
    for col, mse in mse_values.items():
        source_name = labels[col].split(' ')[0] # Use only the first word (Minutes, Speeches, Pubs)
        
        # Determine the status for the display
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
    print(f"\n4. Saved final plot to {PLOT_OUTPUT_PATH.resolve()}")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot_combined_scores()