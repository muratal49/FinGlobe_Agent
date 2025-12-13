#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================
# CONFIG
# ================================================
BASE = "/Users/murat/Desktop/Capstone/FinGlobe_Agent"
DIRS = [
    Path(f"{BASE}/output_boc_only"),
    Path(f"{BASE}/output_noWCB"),
]

BANK_NAMES = {
    "BOE": "Bank of England",
    "BOC": "Bank of Canada"
}

DOC_NAMES = {
    "MIN": "Minutes",
    "SPE": "Speeches",
    "MER": "Merged Documents"
}

# ================================================
# PLOT FUNCTION
# ================================================
def plot_csv(csv_path, out_path):
    df = pd.read_csv(csv_path)

    if not {"month", "model_score", "reference_score"} <= set(df.columns):
        print(f"⚠ Skipping {csv_path.name}, missing required columns")
        return

    # ====== Parse filename ======
    parts = csv_path.stem.split("_")
    if len(parts) < 4:
        print(f"⚠ Skipping {csv_path.name}: invalid format")
        return

    bank_code = parts[2]   # BOE / BOC
    doc_code  = parts[3]   # MIN / SPE / MER

    if bank_code not in BANK_NAMES or doc_code not in DOC_NAMES:
        print(f"⚠ Skipping {csv_path.name}: unknown bank/doc type")
        return

    bank_name = BANK_NAMES[bank_code]
    doc_name  = DOC_NAMES[doc_code]

    mse = float(((df["model_score"] - df["reference_score"])**2).mean())

    # ====== PLOT ======
    plt.figure(figsize=(14,6))

    plt.plot(df["month"], df["reference_score"],
             color="black", linewidth=3, label="Reference")

    plt.plot(df["month"], df["model_score"],
             color="blue", linewidth=2, label="Model Score")

    plt.xticks(rotation=60)
    plt.grid(alpha=0.3)

    title = f"{bank_name} – {doc_name}"
    plt.title(title, fontsize=16)

    # -------- Fix legend: move to upper-right --------
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8)

    # -------- MSE Box: stays in upper-left --------
    plt.annotate(
        f"MSE = {mse:.4f}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✔ Saved plot: {out_path.name}")


# ================================================
# MAIN
# ================================================
def main():
    print("\n📊 Regenerating ALL plots...\n")

    for directory in DIRS:
        if not directory.exists():
            print(f"⚠ Directory missing: {directory}")
            continue

        for csv in directory.glob("*.csv"):
            png = csv.with_suffix(".png")
            plot_csv(csv, png)

    print("\n🎉 ALL PLOTS GENERATED.\n")


if __name__ == "__main__":
    main()
