import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# ✅ Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Keyword panels (guidance for analysis)
panel_A1 = ['inflation expectation', 'interest rate', 'bank rate', 'fund rate',
            'price', 'economic activity', 'inflation', 'employment']
panel_B1 = ['unemployment', 'growth', 'exchange rate', 'productivity',
            'deficit', 'demand', 'job market', 'monetary policy']
panel_A2 = ['anchor', 'cut', 'subdue', 'decline', 'decrease', 'reduce', 'low',
            'drop', 'fall', 'fell', 'decelerate', 'slow', 'pause', 'pausing',
            'stable', 'nonaccelerating', 'downward', 'tighten']
panel_B2 = ['ease', 'easing', 'rise', 'rising', 'increase', 'expand',
            'improve', 'strong', 'upward', 'raise', 'high', 'rapid']


# 1️⃣ Speech Analysis Function
def summarize_all():
    """
    Reads scraped speeches, produces a summary + analytical report,
    assigns hawkish/dovish score, groups by month, and plots results.
    """
    print("🧠 Starting analysis of all speeches...")

    csv_path = Path("data/raw/boe_filtered_speeches.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Speech file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    summaries = []
    total = len(df)

    for i, row in df.iterrows():
        print(f"→ [{i+1}/{total}] {row['title']} by {row['speaker']} ({row['date']})")

        prompt = f"""
        You are a senior central-bank analyst preparing an investor briefing
        on a Bank of England speech.

        1️⃣ Write a clear, factual summary (≈250 words) of the speech, focusing
        on key themes such as inflation, growth, interest rates, employment,
        and policy communication style.

        2️⃣ Then write a professional analytical paragraph (≈150–200 words)
        describing the tone, monetary-policy stance, and discussion of inflation,
        referencing the following economic indicator sets:
            - Policy indicators: {panel_A1}
            - Macroeconomic indicators: {panel_B1}
            - Dovish/softening signals: {panel_A2}
            - Hawkish/strengthening signals: {panel_B2}

        3️⃣ Finally, assign a hawkish/dovish score between -1 (very dovish)
        and +1 (very hawkish).

        Return only valid JSON in this exact structure:
        {{
            "date": "{row['date']}",
            "speaker": "{row['speaker']}",
            "title": "{row['title']}",
            "summary": "...",
            "analysis_report": "...",
            "hawkish_dovish_score": value
        }}

        --- SPEECH START ---
        {str(row['text'])[:6000]}
        --- END ---
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1200
            )
            content = response.choices[0].message.content.strip()
            summaries.append(content)
            print("   ✅ Completed")

        except Exception as e:
            print(f"   ❌ Error analyzing {row['title']}: {e}")
            continue

    # Save analyses
    out_dir = Path("data/analysis_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "speech_analysis_full.jsonl"

    with open(outfile, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s + "\n")

    print(f"\n💾 Saved {len(summaries)} analyses → {outfile}")

    # Create monthly averages
    df_json = pd.DataFrame([json.loads(line) for line in open(outfile)])
    df_json["date"] = pd.to_datetime(df_json["date"], errors="coerce")
    df_json["month"] = df_json["date"].dt.to_period("M")
    monthly = (
        df_json.groupby("month")["hawkish_dovish_score"]
        .mean()
        .reset_index(name="avg_hawkishness")
    )

    monthly_out = "data/analysis_results/speech_monthly_scores.csv"
    monthly.to_csv(monthly_out, index=False)
    print(f"📆 Saved monthly averaged hawkishness scores → {monthly_out}")

    # Plot monthly mean scores
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"].astype(str), monthly["avg_hawkishness"], marker="o", linewidth=2)
    plt.title("Average BoE Speech Hawkishness (Monthly Mean Scores)", fontsize=13)
    plt.xlabel("Month")
    plt.ylabel("Mean Hawkishness Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# 2️⃣ MCP Pipeline Runner
def run_mcp_pipeline():
    print("🚀 MCP Research Agent — Full Speech Analysis (Summary + Tone Enabled)")
    run_boe_tool()
    run_full_eda()
    summarize_all()
    print("✅ MCP Agent completed full workflow.\n")


# 3️⃣ Helper Functions
def run_boe_tool():
    print("📰 Running BoE Speech Scraper Tool...")
    subprocess.run(["python3", "tools/scrape_boe_speeches.py"], check=True)


def run_full_eda():
    print("📊 Running EDA Tool...")
    subprocess.run(["python3", "tools/speech_eda_lda.py"], check=True)


# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_mcp_pipeline()
