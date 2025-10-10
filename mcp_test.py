"""
MCP TEST PIPELINE (v3.1, compatible with OpenAI ≥ 1.0)
Fast agentic prototype combining summarization + hawkish/dovish scoring.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from subprocess import run
from openai import OpenAI
from dotenv import load_dotenv

# ✅ load .env file automatically
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
def load_latest_speech_data():
    candidates = sorted(Path("data/raw").glob("boe*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError("❌ No speech CSV found under data/raw/")
    df = pd.read_csv(candidates[0])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"📄 Loaded {candidates[0].name} with {len(df)} speeches")
    return df

# ---------- Optional EDA ----------
def run_eda_lda():
    print("📊 Running EDA + LDA Topic Analysis ...")
    run(["python3", "tools/speech_eda_lda.py"], check=True)
    print("✅ EDA+LDA analysis completed.\n")

# ---------- Sampling ----------
def sample_one_per_month(df):
    df["month"] = df["date"].dt.to_period("M")
    sample = df.sort_values("date").groupby("month").first().reset_index(drop=True)
    print(f"🧾 Picked {len(sample)} representative speeches (one per month)")
    return sample

# ---------- GPT Summarization + Scoring ----------
def summarize_and_score(text, date, speaker, title):
    prompt = f"""
You are a senior monetary-policy analyst.

Read the following Bank of England speech and:
1. Summarize it in a single analytical paragraph (~250 words) highlighting inflation,
   growth, and monetary-policy tone.
2. Assign a hawkish/dovish tone score between -1 (very dovish) and +1 (very hawkish).
3. Give a concise one-sentence justification.

Return JSON with:
{{
  "month": "{date.strftime('%B, %Y')}",
  "speaker": "{speaker}",
  "summary_250w": "...",
  "hawkish_dovish_score": value,
  "justification": "..."
}}

Speech Title: {title}
Speech Text:
{text[:4000]}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500,
    )
    return completion.choices[0].message.content.strip()

# ---------- Main ----------
def main(run_eda=False):
    df = load_latest_speech_data()
    if run_eda:
        run_eda_lda()

    sample = sample_one_per_month(df)
    results = []
    for _, row in sample.iterrows():
        print(f"🤖 Processing {row['date'].strftime('%B, %Y')} – {row.get('speaker','Unknown')}")
        output = summarize_and_score(
            text=row["text"],
            date=row["date"],
            speaker=row.get("speaker", "Unknown"),
            title=row.get("title", "")
        )
        print(output[:300], "...\n")
        results.append(output)

    out_dir = Path("data/analysis_results")
    ensure_dir(out_dir)
    outfile = out_dir / "monthly_summary_score_fast.jsonl"
    with open(outfile, "w") as f:
        for r in results:
            f.write(r + "\n")

    print(f"✅ Saved combined summaries + scores → {outfile}")

if __name__ == "__main__":
    main(run_eda=False)
