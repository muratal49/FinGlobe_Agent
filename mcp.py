# mcp.py

import os
from click import prompt
import pandas as pd
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# 🌱 Load OpenAI API key from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key)


# if not OPENAI_API_KEY:
#     raise ValueError("Missing OpenAI API key in .env file.")



# 🛠 Tool Path
BOE_TOOL_PATH = "tools/scrape_boe_speeches.py"
CSV_PATH = "data/boe_filtered_speeches.csv"
SUMMARY_PATH = "outputs/summary_boe.csv"

# 1️⃣ Run BoE Speech Scraper
def run_boe_tool():
    print("🔧 Running BoE scraper tool...")
    subprocess.run(
        ["python", BOE_TOOL_PATH],
        check=True
    )
    print("✅ Scraper finished.")

# 2️⃣ Load Speeches and Format for LLM
def summarize_with_gpt():
    print("📄 Reading scraped data...")
    df = pd.read_csv(CSV_PATH)
    summaries = []

    for _, row in df.iterrows():
        prompt = f"""You are an economics analyst.
Summarize the following Bank of England speech focusing on:
- Monetary policy stance
- Hawkish vs. dovish tone
- Key inflation, rate, or committee signals

SPEECH TITLE: {row['title']}
SPEAKER: {row['speaker']}
DATE: {row['date']}

--- START OF SPEECH ---
{row['text'][:3000]}
--- END ---
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        summaries.append(summary)

    df["summary"] = summaries
    df.to_csv(SUMMARY_PATH, index=False)
    print(f"💾 Saved summarized output -> {SUMMARY_PATH}")

# 🔁 Main MCP-style flow
def run_mcp_research_agent():
    print("🧠 Research Agent starting...")
    run_boe_tool()
    summarize_with_gpt()
    print("✅ Agent done.")

if __name__ == "__main__":
    run_mcp_research_agent()
