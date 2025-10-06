import os
import csv
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key)

# File paths
CSV_PATH = "data/boe_filtered_speeches.csv"
SUMMARY_DIR = "summaries/speech_summaries"

system_prompt = (
    "You are a macroeconomics assistant analyzing central bank speeches. "
    "Summarize the speech with focus on hawkish or dovish signals, references to inflation, economic growth, "
    "and monetary policy guidance. Keep the tone neutral and professional."
)

def summarize_speech(speech: dict) -> str:
    prompt = (
        f"Speech Title: {speech['title']}\n"
        f"Speaker: {speech['speaker']}\n"
        f"Date: {speech['date']}\n"
        f"Speech Content:\n{speech['text']}"
    )

    if len(prompt) > 12000:
        prompt = prompt[:12000] + "\n\n[TRUNCATED]"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Failed to summarize {speech['title'][:50]}...: {e}")
        return ""

def write_summary(speech, summary_text):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", speech['title'])[:60]
    filename = f"{speech['date']}_{safe_title}.md"
    filepath = os.path.join(SUMMARY_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {speech['title']}\n")
        f.write(f"**Speaker:** {speech['speaker']}\n\n")
        f.write(f"**Date:** {speech['date']}\n\n")
        f.write(summary_text)
    print(f"✅ Saved summary to {filepath}")

def run():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        speeches = list(csv.DictReader(f))

    speeches = sorted(speeches, key=lambda r: r["date"], reverse=True)[:2]  # Only top 2 for now
    print(f"📄 Loaded {len(speeches)} speeches")

    for speech in speeches:
        print(f"\n🧠 Summarizing: {speech['title'][:60]}...")
        summary = summarize_speech(speech)
        if summary:
            write_summary(speech, summary)

if __name__ == "__main__":
    import re
    run()
