import json
import re
from openai import OpenAI
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()   # loads .env from the current working directory

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You extract structured query parameters from user text.

Return STRICT JSON with keys:
- bank: "england" | "canada" | "no results available for that bank"
- start_date: "YYYY-MM-DD"
- end_date: "YYYY-MM-DD"

If the bank is not England or Canada, set:
bank = "no results available for that bank".
"""

def extract_json_block(text):
    """Extract final JSON block from a model message."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON found in LLM output.")
    return json.loads(m.group(0))

def interpret_query_llm(query):
    """LLM-based query interpreter that returns a structured dict."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query}
        ],
        temperature=0.0,
    )

    # FIX: Use .message.content, NOT .parsed
    raw = response.choices[0].message.content

    data = extract_json_block(raw)

    # enforce required fields
    return {
        "bank": data.get("bank", "no results available for that bank"),
        "start_date": data.get("start_date", "2000-01-01"),
        "end_date": data.get("end_date", "2025-12-31"),
    }
