

# client = OpenAI()
# in src/mcp_project/tools/stance_scorer.py
import os, json, time, re
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

MODEL = "gpt-4o-mini"
SYSTEM = ("You are a macro policy analyst. Score MPC summaries on a dovish↔hawkish scale. Return strict JSON only.")

def getClient():
    load_dotenv(find_dotenv())
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)
client = getClient()


def buildPrompt(summary: str) -> str:
    return (
        "Rate stance.\n"
        "Output JSON:\n"
        '{ "score": float in [0,1], "stance":"dovish|neutral|hawkish", "rationale":"1 sentence" }\n\n'
        f"Summary:\n'''{summary.strip()}'''\n"
        "Return ONLY the JSON."
    )

def parseJson(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else json.loads(text)

def scoreWithOpenai(summary: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": buildPrompt(summary)},
        ],
    )
    txt = r.choices[0].message.content
    out = parseJson(txt)
    s = float(out.get("score", 0.5))
    s = 0.0 if s < 0 else 1.0 if s > 1 else s
    out["score"] = s
    out["stance"] = out.get("stance") or ("hawkish" if s > 0.6 else "dovish" if s < 0.4 else "neutral")
    return out

def scoreAll(inputPath="mpc_minutes.json", outPath="mpc_scores.json", pause=0.5):
    with open(inputPath, "r", encoding="utf-8") as f:
        data = json.load(f)  # {date: summary}
    out = {}
    for date, summary in sorted(data.items()):
        out[date] = scoreWithOpenai(summary)
        time.sleep(pause)

    # ensure cache dir exists
    os.makedirs("cache/output_data", exist_ok=True)  
    stamp = time.strftime("%Y%m%d_%H%M%S")           

    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # write cached snapshot
    cached_path = f"cache/output_data/mpc_scores_{stamp}.json"  
    with open(cached_path, "w", encoding="utf-8") as f:         
        json.dump(out, f, ensure_ascii=False, indent=2)    

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY not set")
    scoreAll()
