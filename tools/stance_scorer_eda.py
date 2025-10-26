# src/mcp_project/tools/stance_scorer.py
import os, json, time, re, math
from typing import Dict, List
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, RateLimitError

MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM = "You are a macro policy analyst. Score MPC summaries on a dovish↔hawkish scale. Return strict JSON only."

# Thresholds for a score in [-1, 1]
LOW_T, HIGH_T = -0.20, 0.20         # neutral band [-0.2, +0.2]
MAX_RETRY = 3

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
        '{ "score": float in [-1,1], "stance":"dovish|neutral|hawkish", "rationale":"2 sentences" }\n\n'
        f"Summary:\n'''{summary.strip()}'''\n"
        "Return ONLY the JSON."
    )

def parseJson(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else json.loads(text)

def _clip_pm1(x: float) -> float:
    # clip to [-1, 1]
    return -1.0 if x < -1 else 1.0 if x > 1 else x

def _stance_from_score(s: float) -> str:
    if s > HIGH_T: return "hawkish"
    if s < LOW_T:  return "dovish"
    return "neutral"

def _one_call(summary: str) -> Dict:
    """Single deterministic call (temperature=0) returning a validated dict on [-1,1]."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
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
            s = float(out.get("score", 0.0))
            s = _clip_pm1(s)
            out["score"] = s
            out["stance"] = out.get("stance") or _stance_from_score(s)
            out["rationale"] = out.get("rationale", "").strip()
            # Optional: provide a 0..1 legacy score if downstream needs it
            out["score01"] = round((s + 1) / 2, 6)
            return out
        except RateLimitError:
            time.sleep(1.5 * attempt)
        except Exception:
            if attempt == MAX_RETRY:
                return {"score": 0.0, "score01": 0.5, "stance":"neutral", "rationale":"fallback after errors"}
            time.sleep(0.5 * attempt)

def scoreWithOpenai(summary: str) -> dict:
    """Backward-compatible single-shot wrapper (now returns -1..1 in 'score')."""
    return _one_call(summary)

def score_ensemble(summary: str, n_runs: int = 5, pause: float = 0.0) -> Dict:
    """Run n repeats; return mean/std over [-1,1], plus a legacy 0..1 mean."""
    scores: List[float] = []
    rationales: List[str] = []
    for _ in range(n_runs):
        out = _one_call(summary)
        scores.append(out["score"])
        if out.get("rationale"): rationales.append(out["rationale"])
        if pause: time.sleep(pause)
    mean = sum(scores) / len(scores)
    var  = sum((x - mean) ** 2 for x in scores) / max(len(scores) - 1, 1)
    std  = math.sqrt(var)
    stance = _stance_from_score(mean)
    rationale = rationales[0] if rationales else ""
    return {
        "mean": round(mean, 6),          # on [-1, 1]
        "std": round(std, 6),
        "mean01": round((mean + 1) / 2, 6),  # legacy 0..1 if you need for old plots
        "scores": scores,
        "stance": stance,
        "rationale": rationale
    }

def score_all(inputPath="mpc_minutes.json", outPath="mpc_scores.json", n_runs: int = 5, pause_each: float = 0.2):
    """Scores all date->summary pairs; writes {date: {mean,std,mean01,scores,stance,rationale}}"""
    with open(inputPath, "r", encoding="utf-8") as f:
        data = json.load(f)  # {date: summary}

    out: Dict[str, Dict] = {}
    for date, summary in sorted(data.items()):
        out[date] = score_ensemble(summary, n_runs=n_runs, pause=0.0)
        if pause_each: time.sleep(pause_each)

    os.makedirs(os.path.dirname(outPath) or ".", exist_ok=True)
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("cache/output_data", exist_ok=True)
    cached_path = f"cache/output_data/mpc_scores_{stamp}.json"
    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

# Backward compatibility: your old entrypoint
def scoreAll(inputPath="mpc_minutes.json", outPath="mpc_scores.json", pause=0.5):
    return score_all(inputPath=inputPath, outPath=outPath, n_runs=1, pause_each=pause)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY not set")
    score_all()
