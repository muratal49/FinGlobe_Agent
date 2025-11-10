#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3A — RoBERTa Scoring, Confidence-Weighted Scoring, Merge, MSE, and Plots

What this script does (as requested):
1) Sentence-level labeling → score per text using a hawk–dove function.
2) Computes three monthly series:
     - minutes_score      (from minutes text only)
     - speech_score       (from speech text only)
     - merged_score       (from CONCAT(minutes + speeches) BUT ONLY for months that have minutes)
3) Also computes confidence-weighted variants:
     - weighted_minutes, weighted_speech, weighted_merged
4) Full outer join of all months with columns:
     month, reference_score, minutes_score, speech_score, merged_score,
     weighted_minutes, weighted_speech, weighted_merged
5) Plots:
   A) All 6 model series vs reference, with an inset showing MSE for each of the six (overlap months only).
      (Pairs share the same color; weighted is dashed.)
   B) Only the merged pair (unweighted + weighted) vs reference — filename prefixed with "merged_only_".

Inputs (created by Step 2):
  data/raw/minutes_boe_monthly.json   -> {"YYYY-MM": "minutes blob text"}
  data/raw/speeches_boe_monthly.json  -> {"YYYY-MM": "speeches blob text (concat)"}
  data/raw/reference_boe_monthly.json -> {"YYYY-MM": float}  (optional)

Output CSV:
  data/raw/merged_boe_scores.csv  (full outer join with required columns)

Plots:
  data/plots/boe_all_series_vs_reference.png
  data/plots/merged_only_boe_series_vs_reference.png

Notes:
- We don't require --end-date; scoring uses whatever months are present in the Step 2 outputs.
- MSE is computed ONLY on months present in BOTH each series and the reference.
"""

from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------- Paths & Model ----------------------
BASE_PATH = Path("/Users/murat/Desktop/Capstone/FinGlobe_Agent").resolve()
DATA_RAW = BASE_PATH / "data" / "raw"
PLOTS_DIR = BASE_PATH / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MINUTES_JSON = DATA_RAW / "minutes_boe_monthly.json"
SPEECHES_JSON = DATA_RAW / "speeches_boe_monthly.json"
REFERENCE_JSON = DATA_RAW / "reference_boe_monthly.json"

OUT_CSV = DATA_RAW / "merged_boe_scores.csv"

MODEL_NAME = "gtfintechlab/model_bank_of_england_stance_label"


# ---------------------- Sentence utilities ----------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # simple, fast splitter

def split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Split on punctuation boundaries; clean tiny fragments
    sents = _SENT_SPLIT_RE.split(text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 5]
    return sents


# ---------------------- Label → bucket mapping ----------------------
@dataclass
class LabelMapping:
    hawkish_idx: List[int]
    dovish_idx: List[int]
    neutral_idx: List[int]
    irrelevant_idx: List[int]

def derive_label_mapping(id2label: Dict[int, str]) -> LabelMapping:
    """
    Try to infer hawkish/dovish/neutral/irrelevant indices from model labels.
    Fallback: assume 3-class [dovish, neutral, hawkish] if we can't infer.
    """
    lower = {i: lab.lower() for i, lab in id2label.items()}
    hawk, dove, neut, irr = [], [], [], []
    for i, lab in lower.items():
        if "hawk" in lab or "tighten" in lab or "hike" in lab:
            hawk.append(i)
        elif "dove" in lab or "ease" in lab or "cut" in lab or "loosen" in lab:
            dove.append(i)
        elif "neutral" in lab:
            neut.append(i)
        elif "irrelevant" in lab or "other" in lab:
            irr.append(i)
    if not (hawk or dove or neut):
        # fallback 3-class: [dovish, neutral, hawkish] in that order
        n = len(id2label)
        if n >= 3:
            dove, neut, hawk = [0], [1], [2]
        elif n == 2:
            dove, hawk = [0], [1]
            neut = []
        else:
            # degenerate: treat last as hawkish
            hawk = [n - 1]
    return LabelMapping(hawkish_idx=hawk, dovish_idx=dove, neutral_idx=neut, irrelevant_idx=irr)


# ---------------------- Scoring functions ----------------------
@dataclass
class TextScore:
    score: float
    weighted_score: float
    n_effective: int

def score_sentences(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    mapping: LabelMapping,
    device: torch.device,
    max_len: int = 256,
) -> TextScore:
    """
    Sentence-level labeling → two scores:
      - score:        (H - D) / max(N_eff, 1)
      - weighted_score: confidence-weighted version using max softmax prob
    Irrelevant sentences are excluded from N_eff and the numerator.
    """
    if not sentences:
        return TextScore(score=0.0, weighted_score=0.0, n_effective=0)

    # Aggregate tallies
    H = D = N = 0
    wH = wD = wN = 0.0  # weighted
    n_eff = 0

    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()[0]
        # probs
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])  # confidence weight

        # Determine bucket
        if pred_idx in mapping.irrelevant_idx:
            continue  # exclude from denominator
        n_eff += 1

        if pred_idx in mapping.hawkish_idx:
            H += 1
            wH += conf
        elif pred_idx in mapping.dovish_idx:
            D += 1
            wD += conf
        else:
            N += 1
            wN += conf

    denom = max(n_eff, 1)
    denom_w = max(wH + wD + wN, 1e-9)  # only effective (non-irrelevant)

    raw = (H - D) / denom
    weighted = (wH - wD) / denom_w
    return TextScore(score=float(raw), weighted_score=float(weighted), n_effective=n_eff)


def score_text_blob(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    mapping: LabelMapping,
    device: torch.device,
) -> TextScore:
    sents = split_into_sentences(text)
    return score_sentences(sents, tokenizer, model, mapping, device)


# ---------------------- Load monthly maps ----------------------
def load_json_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object map at {path}")
    return obj


# ---------------------- MSE helper ----------------------
def mse_overlap(series: pd.Series, reference: pd.Series) -> float | None:
    """Compute MSE on overlapping (non-null) months; return None if no overlap."""
    df = pd.concat([series, reference], axis=1, join="inner").dropna()
    if df.shape[0] == 0:
        return None
    return float(((df.iloc[:, 0] - df.iloc[:, 1]) ** 2).mean())


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, required=False, help="Optional lower bound: YYYY-MM or YYYY-MM-DD")
    # --end-date intentionally optional/ignored for scoring; inputs already filtered upstream
    args = ap.parse_args()

    # Load monthly maps
    if not MINUTES_JSON.exists():
        raise FileNotFoundError(f"Missing minutes monthly JSON: {MINUTES_JSON}")
    if not SPEECHES_JSON.exists():
        raise FileNotFoundError(f"Missing speeches monthly JSON: {SPEECHES_JSON}")

    minutes_map = load_json_map(MINUTES_JSON)      # {"YYYY-MM": text}
    speeches_map = load_json_map(SPEECHES_JSON)    # {"YYYY-MM": text}
    reference_map = {}
    if REFERENCE_JSON.exists():
        reference_map = load_json_map(REFERENCE_JSON)  # {"YYYY-MM": float}

    # Optional: clamp to start-date (by month)
    if args.start_date:
        try:
            mkey = args.start_date[:7]  # YYYY-MM
            minutes_map = {k: v for k, v in minutes_map.items() if k >= mkey}
            speeches_map = {k: v for k, v in speeches_map.items() if k >= mkey}
            if reference_map:
                reference_map = {k: v for k, v in reference_map.items() if k >= mkey}
        except Exception:
            pass

    # Model & tokenizer
    print("🧩 Loading Hugging Face model:", MODEL_NAME)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device set to use {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    mapping = derive_label_mapping(model.config.id2label)

    # Score minutes and speeches per month
    months_all = sorted(set(minutes_map.keys()) | set(speeches_map.keys()) | set(reference_map.keys()))
    rows = []  # will hold per-month raw + weighted + merged

    for m in months_all:
        minutes_txt = minutes_map.get(m, "")
        speech_txt  = speeches_map.get(m, "")
        ref_val     = reference_map.get(m, None)

        # Minutes
        min_score = min_w = None
        if minutes_txt.strip():
            ts = score_text_blob(minutes_txt, tokenizer, model, mapping, device)
            min_score = ts.score
            min_w     = ts.weighted_score

        # Speech
        sp_score = sp_w = None
        if speech_txt.strip():
            ts = score_text_blob(speech_txt, tokenizer, model, mapping, device)
            sp_score = ts.score
            sp_w     = ts.weighted_score

        # Merged (ONLY if minutes exist for this month)
        mg_score = mg_w = None
        if minutes_txt.strip():
            merged_text = (minutes_txt + "\n\n" + speech_txt).strip() if speech_txt else minutes_txt
            ts = score_text_blob(merged_text, tokenizer, model, mapping, device)
            mg_score = ts.score
            mg_w     = ts.weighted_score
        # else: merged stays None (discard speeches-only months)

        rows.append({
            "month": m,
            "reference_score": ref_val,
            "minutes_score": min_score,
            "speech_score": sp_score,
            "merged_score": mg_score,
            "weighted_minutes": min_w,
            "weighted_speech": sp_w,
            "weighted_merged": mg_w,
        })

    df = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    # Save full outer-join CSV with requested columns
    keep_cols = [
        "month",
        "reference_score",
        "minutes_score",
        "speech_score",
        "merged_score",
        "weighted_minutes",
        "weighted_speech",
        "weighted_merged",
    ]
    df[keep_cols].to_csv(OUT_CSV, index=False)
    print(f"💾 Saved merged scores -> {OUT_CSV}")

    # ---------------------- PLOTS ----------------------
    import matplotlib.pyplot as plt

    # Prepare Series
    s_ref  = df.set_index("month")["reference_score"].astype(float, errors="ignore")
    s_min  = df.set_index("month")["minutes_score"].astype(float, errors="ignore")
    s_sp   = df.set_index("month")["speech_score"].astype(float, errors="ignore")
    s_mg   = df.set_index("month")["merged_score"].astype(float, errors="ignore")
    s_minw = df.set_index("month")["weighted_minutes"].astype(float, errors="ignore")
    s_spw  = df.set_index("month")["weighted_speech"].astype(float, errors="ignore")
    s_mgw  = df.set_index("month")["weighted_merged"].astype(float, errors="ignore")

    # MSEs on overlap only
    mse_min  = mse_overlap(s_min,  s_ref)
    mse_minw = mse_overlap(s_minw, s_ref)
    mse_sp   = mse_overlap(s_sp,   s_ref)
    mse_spw  = mse_overlap(s_spw,  s_ref)
    mse_mg   = mse_overlap(s_mg,   s_ref)
    mse_mgw  = mse_overlap(s_mgw,  s_ref)

    # ---- Plot A: all series vs reference, MSE inset ----
    fig, ax = plt.subplots(figsize=(12, 6))
    # palette coupling: use same base color for pair (solid = unweighted, dashed = weighted)
    # let matplotlib choose default colors c0,c1,c2
    # minutes
    l1, = ax.plot(s_min.index, s_min.values, label="minutes_score", linewidth=2)
    ax.plot(s_minw.index, s_minw.values, linestyle="--", linewidth=2, color=l1.get_color(), label="weighted_minutes")
    # speeches
    l2, = ax.plot(s_sp.index, s_sp.values, label="speech_score", linewidth=2)
    ax.plot(s_spw.index, s_spw.values, linestyle="--", linewidth=2, color=l2.get_color(), label="weighted_speech")
    # merged
    l3, = ax.plot(s_mg.index, s_mg.values, label="merged_score", linewidth=2)
    ax.plot(s_mgw.index, s_mgw.values, linestyle="--", linewidth=2, color=l3.get_color(), label="weighted_merged")
    # reference on top
    ax.plot(s_ref.index, s_ref.values, label="reference_score", linewidth=2)

    ax.set_title("BoE Monthly Scores vs Reference")
    ax.set_xlabel("Month")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25)

    # Inset with MSEs (overlap only)
    mse_lines = []
    def _fmt(m): return "—" if m is None or math.isnan(m) else f"{m:.6f}"
    mse_lines.append(f"minutes:        { _fmt(mse_min) }")
    mse_lines.append(f"weighted_min:   { _fmt(mse_minw) }")
    mse_lines.append(f"speeches:       { _fmt(mse_sp) }")
    mse_lines.append(f"weighted_sp:    { _fmt(mse_spw) }")
    mse_lines.append(f"merged:         { _fmt(mse_mg) }")
    mse_lines.append(f"weighted_mg:    { _fmt(mse_mgw) }")

    # Create a small text box inside the plot
    textstr = "MSE (overlap only)\n" + "\n".join(mse_lines)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.995, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    out_png_all = PLOTS_DIR / "boe_all_series_vs_reference.png"
    fig.tight_layout()
    fig.savefig(out_png_all, dpi=150)
    plt.close(fig)
    print(f"🖼  Saved plot -> {out_png_all}")

    # ---- Plot B: merged-only pair vs reference ----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    # reuse same color idea for merged pair
    l_mg, = ax2.plot(s_mg.index, s_mg.values, linewidth=2, label="merged_score")
    ax2.plot(s_mgw.index, s_mgw.values, linestyle="--", linewidth=2, color=l_mg.get_color(), label="weighted_merged")
    ax2.plot(s_ref.index, s_ref.values, linewidth=2, label="reference_score")

    ax2.set_title("Merged Text Monthly Scores vs Reference")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)

    # Inset with MSEs for merged pair only
    mse_lines2 = []
    mse_lines2.append(f"merged:       { _fmt(mse_mg) }")
    mse_lines2.append(f"weighted_mg:  { _fmt(mse_mgw) }")
    textstr2 = "MSE (overlap only)\n" + "\n".join(mse_lines2)
    ax2.text(0.995, 0.02, textstr2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    out_png_merged = PLOTS_DIR / "merged_only_boe_series_vs_reference.png"
    fig2.tight_layout()
    fig2.savefig(out_png_merged, dpi=150)
    plt.close(fig2)
    print(f"🖼  Saved plot -> {out_png_merged}")

    print("✅ 3A scoring/evaluation complete.")


if __name__ == "__main__":
    main()
