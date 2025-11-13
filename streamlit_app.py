# streamlit_app.py
import json
from pathlib import Path
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Path helpers ----------
HERE = Path(__file__).resolve().parent

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "raw").exists():
            return p
    return start

REPO = find_repo_root(HERE)
DATA_RAW = REPO / "data" / "raw"
CACHE_MINUTES = DATA_RAW / "cache_minutes"

SCORES_CSV = DATA_RAW / "merged_boe_scores.csv"
JUSTIFY_CSV = DATA_RAW / "justifications_openai.csv"  # optional

# ---------- Streamlit setup ----------
st.set_page_config(page_title="FinGlobe — BoE Monthly Scoring", page_icon="🏦", layout="wide")
st.title("FinGlobe: BoE Monthly Hawkish/Dovish Scoring")
st.caption(f"Repo: {REPO}")

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_scores(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "month" not in df.columns:
        raise ValueError("Expected a 'month' column (YYYY-MM) in merged_boe_scores.csv")
    df["month"] = df["month"].astype(str).str[:7]
    df["date"]  = pd.to_datetime(df["month"] + "-01", errors="coerce")

    cols = [
        "reference_score",
        "minutes_score",
        "speech_score",
        "merged_score",
        "weighted_minutes",
        "weighted_speech",
        "weighted_merged",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[["month", "date", *cols]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_justifications(justify_csv: Path) -> pd.DataFrame:
    if not justify_csv.exists():
        return pd.DataFrame(columns=["date", "score", "justification"])
    df = pd.read_csv(justify_csv)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "score", "justification"])
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01", errors="coerce")
    return df.dropna(subset=["date"])

@st.cache_data(show_spinner=False)
def load_latest_cache_minutes(cache_dir: Path) -> pd.DataFrame:
    files = sorted(cache_dir.glob("minutes_boe_*.json"), reverse=True)
    if not files:
        return pd.DataFrame(columns=["date", "text"])
    latest = files[0]
    with latest.open("r", encoding="utf-8") as f:
        data = json.load(f)  # {"YYYY-MM" or "YYYY-MM-DD": "text"}
    rows = []
    for k, v in data.items():
        k_m = str(k)[:7]
        dt = pd.to_datetime(k_m + "-01", errors="coerce")
        if pd.notna(dt):
            rows.append({"month": k_m, "date": dt, "text": v})
    df = pd.DataFrame(rows).sort_values("date")
    # add empty score columns so downstream chart/table code works
    for c in [
        "reference_score","minutes_score","speech_score",
        "merged_score","weighted_minutes","weighted_speech","weighted_merged"
    ]:
        df[c] = np.nan
    return df

# ---------- Load data (prefer scored CSV; fallback to cache) ----------
scores = None
errors = []

if SCORES_CSV.exists():
    try:
        scores = load_scores(SCORES_CSV)
    except Exception as e:
        errors.append(f"Error reading merged scores CSV: {e}")

if scores is None or scores.empty:
    if CACHE_MINUTES.exists():
        cache_df = load_latest_cache_minutes(CACHE_MINUTES)
        if cache_df.empty:
            errors.append(f"No cached minutes JSON files found in {CACHE_MINUTES}")
        else:
            st.warning(
                f"⚠️ Using latest cache from {CACHE_MINUTES.name} because "
                f"`{SCORES_CSV.name}` is missing. Run your pipeline to generate scores."
            )
            scores = cache_df.copy()
    else:
        errors.append(
            f"Missing both `{SCORES_CSV.name}` and cache dir `{CACHE_MINUTES}`. "
            f"Run your pipeline (root_agent) first."
        )

if scores is None or scores.empty:
    st.error("\n\n".join(errors))
    st.stop()

just = load_justifications(JUSTIFY_CSV)

# ---------- Two separate date inputs in the sidebar ----------
min_date = scores["date"].min().date()
max_date = scores["date"].max().date()

st.sidebar.header("Date Range")
start_month = st.sidebar.date_input(
    "Start month",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    help="Pick or type the start month."
)
end_month = st.sidebar.date_input(
    "End month",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    help="Pick or type the end month."
)

# Auto-correct if user picks end < start (no hard error, just adjust)
if end_month < start_month:
    st.sidebar.info("End month was before Start month; using Start month for both.")
    end_month = start_month

mask = (scores["date"].dt.date >= start_month) & (scores["date"].dt.date <= end_month)
df = scores.loc[mask].copy()
if df.empty:
    st.info("No data in the selected range.")
    st.stop()

# ---------- Helpers ----------
def mse_overlap(series: pd.Series, reference: pd.Series) -> float | None:
    j = pd.concat([series, reference], axis=1, join="inner").dropna()
    return float(((j.iloc[:, 0] - j.iloc[:, 1]) ** 2).mean()) if not j.empty else None

# ---------- Plot prep ----------
plot_df = df.melt(
    id_vars=["date"],
    value_vars=[
        "reference_score",
        "minutes_score", "weighted_minutes",
        "speech_score",  "weighted_speech",
        "merged_score",  "weighted_merged",
    ],
    var_name="series",
    value_name="score",
)

name_map = {
    "reference_score":  "Reference",
    "minutes_score":    "Minutes",
    "weighted_minutes": "Minutes (weighted)",
    "speech_score":     "Speeches",
    "weighted_speech":  "Speeches (weighted)",
    "merged_score":     "Merged",
    "weighted_merged":  "Merged (weighted)",
}
plot_df["Series"] = plot_df["series"].map(name_map)

pair_root = {
    "Reference":            "Reference",
    "Minutes":              "Minutes",
    "Minutes (weighted)":   "Minutes",
    "Speeches":             "Speeches",
    "Speeches (weighted)":  "Speeches",
    "Merged":               "Merged",
    "Merged (weighted)":    "Merged",
}
plot_df["Pair"] = plot_df["Series"].map(pair_root)

# ---------- Chart (Altair) ----------
st.subheader("Monthly Scores vs Reference")

base = alt.Chart(plot_df).transform_filter(alt.datum.score != None)

color = alt.Color("Pair:N", legend=alt.Legend(title="Series (pairs share color)"))

# One predicate to flag all weighted series for dashed line
weighted_oneof = alt.FieldOneOfPredicate(field="Series", oneOf=[
    "Minutes (weighted)", "Speeches (weighted)", "Merged (weighted)"
])

line_nonref = base.transform_filter(alt.datum.Series != "Reference").mark_line(strokeWidth=2).encode(
    x=alt.X("date:T", title="Month"),
    y=alt.Y("score:Q", title="Score"),
    color=color,
    detail="Series:N",
    strokeDash=alt.condition(
        weighted_oneof,
        alt.value([6, 3]),
        alt.value([0, 0])
    ),
    tooltip=[
        "Series:N",
        alt.Tooltip("date:T", title="Month"),
        alt.Tooltip("score:Q", title="Score", format=".4f")
    ]
)

ref_line = base.transform_filter(alt.datum.Series == "Reference").mark_line(strokeWidth=2).encode(
    x="date:T", y="score:Q", color=color,
    tooltip=["Series:N", "date:T", alt.Tooltip("score:Q", format=".4f")]
)

st.altair_chart((line_nonref + ref_line).interactive(), use_container_width=True)

# ---------- MSE table (overlap with reference only) ----------
ref_s = df.set_index("date")["reference_score"]
series_list = [
    ("Minutes",             df.set_index("date")["minutes_score"]),
    ("Minutes (weighted)",  df.set_index("date")["weighted_minutes"]),
    ("Speeches",            df.set_index("date")["speech_score"]),
    ("Speeches (weighted)", df.set_index("date")["weighted_speech"]),
    ("Merged",              df.set_index("date")["merged_score"]),
    ("Merged (weighted)",   df.set_index("date")["weighted_merged"]),
]
mse_rows = []
for name, s in series_list:
    m = mse_overlap(s, ref_s)
    mse_rows.append({"Series": name, "MSE (overlap only)": np.nan if m is None else m})
mse_df = pd.DataFrame(mse_rows)

st.subheader("MSE vs Reference (overlap months only)")
st.dataframe(mse_df, use_container_width=True)

# ---------- Scores & justifications ----------
tab_out = df[
    ["date", "reference_score", "minutes_score", "speech_score", "merged_score",
     "weighted_minutes", "weighted_speech", "weighted_merged"]
].sort_values("date")

just = just.drop_duplicates(subset=["date"], keep="last")
if not just.empty:
    tab_out = tab_out.merge(just[["date", "score", "justification"]], on="date", how="left")

st.subheader("Scores Table")
st.dataframe(tab_out, use_container_width=True)

if not just.empty:
    st.subheader("Monthly Justifications")
    for _, r in tab_out.iterrows():
        jtxt = r.get("justification")
        if isinstance(jtxt, str) and jtxt.strip():
            with st.expander(f"{r['date'].strftime('%Y-%m')} — justification"):
                st.write(jtxt)

# ---------- Download ----------
st.download_button(
    "Download filtered scores (CSV)",
    data=tab_out.to_csv(index=False).encode("utf-8"),
    file_name=f"scores_{start_month}_to_{end_month}.csv",
    mime="text/csv",
)

st.download_button(
    "Download filtered scores (JSON)",
    data=tab_out.to_json(orient="records", date_format="iso").encode("utf-8"),
    file_name=f"scores_{start_month}_to_{end_month}.json",
    mime="application/json",
)
