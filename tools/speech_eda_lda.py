"""
Speech EDA + LDA Topic Analysis Tool (v5)
-----------------------------------------
Performs EDA, LDA topic modeling, normalized keyword trends, and identifies
distinctive monthly keywords.  Saves all visuals under /data/analysis_results/.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re, string
import seaborn as sns

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ---------- Helpers ----------

def first_day_of_month_n_months_ago(n: int) -> datetime:
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    total_months = (y * 12 + (m - 1)) - n
    y2, m2 = divmod(total_months, 12)
    m2 += 1
    return datetime(y2, m2, 1, tzinfo=timezone.utc)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, title: str):
    """Save figure to /data/analysis_results/."""
    out_dir = Path("data/analysis_results")
    ensure_dir(out_dir)
    safe_title = title.replace(" ", "_").replace("/", "_").lower()
    file_path = out_dir / f"eda_speech_{safe_title}.png"
    fig.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"💾 Saved figure: {file_path}")


# ---------- Load Data ----------

def load_latest_speech_data():
    candidates = sorted(Path("data/raw").glob("boe*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError("❌ No speech CSV found under data/raw/")
    df = pd.read_csv(candidates[0])
    print(f"📄 Loaded {candidates[0].name} with {len(df)} speeches")
    return df


# ---------- EDA ----------

def basic_eda(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly_counts = df["month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_counts.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Number of Speeches per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_xticklabels([m.strftime("%b, %Y") for m in monthly_counts.index], rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "Number of Speeches per Month")
    return df


# ---------- Text Preprocessing ----------

def preprocess_texts(df: pd.DataFrame):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    finance_vocab = {
        "inflation","interest","rate","rates","monetary","policy","growth","market",
        "financial","credit","investment","output","labor","employment","wages",
        "productivity","liquidity","economy","trade","exchange","fiscal","balance",
        "spending","borrowing","banking","currency","money","debt","lending","capital",
        "asset","price","housing","demand","supply","consumption","risk","volatility",
        "forecast","expectation","tightening","easing","inflationary","deflation"
    }

    def clean(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", " ", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
        tokens = [t for t in tokens if t in finance_vocab or len(t) > 5]
        return tokens

    df["tokens"] = df["text"].astype(str).apply(clean)
    df = df[df["tokens"].str.len() > 5]
    print(f"🧹 Retained {len(df)} speeches after token filtering")
    return df


# ---------- LDA ----------

def lda_topic_model(df: pd.DataFrame, num_topics=10):
    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=True
    )

    print("\n💬 LDA Top Topics:")
    for idx, topic in lda_model.show_topics(num_topics=num_topics, num_words=8, formatted=False):
        words = [w for w, _ in topic]
        print(f"Topic {idx+1}: {', '.join(words)}")

    def get_dominant_topic(bow):
        topics = lda_model.get_document_topics(bow)
        return max(topics, key=lambda x: x[1])[0] if topics else None

    df["dominant_topic"] = [get_dominant_topic(bow) for bow in corpus]
    df["topic_keywords"] = df["dominant_topic"].apply(
        lambda t: ", ".join([w for w, _ in lda_model.show_topic(t, topn=6)]) if t is not None else ""
    )

    return df, lda_model


# ---------- Monthly Trends and Distinctive Words ----------

def summarize_monthly_topics(df: pd.DataFrame):
    df["month"] = df["date"].dt.to_period("M")
    cutoff = first_day_of_month_n_months_ago(6).replace(tzinfo=None)
    df_recent = df[df["date"] >= cutoff]

    # Expand topic keywords
    expanded = (
        df_recent.groupby("month")["topic_keywords"]
        .apply(lambda x: ", ".join(x))
        .apply(lambda text: pd.Series(text.split(", ")))
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("keyword")
    )

    keyword_month = expanded.groupby(["month", "keyword"]).size().unstack(fill_value=0)
    speech_counts = df_recent.groupby("month").size()
    normalized = keyword_month.div(speech_counts, axis=0).fillna(0)

    # --- Ignore globally dominant boilerplate words ---
    COMMON_GLOBAL_TERMS = {
        "monetary", "policy", "financial", "market",
        "economy", "economic", "system", "bank", "committee"
    }
    normalized = normalized[[c for c in normalized.columns if c.lower() not in COMMON_GLOBAL_TERMS]]

    # --- Top 7 keywords globally (after filtering) ---
    top7 = normalized.sum(axis=0).sort_values(ascending=False).head(7).index
    normalized_top = normalized[top7]

    # Trend Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    normalized_top.plot(ax=ax, marker="o")
    ax.set_title("Monthly Usage Trends of Top 7 Keywords (Filtered & Normalized)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized Frequency")
    ax.set_xticklabels([m.strftime("%b, %Y") for m in normalized_top.index], rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "Monthly Usage Trends of Top 7 Keywords (Filtered & Normalized)")

    # --- Identify distinctive monthly keywords ---
    overall_avg = normalized.mean()
    distinctive_words = {}
    for month in normalized.index:
        diff = (normalized.loc[month] - overall_avg).sort_values(ascending=False)
        distinct = [w for w in diff.index if w not in top7][:2]  # skip global top words
        distinctive_words[month] = distinct

    # --- Create summary table ---
    summary_df = pd.DataFrame([
        {
            "Month": month.strftime("%B, %Y"),
            "Distinctive Keywords": ", ".join(words)
        }
        for month, words in distinctive_words.items()
    ])

    print("\n🧭 Distinctive Monthly Keywords (excluding top common terms):")
    print(summary_df.to_string(index=False))

    # --- Heatmap of normalized frequencies ---
    import seaborn as sns
    diff_matrix = normalized_top.copy()
    diff_matrix.index = [m.strftime("%b, %Y") for m in diff_matrix.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(diff_matrix.T, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Normalized Freq'})
    ax.set_title("Heatmap: Normalized Frequency of Top 7 Filtered Keywords by Month")
    plt.tight_layout()
    save_figure(fig, "Heatmap Normalized Frequency of Top 7 Filtered Keywords by Month")

    # Save summary CSV
    out_table = Path("data/analysis_results/monthly_distinctive_keywords_filtered.csv")
    ensure_dir(out_table.parent)
    summary_df.to_csv(out_table, index=False)
    print(f"\n💾 Saved distinctive monthly keywords summary to {out_table}")

# ---------- Main ----------

def main():
    df = load_latest_speech_data()
    df = basic_eda(df)
    df = preprocess_texts(df)
    df, _ = lda_topic_model(df)
    summarize_monthly_topics(df)

    out_path = Path("data/analysis_results/boe_speeches_topics.csv")
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved detailed topic analysis to {out_path}")


if __name__ == "__main__":
    main()
