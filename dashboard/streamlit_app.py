import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
import sys

# Make tools importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "tools"))

from root_agent import run_root_agent


st.set_page_config(
    page_title="FinGlobe MacroX Dashboard",
    layout="wide"
)

st.title("📊 FinGlobe MacroX – Hawkish/Dovish Stance Intelligence Dashboard")

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
st.sidebar.header("Query Controls")

query = st.sidebar.text_input(
    "Enter a natural-language query:",
    placeholder="e.g., 'market in canada in 2025' or 'england 2023'"
)

run_btn = st.sidebar.button("Run Analysis")

# ------------------------------------------
# MAIN CONTENT
# ------------------------------------------
if run_btn:
    with st.spinner("Processing your request..."):
        result = run_root_agent(query)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    if "warning" in result:
        st.warning(result["warning"])

    st.success(f"Completed for **{result['bank'].upper()}** — {result['range']}")

    # --------------------------------------
    # INTERACTIVE PLOTLY PLOT
    # --------------------------------------
    st.subheader("📈 Model Score Trend (Interactive, model_score only)")

    df_plot = pd.DataFrame(result["df_plot"])

    if df_plot.empty:
        st.warning("No plot data available.")
    else:
        fig = px.line(
            df_plot,
            x="month",
            y="model_score",
            markers=True,
            title=f"Model Score Trend – {result['bank'].upper()}",
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Model Score",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8),
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------
    # TABLE: month, model_score, summary, justification
    # --------------------------------------
    st.subheader("📊 Monthly Summary + Justification")

    df_table = pd.DataFrame(result["table"])
    if df_table.empty:
        st.info("No monthly records available for this query.")
    else:
        st.dataframe(df_table, use_container_width=True)

    # --------------------------------------
    # STRUCTURED JSON OUTPUTS
    # --------------------------------------
    st.subheader("📚 Detailed Structured Outputs (per month)")

    for ym, js in result["justifications"].items():
        with st.expander(f"{ym}"):
            st.json(js)

    # --------------------------------------
    # DOWNLOAD
    # --------------------------------------
    st.download_button(
        "⬇️ Download Full JSON Output",
        data=json.dumps(result, indent=2),
        file_name="finglobe_output.json",
        mime="application/json",
    )
