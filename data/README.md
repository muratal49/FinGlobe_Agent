# 🏦 FinGlobe_Agent — Bank of England MacroX Capstone

### *Automated multi-agent pipeline for central-bank sentiment analysis*  
**University of Rochester – MacroX FinGlobe Capstone (Fall 2025)**  
Lead: **Murat Al** | Collaborators: Saruul, Praveen, Yibin  

---

## 📘 Project Overview

**FinGlobe_Agent** is a multi-agent AI pipeline that automatically scrapes, processes, scores, and interprets the **Bank of England’s monetary policy communications** — including **MPC Minutes, Speeches, and Reports** — to analyze *hawkish vs. dovish* sentiment.

The project is part of the **MacroX FinGlobe Capstone**, integrating LLM-based justifications and traditional NLP stance models to provide data-driven central-bank sentiment insights.

---

## 🚀 End-to-End Pipeline

| Step | Script | Output | Description |
|------|---------|---------|-------------|
| **1A** | `tools/meeting_scraper.py` | `data/raw/minutes_boe.json` | Scrapes MPC meeting minutes (Bank of England site). |
| **1B** | `tools/scrape_boe_speeches.py` | `data/raw/boe_filtered_speeches_conclusion.csv` | Collects and filters speeches related to monetary policy and inflation. |
| **2** | `tools/preparing_scraped_docs.py` | Monthly JSONs (`minutes_boe_monthly.json`, `speeches_boe_monthly.json`, `reference_boe_monthly.json`) | Cleans and aggregates all raw text by month. |
| **3A** | `tools/roberta_merged_score_evaluate.py` | `data/raw/merged_boe_scores.csv`, plots | Applies a fine-tuned RoBERTa stance model, computes MSE vs. reference scores, and generates monthly evaluation plots. |
| **3B** | `tools/openai_merge_justify.py` | `data/raw/justifications_openai.csv` | Uses GPT-4o to produce ~300-word natural-language justifications for each monthly sentiment score. |
| **ROOT** | `tools/root_agent.py` | Full automation | Runs all steps sequentially with user-provided date range. |

---

## 🧠 Example Usage

```bash
python3 tools/root_agent.py --start-date 2024-08-01 --end-date 2025-01-01
```

---

## 📂 Output Summary

| File | Description |
|------|--------------|
| `data/raw/minutes_boe.json` | Raw MPC minutes text |
| `data/raw/boe_filtered_speeches_conclusion.csv` | Filtered speeches with conclusion sections |
| `data/raw/minutes_boe_monthly.json` | Aggregated monthly minutes |
| `data/raw/speeches_boe_monthly.json` | Aggregated monthly speeches |
| `data/raw/reference_boe_monthly.json` | (Optional) Ground-truth reference scores |
| `data/raw/merged_boe_scores.csv` | Merged stance model results & weighted scores |
| `data/raw/justifications_openai.csv` | GPT-generated justifications (≈300 words each) |
| `data/plots/` | Auto-generated MSE and comparison charts |

---

## 🧰 Environment Setup

```bash
conda create -n finagent python=3.10
conda activate finagent
pip install -r requirements.txt
```

### `.env` configuration
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

---

## 🧩 Project Architecture

```
FinGlobe_Agent/
│
├── tools/
│   ├── meeting_scraper.py
│   ├── scrape_boe_speeches.py
│   ├── preparing_scraped_docs.py
│   ├── roberta_merged_score_evaluate.py
│   ├── openai_merge_justify.py
│   └── root_agent.py
│
├── data/
│   ├── raw/
│   │   ├── minutes_boe.json
│   │   ├── boe_filtered_speeches_conclusion.csv
│   │   ├── merged_boe_scores.csv
│   │   └── justifications_openai.csv
│   └── plots/
│
├── requirements.txt
├── .env
└── README.md
```

---

## 🧑‍💻 Contributors

| Name | Role | Key Contributions |
|------|------|------------------|
| **Murat Al** | Project Lead | Root Agent design, Model integration, Prompt Engineering |
| **Saruul** | Analyst | EDA, Visualization, Topic Summaries |
| **Praveen** | Researcher | Minutes analysis, Ground truth scoring |
| **Yibin** | Engineer | Dashboard integration, GUI pipeline controls |

---

## 📜 License
MIT License © 2025 — FinGlobe Team, University of Rochester

---

**📍 Repository:** [https://github.com/muratal49/FinGlobe_Agent](https://github.com/muratal49/FinGlobe_Agent)
