# FinGlobe Agent
Agentic NLP System for Central Bank Communication Analysis

FinGlobe Agent is an end-to-end, agentic NLP pipeline designed to analyze central bank communications from the Bank of England (BoE) and the Bank of Canada (BoC). The system extracts, preprocesses, scores, explains, and visualizes hawkish/dovish monetary policy stance using a unified transformer model, intelligent caching, and an interactive Streamlit dashboard.

The pipeline is optimized for reproducibility, incremental updates, and fast repeat queries. Once a query has been processed, subsequent runs return results instantly without re-scraping, re-scoring, or re-generating explanations.

------------------------------------------------------------
What the System Does
------------------------------------------------------------

Given a natural-language query such as:

"market conditions in Canada in 2024"

FinGlobe Agent will:

1. Interpret the query
   - Detect the central bank (BoE or BoC)
   - Infer the date range (year, month, or range)
   - Expand context when needed for meaningful visualization

2. Check cached artifacts
   - Monthly merged corpora
   - Model score CSV outputs
   - GPT-generated justifications

3. Execute tools only if necessary
   - Scrape missing documents only
   - Rebuild monthly corpora only when new data exists
   - Score only newly added months
   - Generate justifications only for missing months

4. Return results instantly if cached
   - No redundant scraping
   - No redundant model inference
   - No repeated OpenAI API calls

5. Visualize and explain results
   - Interactive model score time series
   - Monthly table with:
     - model score
     - short summary
     - full justification text

------------------------------------------------------------
Project Structure
------------------------------------------------------------

FinGlobe_Agent/
│
├── tools/
│   ├── root_agent.py
│   ├── query_interpreter_llm.py
│   ├── a_boe_minutes_scraper_full.py
│   ├── a_boe_speech_scraper.py
│   ├── a_boc_policy_mpr.py
│   ├── a_boc_speeches.py
│   ├── a_preparing_scraped_docs.py
│   ├── a_roberta_score_evaluate_NoWCB.py
│   └── a_openai_justification.py
│
├── data/
│   └── raw/
│       ├── minutes_*_monthly.json
│       ├── speeches_*_monthly.json
│       ├── merged_*_monthly.json
│       └── reference_*_monthly.json
│
├── output_final/
│   └── *_BANK_*_MER.csv
│
├── justifications/
│   ├── BOE/
│   │   └── YYYY-MM.json
│   └── BOC/
│       └── YYYY-MM.json
│
├── dashboard/
│   └── streamlit_app.py
│
├── .env
└── README.md

------------------------------------------------------------
Models Used
------------------------------------------------------------

Stance Classification Model:
- gtfintechlab/model_bank_of_england_stance_label
- Used for both BoE and BoC to maintain a unified scoring baseline

Justification Generation:
- OpenAI GPT-4o
- Produces long-form explanations and structured summaries
- Outputs are cached per month

------------------------------------------------------------
Outputs
------------------------------------------------------------

For each month, the system produces:

- model_score:
  Normalized hawkish–dovish score computed as:
  (hawkish − dovish) / total labels

- summary:
  A short (2–3 sentence) human-readable explanation of the month’s stance

- justification:
  A detailed narrative explaining policy signals, economic conditions, and
  evidence extracted from the source texts

- interactive plot:
  Model score over time with zoom, hover, and pan functionality

------------------------------------------------------------
Caching and Performance
------------------------------------------------------------

FinGlobe Agent is fully cache-aware.

Component behavior:
- Scraping runs only if monthly data is missing
- Preprocessing runs only after new scraping
- Scoring runs only when new months are added
- Justifications are generated once per month and reused
- Repeat queries return results instantly

------------------------------------------------------------
How to Run
------------------------------------------------------------

1. Install dependencies
   pip install -r requirements.txt

2. Set OpenAI API key
   Create a file named .env in the project root:
   OPENAI_API_KEY=your_api_key_here

3. Launch the dashboard
   streamlit run dashboard/streamlit_app.py

------------------------------------------------------------
Example Queries
------------------------------------------------------------

- canada 2024
- market conditions in england in 2023
- bank of canada inflation outlook
- boe january 2022

------------------------------------------------------------
Design Principles
------------------------------------------------------------

- Agentic orchestration with explicit tool control
- Minimal recomputation through caching
- Monthly canonical datasets for reproducibility
- Explainability-first design
- Human-readable outputs for policy analysis

------------------------------------------------------------

## 🧑‍💻 Contributors

| Name | 
|------|
| **Murat Al** 
| **Saruultug Batbayar** 
| **Praveen Kumar Anwla** 
| **Yibin Wang** 
---

## 📜 License
MIT License © 2025 — FinGlobe Team, University of Rochester

---

**📍 Repository:** [https://github.com/muratal49/FinGlobe_Agent](https://github.com/muratal49/FinGlobe_Agent)
