# 🧠 FinGlobe Agent  
### Central Bank Sentiment Intelligence (BoE & BoC)

The system uses a multi-agent architecture inspired by the MCP (Multi-Context Protocol) framework:

## 📘 Overview
**FinGlobe Agent** is an agentic research and analysis framework designed to automatically **collect, summarize, and analyze** central bank communications — such as **speeches, minutes, and policy statements** from the **Bank of England (BoE)** and **Bank of Canada (BoC)** — to quantify **hawkish–dovish sentiment** ahead of monetary policy decisions.

The system follows a **multi-agent architecture** inspired by the *Multi-Context Protocol (MCP)* design:
1. **Research Agent** → Scrapes and filters recent content (BoE/BoC speeches, reports).  
2. **Summarization Agent** → Generates structured summaries using LLMs.  
3. **Analysis Agent** → Evaluates sentiment and computes hawkish–dovish scores.  

---

## 🏗️ Repository Structure
```
FinGlobe_Agent/
│
├── mcp.py                         # Main pipeline: scrape → summarize → analyze → score
│
├── tools/
│   ├── scrape_boe_speeches.py     # Scrapes recent BoE speeches (date/topic filters)
│   ├── fetch_minutes.py           # (planned) Fetches MPC meeting minutes
│   ├── fetch_content.py           # (planned) Generic text fetcher
│   └── filter_articles.py         # (planned) Filters and cleans scraped data
│
├── agents/
│   ├── root_agent.py              # Orchestrates multi-agent workflow
│   ├── research_agent.py          # Handles scraping and data retrieval
│   └── analysis_agent.py          # Evaluates hawkish–dovish sentiment
│
├── data/
│   ├── raw/                       # Raw scraped speeches/articles
│   ├── processed/                 # Cleaned summaries
│   └── analysis/                  # Final sentiment scores and metrics
│
├── logs/                          # Runtime and debug logs
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/muratal49/FinGlobe_Agent.git
cd FinGlobe_Agent
```

### 2️⃣ Create and Activate Environment
```bash
conda create -n finglobe python=3.10
conda activate finglobe
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If you’re using OpenAI or other APIs, create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

---

## 🚀 Usage Workflow

### 🕸️ Step 1 — Scrape Speeches
```bash
python tools/scrape_boe_speeches.py
```
- Collects recent BoE speeches  
- Filters by **date** (last 8 months) and **topic** (Monetary Policy / Inflation)  
- Saves to:
  ```
  data/raw/boe_speeches_<YYYYMMDD>.csv
  ```

### ✍️ Step 2 — Summarize Content
```bash
python mcp.py --mode summarize
```
- Runs the MCP summarization process  
- Outputs short summaries to:
  ```
  data/processed/boe_summaries_<YYYYMMDD>.csv
  ```

### 📊 Step 3 — Analyze & Score Sentiment
```bash
python mcp.py --mode analyze
```
- Computes **hawkish/dovish sentiment scores** for each speech  
- Saves results to:
  ```
  data/analysis/hawk_dove_scores_<YYYYMMDD>.csv
  ```

### 🔄 Step 4 — Full Automated Pipeline
```bash
python mcp.py
```
Runs all modules sequentially:  
> **Scrape → Summarize → Analyze → Score**

---

## 📈 Output Example

| Date       | Speaker          | Title                               | Hawkish_Dovish_Score | Summary |
|-------------|------------------|-------------------------------------|----------------------|----------|
| 2025-09-12  | Sarah Breeden    | Bumps in the Road                   | -0.32 *(Dovish)*     | Supply-side pressures are easing... |
| 2025-09-27  | Megan Greene     | The Supply Side Demands Attention   | +0.41 *(Hawkish)*    | Tight labor market justifies...     |

---

## 🤝 Collaboration Notes
- **Environment:** All collaborators should use the same `finglobe` conda environment.  
- **Data Handling:**  
  - Large raw CSVs stay in `/data/raw/` and are **not** committed to GitHub.  
  - Use `.gitignore` to exclude large files.  
- **Branching Workflow:**  
  - Use feature branches (e.g., `feature/analysis`, `feature/scraper`)  
  - Use clear, descriptive commit messages.  
- **Logging:**  
  - Runtime and debug info saved to `/logs/` — check before re-running long scrapes.

---

## 🔧 Development Status
| Component | Status | Description |
|------------|---------|-------------|
| `scrape_boe_speeches.py` | ✅ Working | Scrapes and saves BoE speeches |
| `mcp.py` | ✅ Working | Summarizes and scores hawkish/dovish sentiment |
| `fetch_minutes.py` | 🔜 Planned | Parse MPC meeting minutes |
| `analysis_agent.py` | 🔜 In Progress | Improve scoring model calibration |
| `fetch_boc_speeches.py` | 🔜 Planned | Add Bank of Canada data pipeline |

---

## 🧩 Next Steps
- [ ] Add Bank of Canada (BoC) scraping module  
- [ ] Expand sentiment lexicon for hawkish/dovish detection  
- [ ] Add monthly visualization of sentiment trends  
- [ ] Automate periodic scraping (e.g., cron or GitHub Actions)

---

## 🔗 Useful Links
- [Bank of England – Speeches](https://www.bankofengland.co.uk/news/speeches)  
- [Bank of Canada – Deliberations]( (https://www.bankofcanada.ca/publications/summary-governing-council-deliberations/)
- [Bank of Canada – Monetray Policy Reports] (https://www.bankofcanada.ca/publications/mpr/)
- 
- [GitHub Repository](https://github.com/muratal49/FinGlobe_Agent)

---

## ⚡ Quick Start for Collaborators
If you just want to test everything end-to-end:
```bash
# 1. Clone & install
git clone https://github.com/muratal49/FinGlobe_Agent.git
cd FinGlobe_Agent
pip install -r requirements.txt

# 2. Run the pipeline
python mcp.py
```

You’ll get:
```
data/
 ├─ raw/          # scraped speeches
 ├─ processed/    # summaries
 └─ analysis/     # hawkish/dovish scores
```

---

**Maintainers:**  
👤 Murat A. • 👤 Praveen Kumar A. • 👤 Saruultug B. • 👤 Yibin W.  
*(Team FinGlobe — University of Rochester, MS Data Science Capstone 2025)*  
