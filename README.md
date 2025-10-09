🧠 FinGlobe Agent
Central Bank Sentiment Intelligence (BoE & BoC)
📘 Overview

FinGlobe Agent is an agentic research and analysis framework designed to automatically collect, summarize, and analyze central-bank communications (e.g., Bank of England and Bank of Canada) to measure hawkish–dovish sentiment ahead of monetary-policy decisions.

The system uses a multi-agent architecture inspired by the MCP (Multi-Context Protocol) framework:

Research Agent → Scrapes and filters recent content (speeches, minutes, statements).

Summarization Agent → Generates structured summaries of the scraped text.

Analysis Agent → Quantifies sentiment using hawkish/dovish scoring models.

🏗️ Repository Structure
FinGlobe_Agent/
│
├── mcp.py                         # Main pipeline: scrape → summarize → analyze → score
│
├── tools/
│   ├── scrape_boe_speeches.py     # Scrapes recent BoE speeches (filters by date/topic)
│   ├── fetch_minutes.py           # (planned) Fetches MPC meeting minutes
│   ├── fetch_content.py           # (planned) Generalized HTML/text fetcher
│   └── filter_articles.py         # (planned) Cleans and filters relevant content
│
├── agents/
│   ├── root_agent.py              # Orchestrates multi-agent tasks
│   ├── research_agent.py          # Handles retrieval and scraping logic
│   └── analysis_agent.py          # Evaluates hawkish–dovish sentiment
│
├── data/
│   ├── raw/                       # Raw scraped data
│   ├── processed/                 # Cleaned text and summaries
│   └── analysis/                  # Sentiment results and scores
│
├── logs/                          # Runtime and progress logs
│
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-org-or-username>/FinGlobe_Agent.git
cd FinGlobe_Agent

2. Create Environment
conda create -n finglobe python=3.10
conda activate finglobe

3. Install Dependencies
pip install -r requirements.txt


(If you use OpenAI API for summarization or analysis, add a .env file with your key.)

OPENAI_API_KEY=your_api_key_here

🚀 Usage Workflow
Step 1 — Scrape Speeches
python tools/scrape_boe_speeches.py


Collects speeches from the Bank of England’s site

Filters by recent date and topic (e.g., Monetary Policy Committee, Inflation)

Saves output to:

data/raw/boe_speeches_<YYYYMMDD>.csv

Step 2 — Summarize Content
python mcp.py --mode summarize


Runs the MCP summarization pipeline

Produces clean text summaries in data/processed/

Step 3 — Analyze & Score
python mcp.py --mode analyze


Computes hawkish/dovish scores for each speech

Saves results to:

data/analysis/hawk_dove_scores_<YYYYMMDD>.csv

Step 4 — Full Pipeline (One-Click Run)
python mcp.py


Runs all steps:

Scrape → Summarize → Analyze → Score

📊 Output Example
Date	Speaker	Title	Hawkish_Dovish_Score	Summary
2025-09-12	Sarah Breeden	“Bumps in the Road”	-0.32 (Dovish)	Supply-side pressures are easing...
2025-09-27	Megan Greene	“The Supply Side Demands Attention”	+0.41 (Hawkish)	Tight labor market justifies...
👥 Collaboration Notes

Environment consistency: All collaborators should use the same conda environment (finglobe).

Data handling:

Large raw data files should not be committed — they are stored in /data/raw/ and excluded via .gitignore.

Processed summaries and analysis results can be shared if small.

Branching:

Use feature branches: feature/analysis, feature/scraper, etc.

Commit descriptive messages.

Logging:

All runtime logs are saved in /logs/; please check before rerunning a failed scrape.

🧩 Next Steps

 Add BoC speech scraping support (scrape_boc_speeches.py)

 Expand sentiment lexicon for hawkish/dovish detection

 Visualize policy tone trends across months

 Deploy the MCP pipeline as a periodic cron job

🔗 Useful Links

Bank of England Speeches

Bank of Canada Speeches

GitHub Repository
