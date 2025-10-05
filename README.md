# FinGlobe Agent
Multi-agent system for hawkish/dovish sentiment analysis using Bank of England (BoE) and Bank of Canada (BoC) data.

A modular, agent-based system that scrapes and analyzes central bank communications to detect **hawkish/dovish sentiment** from the **Bank of England (BoE)** and **Bank of Canada (BoC)**.

---

## 🔍 Project Goal

Quantify monetary policy tone over time using:
- 📝 Meeting Minutes
- 📣 Speeches
- 📊 Bank Rate Announcements

For both:
- 🇬🇧 Bank of England (BoE)
- 🇨🇦 Bank of Canada (BoC)

---

## 🤖 Agent Architecture

| Agent           | Role |
|----------------|------|
| `root_agent`    | Orchestrates all steps |
| `research_agent`| Scrapes news, minutes, speeches |
| `analysis_agent`| Scores sentiment and trends |

---

## 🧰 Tools

| Tool                  | Function |
|-----------------------|----------|
| `fetch_boe_listing.py`| Scrape article metadata from BoE |
| `fetch_boe_content.py`| Get full text from BoE articles |
| `fetch_boc_listing.py`| Scrape article metadata from BoC |
| `fetch_boc_content.py`| Get full text from BoC articles |
| `sentiment_model.py`  | Score documents for hawkish/dovish tone |

---

## 📁 Project Structure


