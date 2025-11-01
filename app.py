import os
from dotenv import load_dotenv
load_dotenv(".env")
import json, time, datetime as dt, logging, os
from mcp.server.fastmcp import FastMCP
from .tools import stance_scorer as scorer
from mcp_project.utils.utils import logger
from mcp_project.utils.keyword_signal import keywordScorer
from statistics import mean
    
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # finds ../.env when you run inside src/
    
mcp = FastMCP("FinGlobeMCP")


# app.py  (only new/changed bits)

@mcp.tool()
def scrapeMpcMinutes(start: str = "", end: str = "", headless: bool = False) -> dict:
    """
    Scrape BoE MPC summaries within a date range (inclusive).
    Dates accepted: 'YYYY-MM-DD' or 'DD/MM/YYYY'. Empty = default (Jan 1 this year → today).
    Writes mpc_minutes.json and a cached snapshot. Returns file info.
    """
    t0 = time.perf_counter()
    logger.info("scrapeMpcMinutes: start | start=%s end=%s headless=%s", start, end, headless)
    from .tools import meeting_scraper
    s = start or None
    e = end or None
    meeting_scraper.run(start=s, end=e, headless=headless, write_files=True)

    with open("mpc_minutes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    path = os.path.abspath("mpc_minutes.json")
    cached_dir = os.path.abspath("cache/scrapped_data")
    logger.info("scrapeMpcMinutes: done | items=%d | file=%s | %.2fs",
                len(data), path, time.perf_counter() - t0)
    return {"path": path, "items": len(data), "cache_dir": cached_dir}




@mcp.tool()
def scoreMpcSummaries(model: str = "gpt-4o-mini", pause: float = 0.25, runs: int = 10) -> dict:
    """Run scoring N times, write one consolidated file, and compute per-month average scores."""
    t0 = time.perf_counter()
    logger.info("scoreMpcSummaries: start | model=%s pause=%.2f runs=%d", model, pause, runs)


    # load summaries (date -> text)
    with open("mpc_minutes.json","r",encoding="utf-8") as _f:
        minutes = json.load(_f)
    summaries = {}
    if isinstance(minutes, list):
        for it in minutes:
            d = it.get("date") or it.get("Date") or it.get("meeting_date")
            s = it.get("summary") or it.get("Summary") or it.get("text") or ""
            if d: summaries[str(d)] = str(s)
    elif isinstance(minutes, dict):
        for d, it in minutes.items():
            if isinstance(it, dict):
                summaries[str(d)] = str(it.get("summary",""))
            else:
                summaries[str(d)] = str(it)


    from glob import glob
    scorer.MODEL = model
    all_runs = []

    # snapshot existing timestamped files once
    existing_archives = set(glob("cache/output_data/mpc_scores_*.json"))

    for i in range(runs):
        scorer.scoreAll(inputPath="mpc_minutes.json", outPath="mpc_scores.json", pause=pause)
        with open("mpc_scores.json", "r", encoding="utf-8") as f:
            res = json.load(f)

        # enrich each date in this run with kw_* fields
        for d, obj in list(res.items()):
            if not isinstance(obj, dict): 
                continue
            txt = summaries.get(d, "")
            kw_s, kw_st, kw_rat = keywordScorer(txt)
            obj["kw_score"] = kw_s
            obj["kw_stance"] = kw_st
            obj["kw_rationale"] = kw_rat

        all_runs.append(res)

        # cleanup per-run archives
        now_archives = set(glob("cache/output_data/mpc_scores_*.json"))
        new_archives = sorted(now_archives - existing_archives)
        for p in new_archives:
            try:
                os.remove(p)
            except Exception as e:
                logger.warning("cleanup skip: %s (%s)", p, e)

        if pause > 0:
            time.sleep(pause)

    # per-month averages
    def avgByMonth(runs_list, field="score"):
        buckets = {}
        for run in runs_list:
            for d, obj in run.items():
                if not isinstance(obj, dict) or field not in obj:
                    continue
                month = d[:7]
                try:
                    val = float(obj[field])
                except Exception:
                    continue
                buckets.setdefault(month, []).append(val)
        return {m: (sum(v)/len(v) if v else float("nan")) for m, v in sorted(buckets.items())}

    monthly_avgs = avgByMonth(all_runs, field="score")        # LLM output
    monthly_kw_avgs = avgByMonth(all_runs, field="kw_score")  # keyword signal

    # collect rationales and stance counts per month (from all runs; LLM rationale)
    monthly_rationales, monthly_stance_counts = {}, {}
    for run in all_runs:
        for d, obj in run.items():
            if not isinstance(obj, dict):
                continue
            m = d[:7]
            r = obj.get("rationale")
            if r:
                monthly_rationales.setdefault(m, []).append(r)
            s = obj.get("stance")
            if s:
                monthly_stance_counts.setdefault(m, {}).setdefault(s, 0)
                monthly_stance_counts[m][s] += 1

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath("cache/model_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ts}.json")

    consolidated = {
        ts: {
            "model": model,
            "runs": all_runs,                        # 10 raw runs (now with kw_* fields)
            "monthly_avg_scores": monthly_avgs,
            "monthly_avg_kw_scores": monthly_kw_avgs,
            "monthly_rationales": monthly_rationales,
            "monthly_stance_counts": monthly_stance_counts
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)

    logger.info("scoreMpcSummaries: done | runs=%d | months=%d | file=%s | %.2fs",
                runs, len(monthly_avgs), out_path, time.perf_counter() - t0)

    return {
        "path": out_path,
        "items": runs,
        "cache_dir": out_dir,
        "monthly_avg_scores": monthly_avgs,
        "monthly_avg_kw_scores": monthly_kw_avgs,
        "monthly_rationales": monthly_rationales,
        "monthly_stance_counts": monthly_stance_counts
    }

@mcp.tool()
def readScores() -> dict:
    """Return latest consolidated scores if present, else fallback to mpc_scores.json."""
    latest_dir = "cache/model_output"
    if os.path.isdir(latest_dir):
        files = sorted([p for p in os.listdir(latest_dir) if p.endswith(".json")])
        if files:
            latest = os.path.join(latest_dir, files[-1])
            logger.info("readScores: consolidated %s", latest)
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)

    path = os.path.abspath("mpc_scores.json")
    logger.info("readScores: fallback %s", path)
    if not os.path.exists("mpc_scores.json"):
        return {"error": "mpc_scores.json not found"}
    with open("mpc_scores.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    logger.info("App entry: args=%s", sys.argv[1:])
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "scrape":
            # usage:
            # python -m mcp_project.app scrape                # defaults
            # python -m mcp_project.app scrape 2024-01-01 2024-12-31
            s = sys.argv[2] if len(sys.argv) > 2 else ""
            e = sys.argv[3] if len(sys.argv) > 3 else ""
            print(scrapeMpcMinutes(start=s, end=e))

        elif cmd == "read":
            print(readScores())
        # app.py (pipeline branch only — replace the whole 'pipeline' elif)

        elif cmd == "pipeline":
            import glob, shutil

            # Optional date args:
            # python -m mcp_project.app pipeline
            # python -m mcp_project.app pipeline 2024-01-01 2024-12-31
            s = sys.argv[2] if len(sys.argv) > 2 else ""
            e = sys.argv[3] if len(sys.argv) > 3 else ""

            # If a range is provided, scrape that range and rely on the range cache file
            if s or e:
                print(f"🔵 Scraping for range {s or '[default-start]'} → {e or '[today]'} ...")
                scrapeMpcMinutes(start=s, end=e)

            # Prefer range-named cache first (if dates provided)
            range_glob = []
            if s or e:
                from mcp_project.utils.utils import iso_for_filename
                s_tag = iso_for_filename(s or dt.date(dt.date.today().year, 1, 1))
                e_tag = iso_for_filename(e or dt.date.today())
                range_glob = glob.glob(f"cache/scrapped_data/mpc_minutes_{s_tag}_{e_tag}.json")

            cached_consolidated = sorted(glob.glob("cache/model_output/*.json"))

            if cached_consolidated:
                latest = cached_consolidated[-1]
                print(f"🟢 Using consolidated output: {latest}")
                with open(latest, "r", encoding="utf-8") as f:
                    print(json.dumps(json.load(f), indent=2))
            else:
                # choose scrape source in priority order:
                # 1) range cache (if asked for a range), 2) latest timestamped cache, 3) fresh scrape
                source_path = None
                if range_glob:
                    source_path = range_glob[0]
                    print(f"🟡 Using range cached scraped data: {source_path}")
                else:
                    cached_scrapes = sorted(glob.glob("cache/scrapped_data/mpc_minutes_*.json"))
                    if cached_scrapes:
                        source_path = cached_scrapes[-1]
                        print(f"🟡 Using cached scraped data: {source_path}")

                if source_path:
                    with open(source_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    with open("mpc_minutes.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    print("🔵 No cached data found — scraping fresh...")
                    scrapeMpcMinutes(start=s, end=e)

                print("⚙️  Running scoring step (10 runs)...")
                out = scoreMpcSummaries(model="gpt-4o-mini", runs=10)
                print(json.dumps(out, indent=2))
        else:
            print("Usage: python -m mcp_project.app [scrape|score [model]|read]")
    else:
        logger.info("MCP mode: waiting for host on stdio… (logs also in fin_globe_mcp.log)")
        mcp.run()

