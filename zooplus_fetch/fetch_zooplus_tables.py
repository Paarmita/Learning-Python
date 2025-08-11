#!/usr/bin/env python3
"""
Fetch Zooplus product-comparison JSON for article IDs from a spreadsheet.

Usage examples:
  python fetch_zooplus_tables.py --input article_ids.csv
  python fetch_zooplus_tables.py --input article_ids.xlsx --sheet Sheet1 --max 50

Outputs created in the current folder:
  - zooplus_results.json  (array of JSON responses)
  - zooplus_results.csv   (flattened table if possible)
  - zooplus_fetch_log.csv (status per article_id)
"""

import argparse
import json
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from pandas import json_normalize
from tqdm import tqdm


BASE_URL = "https://www.zooplus.de/product-comparison/api/v1/sites/2/en-DE/table"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZooplusFetcher/1.0; +https://example.com)",
    "Accept": "application/json",
}

def read_article_ids(path: str, sheet: Optional[str]) -> List[str]:
    path_lower = path.lower()

    # Read file with safe defaults (BOM-safe for CSV)
    if path_lower.endswith(".csv"):
        # If your CSV uses semicolons (common in DE locales), add sep=";"
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
    else:
        raise ValueError("Input must be a .csv or .xlsx file")

    # Normalize header names: remove BOM/zero-width spaces/NBSP, lowercase, strip
    def norm(s):
        return (
            str(s)
            .replace("\ufeff", "")  # BOM
            .replace("\u200b", "")  # zero-width space
            .replace("\xa0", " ")   # non-breaking space
            .strip()
            .lower()
        )

    normalized = [norm(c) for c in df.columns]
    df.columns = normalized

    # Accept a few reasonable synonyms, pick the first match
    candidates = {"article_id", "article id", "articleid", "id"}
    colname = next((c for c in df.columns if c in candidates), None)

    if not colname:
        raise ValueError(f"Couldn't find 'article_id'. Saw columns: {list(df.columns)}")

    ids = (
        df[colname]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Deduplicate while preserving order
    seen, uniq = set(), []
    for x in ids:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq



def fetch_one(article_id: str, timeout: float = 20.0, retries: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
    """
    Fetch one article JSON with simple retry/backoff.
    Returns dict with keys: article_id, ok (bool), status_code, data (dict or None), error (str or None)
    """
    params = {"article_id": article_id}
    attempt = 0
    last_err = None
    while attempt < retries:
        try:
            resp = requests.get(BASE_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                try:
                    return {
                        "article_id": article_id,
                        "ok": True,
                        "status_code": 200,
                        "data": resp.json(),
                        "error": None,
                    }
                except Exception as je:
                    return {
                        "article_id": article_id,
                        "ok": False,
                        "status_code": 200,
                        "data": None,
                        "error": f"JSON parse error: {je}",
                    }
            else:
                # Retry for server/ratelimit errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff ** attempt)
                    attempt += 1
                    continue
                # Non-retryable client error
                return {
                    "article_id": article_id,
                    "ok": False,
                    "status_code": resp.status_code,
                    "data": None,
                    "error": f"HTTP {resp.status_code}",
                }
        except requests.RequestException as re:
            last_err = str(re)
            time.sleep(backoff ** attempt)
            attempt += 1

    return {
        "article_id": article_id,
        "ok": False,
        "status_code": None,
        "data": None,
        "error": last_err or "Unknown error",
    }


def save_outputs(results: List[Dict[str, Any]]) -> None:
    # Save a simple log
    log_rows = [
        {"article_id": r["article_id"], "ok": r["ok"], "status_code": r["status_code"], "error": r["error"]}
        for r in results
    ]
    pd.DataFrame(log_rows).to_csv("zooplus_fetch_log.csv", index=False)

    # Keep only successful data
    data_rows = [dict(article_id=r["article_id"], **(r["data"] if isinstance(r["data"], dict) else {"_raw": r["data"]}))
                 for r in results if r["ok"] and r.get("data") is not None]

    # JSON (array of objects)
    with open("zooplus_results.json", "w", encoding="utf-8") as f:
        json.dump(data_rows, f, ensure_ascii=False, indent=2)

    # CSV (flatten nested JSON best-effort)
    if data_rows:
        try:
            df = json_normalize(data_rows, sep=".")
            df.to_csv("zooplus_results.csv", index=False)
        except Exception as e:
            # Fallback: save a CSV with two columns if flattening fails
            fallback = pd.DataFrame(
                [{"article_id": r["article_id"], "json": json.dumps(r.get("data"), ensure_ascii=False)} for r in results if r["ok"]]
            )
            fallback.to_csv("zooplus_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Fetch Zooplus product comparison JSON by article_id.")
    parser.add_argument("--input", required=True, help="Path to CSV or XLSX with a column named 'article_id'.")
    parser.add_argument("--sheet", default=None, help="Sheet name (if using Excel).")
    parser.add_argument("--max", type=int, default=50, help="Max number of article_ids to process.")
    parser.add_argument("--rate", type=float, default=0.4, help="Seconds to wait between requests (politeness).")
    args = parser.parse_args()

    article_ids = read_article_ids(args.input, args.sheet)
    if not article_ids:
        print("No article IDs found. Make sure your file has a column 'article_id'.")
        return

    article_ids = article_ids[: args.max]

    results: List[Dict[str, Any]] = []
    print(f"Fetching {len(article_ids)} articles...")
    for aid in tqdm(article_ids, desc="Downloading", unit="req"):
        res = fetch_one(aid)
        results.append(res)
        time.sleep(args.rate)

    save_outputs(results)
    print("Done! Files created:\n  - zooplus_results.json\n  - zooplus_results.csv\n  - zooplus_fetch_log.csv")


if __name__ == "__main__":
    main()
