"""
Official SEC Company-to-CIK Resolver Tool (Async)
Designed for AI Agent Tools (MCP/A2A-safe patterns)
Uses official SEC company_tickers.json mapping
"""

import aiohttp
import json
import difflib
import os
import sys
from typing import Optional, Tuple

# SEC headers
SEC_HEADERS = {
    "User-Agent": "KiarashAI/1.0 (kiarash996@yahoo.com)",
    "Accept": "application/json",
}

# Local file path (recommended)
COMPANY_TICKER_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "company_tickers.json")
)

# -------------------------------------------------------------------
# Helper: clean company name
# -------------------------------------------------------------------
def clean_company_name(name: str) -> str:
    return name.lower().replace(",", "").replace(".", "").strip()

# -------------------------------------------------------------------
# Load local SEC ticker mapping
# -------------------------------------------------------------------
async def load_local_tickers() -> dict:
    try:
        with open(COMPANY_TICKER_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[CIK] Local ticker file not found: {COMPANY_TICKER_FILE}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[CIK] Failed to load ticker file: {e}", file=sys.stderr)
        return {}

# -------------------------------------------------------------------
# Fallback: fetch SEC live mapping
# -------------------------------------------------------------------
async def fetch_sec_live_tickers(headers: dict = None) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.sec.gov/files/company_tickers_exchange.json",
                headers=headers or SEC_HEADERS,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                return {"data": data.get("data", [])}
    except Exception as e:
        print(f"[CIK] Live SEC fetch failed: {e}", file=sys.stderr)
        return {}

# -------------------------------------------------------------------
# Unified CIK Resolver
# -------------------------------------------------------------------
async def resolve_cik(company_name: str = None, ticker_symbol: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve company name or ticker to SEC CIK.
    Returns (cik, ticker)
    """
    if not company_name and not ticker_symbol:
        return None, None

    clean_name_query = clean_company_name(company_name or "")
    ticker_upper = (ticker_symbol or "").upper()

    # 1️⃣ Try local file first
    mapping = await load_local_tickers()
    name_to_cik = { clean_company_name(entry.get("title","")): entry.get("cik_str") 
                    for entry in mapping.values() if entry.get("cik_str") }

    ticker_to_cik = { entry.get("ticker","").upper(): entry.get("cik_str") 
                      for entry in mapping.values() if entry.get("cik_str") }

    # Check exact ticker match
    if ticker_upper in ticker_to_cik:
        return ticker_to_cik[ticker_upper], ticker_upper

    # Check exact name match
    if clean_name_query in name_to_cik:
        return name_to_cik[clean_name_query], ticker_upper or None

    # Fuzzy name match
    best_match = difflib.get_close_matches(clean_name_query, name_to_cik.keys(), n=1, cutoff=0.55)
    if best_match:
        match = best_match[0]
        return name_to_cik[match], ticker_upper or None

    # 2️⃣ Fallback: SEC live JSON fetch
    sec_data = await fetch_sec_live_tickers()
    rows = sec_data.get("data", [])

    for row in rows:
        if len(row) < 3:
            continue
        cik_raw = row[0]
        name = clean_company_name(str(row[1]))
        ticker = str(row[2]).upper()

        try:
            cik_str = str(int(cik_raw)).zfill(10)
        except:
            continue

        if ticker_upper and ticker_upper == ticker:
            return cik_str, ticker
        if clean_name_query and clean_name_query == name:
            return cik_str, ticker

    return None, None
