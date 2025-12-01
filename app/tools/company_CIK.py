"""
Official SEC Company-to-CIK Resolver Tool (Async)
Designed for AI Agent Tools (MCP/A2A-safe patterns)
Uses official SEC company_tickers.json mapping
"""

import aiohttp
import json
import backoff
import sys
import difflib
from typing import Optional
import os

# -------------------------------------------------------------------
# SEC-compliant headers
# -------------------------------------------------------------------
SEC_HEADERS = {
    "User-Agent": "KiarashAI/1.0 (kiarash996@yahoo.com)",
    "Accept": "application/json",
}

# Path to the official mapping
# You can download it from:
# https://www.sec.gov/files/company_tickers.json
COMPANY_TICKER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "company_tickers.json"))



# -------------------------------------------------------------------
# Rate limit helper for cases where you fetch the mapping remotely
# (even if local, this keeps tool future-proof)
# -------------------------------------------------------------------

def is_rate_limit_error(exc):
    """
    Treat 429, 503, 504 as retry-able SEC load issues
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 503, 504)
    return False


# -------------------------------------------------------------------
# Load the SEC company ticker map (local file recommended)
# Format: { "0": { "ticker": "...", "title": "...", "cik_str": "0000320193" }, ... }
# -------------------------------------------------------------------
async def load_ticker_file() -> dict:
    try:
        with open(COMPANY_TICKER_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"Ticker file not found: {COMPANY_TICKER_FILE}"}
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------
# Main CIK Resolver Tool
# -------------------------------------------------------------------
async def resolve_cik(company_name: str) -> dict:
    """
    Resolve company name (even approximate) to best CIK.

    Args:
        company_name: Name or approximate name ("Aple Inc", "Apple", "Tesla Motors", etc.)

    Returns:
        dict: { "company": "...", "cik": "...", "match_type": "exact|fuzzy" }
    """

    # Normalize input for LLM-safe patterns
    clean_name = company_name.strip().lower()

    # Load mapping
    mapping = await load_ticker_file()
    if "error" in mapping:
        return mapping  # Early return

    # Build a searchable list
    name_to_cik = {}
    names = []

    for entry in mapping.values():
        title = entry.get("title", "").lower()
        cik = entry.get("cik_str", "")

        name_to_cik[title] = cik
        names.append(title)

    # First: Try exact match
    if clean_name in name_to_cik:
        return {
            "company": company_name,
            "matched_name": clean_name,
            "cik": name_to_cik[clean_name],
            "match_type": "exact"
        }

    # Second: Fuzzy semantic match using difflib (very effective for LLM-misspellings)
    best_match = difflib.get_close_matches(clean_name, names, n=1, cutoff=0.55)

    if best_match:
        match = best_match[0]
        cik = name_to_cik[match]
        return {
            "company": company_name,
            "matched_name": match,
            "cik": cik,
            "match_type": "fuzzy"
        }

    # No match found
    return {
        "company": company_name,
        "error": "No matching company found in SEC ticker database."
    }
