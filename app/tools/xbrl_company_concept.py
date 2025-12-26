"""
Official SEC XBRL Company-Concept API Tool (Async)
Follows A2A/MCP-safe patterns, with strict rate-limit handling.
"""

import aiohttp
import backoff
import sys
import json
from typing import Optional
from tools.edgar_submissions import normalize_cik_to_10
from tools.company_CIK import get_env_var

# --- Required SEC Headers ---
name = get_env_var("YOUR_NAME")
email = get_env_var("EMAIL_ADDRESS")
SEC_HEADERS = {
        "User-Agent" : f"{name} ({email})",
        "Accept": "application/json",
}

BASE_URL = "https://data.sec.gov/api/xbrl/companyconcept"


def normalize_taxonomy(tax: str) -> str:
    """
    Normalize taxonomy names:
    - lowercase
    - replace underscores with hyphens
    - strip spaces
    Valid examples:
        us_gaap   -> us-gaap
        US-GAAP   -> us-gaap
        ifrs_full -> ifrs-full
    """
    return tax.strip().lower().replace("_", "-").replace(" ", "")


def normalize_concept(concept: str) -> str:
    """
    Normalize concept/tag names.
    Usually camelCase or PascalCase — SEC is case-sensitive.
    Here we only strip spaces.
    """
    return concept.strip().replace(" ", "")


def is_rate_limit_error(exc):
    """
    Rate-limit and transient server errors from SEC.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 503, 504)
    return False


# --------------------- Main API Function ---------------------

@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_time=45,
    jitter=backoff.full_jitter,
)
async def fetch_company_concept(cik10: str, taxonomy: str, concept: str) -> dict:
    """
    Fetches the XBRL company-concept JSON for a given entity (CIK) and concept.

    Args:
        cik10: Raw CIK (will be normalized)
        taxonomy: XBRL taxonomy (e.g., "us-gaap")
        concept: Concept tag (e.g., "AccountsPayableCurrent")

    Returns:
        dict containing full SEC JSON OR {"error": "..."}
    """

    cik_norm = normalize_cik_to_10(cik10)
    tax_norm = normalize_taxonomy(taxonomy)
    concept_norm = normalize_concept(concept)

    url = f"{BASE_URL}/CIK{cik_norm}/{tax_norm}/{concept_norm}.json"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=SEC_HEADERS) as resp:
                resp.raise_for_status()
                return await resp.json()

    except aiohttp.ClientResponseError as e:
        return {
            "error": f"SEC returned HTTP {e.status}",
            "details": e.message,
            "normalized_request": {
                "cik": cik_norm,
                "taxonomy": tax_norm,
                "concept": concept_norm,
                "url": url,
            },
        }

    except aiohttp.ClientConnectionError as e:
        return {"error": f"Connection error: {str(e)}"}

    except Exception as e:
        return {"error": f"Unknown error: {str(e)}"}
