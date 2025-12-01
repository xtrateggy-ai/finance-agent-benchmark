"""
Official SEC XBRL Frames API Tool (Async)
Fetches the most recent fact for a reporting entity based on period, taxonomy, concept, and unit.
"""

import aiohttp
import backoff
import sys
import asyncio
from typing import Dict

# --- SEC-Compliant Headers ---
SEC_HEADERS = {
    "User-Agent": "KiarashAI/1.0 (kiarash996@yahoo.com)",
    "Accept": "application/json",
}

BASE_URL = "https://data.sec.gov/api/xbrl/frames"


# ------------------------
# Helpers
# ------------------------
def normalize_taxonomy(tax: str) -> str:
    """
    Normalize taxonomy: lowercase, replace underscores with hyphen
    """
    return tax.strip().lower().replace("_", "-").replace(" ", "")


def normalize_concept(concept: str) -> str:
    """
    Normalize concept name: strip spaces
    """
    return concept.strip().replace(" ", "")


def normalize_unit(unit: str) -> str:
    """
    Normalize unit of measure (e.g., USD, USD-per-shares)
    """
    return unit.strip().replace(" ", "")


def normalize_period(period: str) -> str:
    """
    Normalize period format:
    - CY####  -> annual
    - CY####Q# -> quarterly
    - CY####Q#I -> instantaneous
    Only very basic validation here (pattern match could be improved)
    """
    period = period.strip().upper()
    if period.startswith("CY") and (len(period) >= 6):
        return period
    return period  # let SEC API reject invalid periods


def is_rate_limit_error(exc) -> bool:
    """
    Rate-limit / transient server errors
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 503, 504)
    return False


# ------------------------
# Fetch Frames API
# ------------------------
@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_time=45,
    jitter=backoff.full_jitter,
)
async def fetch_frames(
    taxonomy: str,
    concept: str,
    unit: str,
    period: str,
) -> Dict:
    """
    Fetch the XBRL frame data for a given combination of taxonomy, concept, unit, period
    """

    tax_norm = normalize_taxonomy(taxonomy)
    concept_norm = normalize_concept(concept)
    unit_norm = normalize_unit(unit)
    period_norm = normalize_period(period)

    url = f"{BASE_URL}/{tax_norm}/{concept_norm}/{unit_norm}/{period_norm}.json"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=SEC_HEADERS, timeout=30) as resp:
                resp.raise_for_status()
                return await resp.json()

    except aiohttp.ClientResponseError as e:
        return {
            "error": f"HTTP {e.status}",
            "details": e.message,
            "url": url,
            "normalized_request": {
                "taxonomy": tax_norm,
                "concept": concept_norm,
                "unit": unit_norm,
                "period": period_norm,
            }
        }

    except aiohttp.ClientConnectionError as e:
        return {"error": f"Connection error: {str(e)}", "url": url}

    except asyncio.TimeoutError:
        return {"error": "Request timed out.", "url": url}

    except Exception as e:
        return {"error": f"Unknown error: {str(e)}", "url": url}
