"""
Official SEC XBRL CompanyFacts API Tool (Async)
Fetches all XBRL facts for a company in a single JSON call.
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

BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"


# ------------------------
# Helpers
# ------------------------
def normalize_cik(cik: str) -> str:
    """
    Convert any numeric CIK to 10-digit string with leading zeros.
    """
    if not cik:
        return None
    digits = "".join(filter(str.isdigit, str(cik)))
    if not digits or len(digits) > 10:
        return None
    return digits.zfill(10)


def is_rate_limit_error(exc):
    """
    SEC rate-limit and transient errors.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 503, 504)
    return False


# ------------------------
# Fetch CompanyFacts
# ------------------------
@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_time=45,
    jitter=backoff.full_jitter,
)
async def fetch_companyfacts(cik_input: str) -> dict:
    """
    Fetch all XBRL company facts for a given CIK.
    """
    cik10 = normalize_cik(cik_input)
    if not cik10:
        return {"error": "Invalid CIK. Provide a numeric value up to 10 digits (e.g., '320193' or '0000320193')."}

    url = f"{BASE_URL}/CIK{cik10}.json"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=SEC_HEADERS, timeout=30) as resp:
                resp.raise_for_status()
                return await resp.json()

    except aiohttp.ClientResponseError as e:
        return {
            "error": f"HTTP {e.status}",
            "details": e.message,
            "url": url
        }

    except aiohttp.ClientConnectionError as e:
        return {"error": f"Connection error: {str(e)}", "url": url}

    except asyncio.TimeoutError:
        return {"error": "Request timed out.", "url": url}

    except Exception as e:
        return {"error": f"Unknown error: {str(e)}", "url": url}
