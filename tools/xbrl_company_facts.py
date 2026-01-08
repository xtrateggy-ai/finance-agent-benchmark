"""
Official SEC XBRL CompanyFacts API Tool (Async)
Fetches all XBRL facts for a company in a single JSON call.
"""

import aiohttp
import backoff
import sys
import asyncio
from typing import Dict
from tools.edgar_submissions import normalize_cik_to_10
from tools.company_CIK import get_env_var

# --- SEC-Compliant Headers ---
name = get_env_var("YOUR_NAME")
email = get_env_var("EMAIL_ADDRESS")
SEC_HEADERS = {
        "User-Agent" : f"{name} ({email})",
        "Accept": "application/json",
}

BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

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
    cik10 = normalize_cik_to_10(cik_input)
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
