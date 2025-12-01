"""
EDGAR Submissions Tool (CIK-Only, Async)
Now supports short numeric CIK such as "320193" -> "0000320193"

Fetches:
    https://data.sec.gov/submissions/CIK##########.json
"""

import aiohttp
import backoff
import json
import sys
import re
import asyncio
from typing import Optional, Dict


# -----------------------
# SEC-Compliant Headers
# -----------------------
SEC_HEADERS = {
    "User-Agent": "KiarashAI/1.0 (kiarash996@yahoo.com)",  # update for production
    "Accept": "application/json",
    "Content-Type": "application/json",
}

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"


# -----------------------
# Helper: Rate Limit Check
# -----------------------
def is_rate_limit_error(exc) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 503, 504)
    return False


# -----------------------
# Helper: Normalize CIK
# -----------------------
def normalize_cik_to_10(cik: str) -> Optional[str]:
    """
    Accepts:
        "320193"
        "0000320193"
        "  320193  "
        "CIK 320193" (LLM-safe cleanup)

    Normalizes to:
        "0000320193"

    Returns:
        10-digit CIK string, or None if invalid.
    """
    if not cik:
        return None

    # Extract digits only (safe for LLM messy input)
    digits = re.sub(r"\D", "", cik)

    if not digits:
        return None

    # CIK cannot exceed 10 digits
    if len(digits) > 10:
        return None

    # Left-pad with zeros to 10 digits
    return digits.zfill(10)


# -----------------------
# Main Submissions Fetcher (with backoff)
# -----------------------
@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_time=60,
    jitter=backoff.full_jitter,
)
async def fetch_submissions(cik10: str) -> Dict:
    url = SUBMISSIONS_URL.format(cik10=cik10)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=SEC_HEADERS, timeout=30) as resp:
                resp.raise_for_status()
                return await resp.json()

    except aiohttp.ClientResponseError as e:
        return {"error": f"HTTP {e.status}: {e.message}"}

    except aiohttp.ClientConnectionError as e:
        return {"error": f"Connection error: {str(e)}"}

    except asyncio.TimeoutError:
        return {"error": "Request timed out."}

    except Exception as e:
        return {"error": str(e)}


# -----------------------
# High-Level Tool Logic
# -----------------------
async def submissions_tool(cik_input: str) -> Dict:
    """
    Accepts any numeric CIK (short or full):
        "320193"
        "0000320193"

    Returns parsed submissions JSON or structured errors.
    """

    cik10 = normalize_cik_to_10(cik_input)
    if not cik10:
        return {
            "error": "Invalid CIK. Provide a numeric value up to 10 digits "
                     "(e.g., '320193' or '0000320193')."
        }

    # Fetch submissions
    data = await fetch_submissions(cik10)

    # Return structured result
    if "error" in data:
        return {"cik10": cik10, "error": data["error"]}

    return {"cik10": cik10, "submissions": data}
