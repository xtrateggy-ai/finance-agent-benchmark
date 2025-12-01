# app/tools/edgar_search.py
"""
SEC EDGAR Search Tool
Based on Vals.ai EDGARSearch class with async support
"""
import os
import sys
import json
import aiohttp
import backoff
from utils.env_setup import init_environment

init_environment()


def is_429(exception):
    """Check if exception is a 429 rate limit error"""
    return (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )


@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    max_tries=8,
    base=2,
    factor=3,
    jitter=backoff.full_jitter,
    giveup=lambda e: not is_429(e),
)
async def edgar_search(
    query: str,
    form_types: list = None,
    ciks: list = None,
    start_date: str = None,
    end_date: str = None,
    page: int = 1,
    top_n_results: int = 5
) -> dict:
    """
    Search the EDGAR Database through the SEC API.
    
    Args:
        query: The keyword or phrase to search
        form_types: List of form types to search (e.g., ['10-K', '10-Q'])
        ciks: List of CIKs to filter by
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format  
        page: Pagination for results
        top_n_results: The top N results to return
        
    Returns:
        dict: {"filings": [...]} or {"error": "..."}
    """
    api_key = os.getenv("SEC_API_KEY")
    
    if not api_key:
        print(f"[EDGAR_SEARCH] SEC_API_KEY not set, returning error.", file=sys.stderr)
        return {"error": "SEC_API_KEY not set"}
    
    # Parse form_types if it's a string representation
    if isinstance(form_types, str) and form_types.startswith("["):
        try:
            form_types = json.loads(form_types.replace("'", '"'))
        except json.JSONDecodeError:
            form_types = [item.strip(" \"'") for item in form_types[1:-1].split(",")]
    
    # Parse ciks if it's a string representation
    if isinstance(ciks, str) and ciks.startswith("["):
        try:
            ciks = json.loads(ciks.replace("'", '"'))
        except json.JSONDecodeError:
            ciks = [item.strip(" \"'") for item in ciks[1:-1].split(",")]
    
    # Build payload
    payload = {
        "query": query,
        "formTypes": form_types or [],
        "ciks": ciks or [],
        "startDate": start_date or "2024-01-01",
        "endDate": end_date or "2024-12-31",
        "page": str(page),
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sec-api.io/full-text-search",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
        filings = result.get("filings", [])[:top_n_results]
        return {"filings": filings}
        
    except Exception as e:
        return {"error": f"SEC API error: {str(e)}"}
