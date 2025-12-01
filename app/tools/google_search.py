# app/tools/google_search.py
import os
import sys
import httpx
#from agentbeats import tool

# Set and sys.path and Load environment variable 
from utils.env_setup import init_environment
init_environment()

#@tool
async def google_search(query: str, verbose: False) -> dict:
    """Google web search for financial context."""
    #if verbose:
    print(f"[GOOGLE_SEARCH] Query={query}", file=sys.stderr)
        
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    if not api_key or not cx:
        #if verbose:
        print(f"[GOOGLE_SEARCH] GOOGLE_API_KEY or GOOGLE_CX missing, returning error.", file=sys.stderr)
        return {"error": "GOOGLE_API_KEY or GOOGLE_CX missing"}
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": 5}
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            print(f"[GOOGLE_SEARCH] Calling client.get().", file=sys.stderr)
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return {"results": [item.get("snippet", "") for item in items]}
        except Exception as e:
            return {"error": str(e)}