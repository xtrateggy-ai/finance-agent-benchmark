# app/tools/serp_search.py
import os
import sys
import httpx

# Set and sys.path and Load environment variable 
from utils.env_setup import init_environment
init_environment()


async def serp_search(query: str, verbose: bool = False) -> dict:
    """SERP API search for rich organic results."""
    if verbose:
        print(f"serp_search(): query={query}", file=sys.stderr)
    
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return {"error": "SERP_API_KEY missing"}
    
    url = "https://serpapi.com/search"
    params = {
        "api_key": api_key,
        "q": query,
        "num": 5
    }
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract organic results
            organic = data.get("organic_results", [])
            results = []
            for item in organic[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
            
            return {"results": results}
            
        except Exception as e:
            return {"error": str(e)}
