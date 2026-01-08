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

def get_env_var(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set")
    return value

name = get_env_var("YOUR_NAME")
email = get_env_var("EMAIL_ADDRESS")
SEC_HEADERS = {
        "User-Agent" : f"{name} ({email})",
        "Accept": "application/json",
}

def clean_company_name(name: str) -> str:
    """Normalize company name for matching"""
    s = name.upper()
    s = re.sub(r"\b(CORPORATION|CORP|INC|LLC|LLP|LTD|LIMITED|CO|COMPANY|PLC|AG|SA)\b\.?", "", s)
    s = re.sub(r"[.,&''\-—–/\\()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^THE\s+", "", s)
    s = re.sub(r"\s+THE$", "", s)
    return s.lower().strip()

# -------lookup in archive files for company cik---------
async def get_cik_from_archive(company_name: str,
                               use_disk_cache: bool = False,
                               cache_filename: str = None
                               ) -> Optional[str]:
    """Fallback: search SEC's archive"""
    search_lower = clean_company_name(company_name.strip())

    print(f"[SEC] Looking up CIK for: {company_name}")
    
    file_loaded=False
    data = ""
    try:
        # Disk cache check
        if use_disk_cache and cache_filename:
            cache_dir = Path("data/sec")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / cache_filename

            if cache_file.exists():
                try:
                    data = cache_file.read_text(encoding='utf-8')
                    print(f"[CACHE] get_cik_from_archive(): Read from disk: {cache_filename}", flush=True)
                    file_loaded=True
                    
                except Exception as e:
                    print(f"[CACHE] get_cik_from_archive(): Failed to read {cache_filename}: {e}")
                    return ""
        
        # If file is not loaded, use SEC Website
        if file_loaded == False:
                       
            async with aiohttp.ClientSession() as session:
                print("[SEC_SEARCH] Starting download of cik-lookup-data.txt (~20MB)...")
                async with session.get(
                    "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt",
                    headers=SEC_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        return None
    
                    content = b""
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        if chunk:
                            content += chunk
    
                    data = content.decode('utf-8', errors='ignore')
                    print(f"[SEC] FULL FILE DOWNLOADED: {len(data):,} characters")
                    
                    # Save to disk, if flag is True
                    if use_disk_cache and cache_filename:
                        try:
                            cache_file.write_text(data, encoding="utf-8")
                            print(f"[SEC] get_cik_from_archive(): Saved to disk: {cache_filename}")
                        except Exception as e:
                            print(f"[SEC] get_cik_from_archive(): Save error: {e}")
        
        # Look for CIK number
        for line in data.splitlines():
            if ':' not in line:
                continue
            
            name_part, cik_part = line.split(":", 1)
            cik = cik_part.strip().zfill(10).split(":", 1)[0]
            clean_name = clean_company_name(name_part)
            
            if cik != "0000000000" and clean_name == search_lower:
                return cik

        return None

    except Exception:
        return None

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
