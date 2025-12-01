#!/usr/bin/env python3
"""
Enhanced SEC Search Tool - Complete Implementation
Handles all question types from public.csv with robust extraction
"""

import os
import sys
import re
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from llama_index.core import Document
from bs4 import BeautifulSoup
from tools.local_llm_rag import QuestionAnsweringExtractor

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_company_name(name: str) -> str:
    """Normalize company name for matching"""
    s = name.upper()
    s = re.sub(r"\b(CORPORATION|CORP|INC|LLC|LLP|LTD|LIMITED|CO|COMPANY|PLC|AG|SA)\b\.?", "", s)
    s = re.sub(r"[.,&''\-—–/\\()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^THE\s+", "", s)
    s = re.sub(r"\s+THE$", "", s)
    return s.lower().strip()


"""
def _filter_by_date(
    filings: List[dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    #Filter filings by date range
    if not start_date and not end_date:
        return filings
    
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    def in_range(d: str) -> bool:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if start and dt < start:
            return False
        if end and dt > end:
            return False
        return True

    return [f for f in filings if in_range(f["filing_date"])]
"""

# ============================================================
# CIK LOOKUP
# ============================================================

async def get_cik_from_ticker_or_name(
    company_name: str = None,
    ticker_symbol: str = None,
    headers: dict = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve company identifier to 10-digit CIK.
    Returns: (cik, ticker)
    """
    wcompany_name = company_name.strip() if company_name else ""
    wticker_symbol = ticker_symbol.strip() if ticker_symbol else ""
    
    if not wcompany_name and not wticker_symbol:
        return None, None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.sec.gov/files/company_tickers_exchange.json",
                headers=headers or {"User-Agent": "Finance-Agent contact@example.com"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    return None, None
                resp.raise_for_status()
                data = await resp.json()

        rows = data.get("data", [])
        company_clean = clean_company_name(wcompany_name)
        ticker_upper = wticker_symbol.upper()

        for row in rows:
            if len(row) < 4:
                continue
            
            #cik_raw, name, ticker = row[0], str(row[1]).lower(), str(row[2]).strip().upper()

            cik_raw  = row[0]                                  #  0 = CIK (int or str)         
            name     = clean_company_name(str(row[1]).lower()) #  1 = Legal name
            ticker   = str(row[2]).strip().upper()             #  2 = Ticker
            #exchange = row[3]                                  #  3 = Exchange

            #print(f"[SEC_SEARCH] row[0]={row[0]} row[1]={row[1]} row[2]={row[2]}")
            #print(f"[SEC_SEARCH] Comparying query_lower={query_lower} name={name}")
                
            
            try:
                cik_str = str(int(cik_raw)).zfill(10)
            except (ValueError, TypeError):
                continue

            if ticker == ticker_upper:
                print(f"[SEC] Ticker '{ticker}' match → CIK {cik_str}")
                return cik_str, ticker

            if company_clean == clean_company_name(name):
                print(f"[SEC_SEARCH] Name match '{name[:50]}...' ticker={ticker} → CIK {cik_str}")
                return cik_str, ticker

        return None, None

    except Exception as e:
        print(f"[SEC] Lookup failed: {e}", file=sys.stderr)
        return None, None


async def get_cik_from_archive(company_name: str,
                               use_disk_cache: bool = False,
                               cache_filename: str = None
                               ) -> Optional[str]:
    """Fallback: search SEC's archive"""
    search_lower = clean_company_name(company_name.strip())

    headers = {
        "User-Agent": "Finance-Agent contact@example.com",
        "Accept-Encoding": "gzip, deflate",
    }

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
                    headers=headers,
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


# ============================================================
# HTML/XML FETCHING
# ============================================================
async def fetch_filing_and_exhibit_html(
    url: str,
    cik: str,
    accession: str,
    headers: dict,
    timeout: int = 90,
    use_disk_cache: bool = False,
    cache_filename: str = None
) -> Tuple[str, Optional[str]]:
    """
    Fetch filing AND its Exhibit 99.1 (if exists) in one go.
    
    Returns:
        (filing_text, exhibit_text)
    """
    #print(f"[DEBUG] Starting fetch for accession: {accession}")
    
    # ════════════════════════════════════════════════════════════
    # STEP 1: Check disk cache for both files
    # ════════════════════════════════════════════════════════════
    filing_text = None
    exhibit_text = None
    
    if use_disk_cache and cache_filename:
        cache_dir = Path("data/sec")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        filing_cache = cache_dir / cache_filename
        exhibit_cache = cache_dir / cache_filename.replace('.txt', '_ex991.txt')
        
        # Try to load from cache
        if filing_cache.exists():
            try:
                filing_text = filing_cache.read_text(encoding='utf-8')
                print(f"[CACHE] Read filing from disk: {cache_filename}", flush=True)
            except Exception:
                pass
        
        if exhibit_cache.exists():
            try:
                exhibit_text = exhibit_cache.read_text(encoding='utf-8')
                print(f"[CACHE] Read exhibit from disk: {cache_filename.replace('.txt', '_ex991.txt')}", flush=True)
            except Exception:
                pass
        
        # If both cached, return them
        if filing_text and exhibit_text:
            return filing_text, exhibit_text
        
        if filing_text and not exhibit_text:
            # Filing cached but no exhibit - might not exist
            return filing_text, None
    
    # ════════════════════════════════════════════════════════════
    # STEP 2: Download original HTML (not cached, or cache miss)
    # ════════════════════════════════════════════════════════════
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch original HTML
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 429:
                        wait_time = int(response.headers.get('Retry-After', 10))
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status != 200:
                        return "", None
                    
                    original_html = await response.text()
                
                # ════════════════════════════════════════════════════════
                # STEP 3: Look for Exhibit 99.1 link in ORIGINAL HTML
                # ════════════════════════════════════════════════════════
                exhibit_url = None
                soup = BeautifulSoup(original_html, 'html.parser')
                
                for tag in soup.find_all(['a', 'td', 'tr']):
                    text = tag.get_text(strip=True).lower()
                    href = None
                    
                    if tag.name == 'a' and tag.get('href'):
                        href = tag['href']
                    elif tag.name in ['td', 'tr'] and '99.1' in text:
                        a_tag = tag.find('a', href=True)
                        if a_tag:
                            href = a_tag['href']
                    
                    if not href:
                        continue
                    
                    href_lower = href.lower()
                    
                    # Check if this is Exhibit 99.1
                    match_conditions = [
                        '99.1' in text,
                        'exhibit 99' in text,
                        'press release' in text,
                        'earnings' in href_lower and '.htm' in href_lower,
                        'press' in href_lower and '.htm' in href_lower,
                        'q1fy' in href_lower or 'q2fy' in href_lower or 
                        'q3fy' in href_lower or 'q4fy' in href_lower,
                    ]
                    
                    if any(match_conditions):
                        # Build full URL
                        if href.startswith('http'):
                            exhibit_url = href
                        elif href.startswith('/'):
                            exhibit_url = f"https://www.sec.gov{href}"
                        else:
                            acc_clean = accession.replace('-', '')
                            exhibit_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{href}"
 
                        if exhibit_url:
                            print(f"[EXHIBIT] URL found: {exhibit_url}")
                        else:
                            print(f"[EXHIBIT] No URL found in 8-K {accession}")
                            
                        print(f"   [EXHIBIT] Found: {href}", flush=True)
                        break
                
                # ════════════════════════════════════════════════════════
                # STEP 4: Fetch Exhibit 99.1 if found
                # ════════════════════════════════════════════════════════
                if exhibit_url:
                    try:
                        async with session.get(
                            exhibit_url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as ex_response:
                            if ex_response.status == 200:
                                exhibit_html = await ex_response.text()
                                
                                # Clean exhibit HTML
                                ex_soup = BeautifulSoup(exhibit_html, 'html.parser')
                                for tag in ex_soup(['script', 'style']):
                                    tag.decompose()
                                
                                exhibit_text = ex_soup.get_text(separator="\n", strip=True)
                                lines = [line.strip() for line in exhibit_text.splitlines() if line.strip()]
                                exhibit_text = "\n".join(lines)
                                
                                print(f"   [EXHIBIT] Fetched: {len(exhibit_text):,} chars", flush=True)
                    except Exception as e:
                        print(f"   [EXHIBIT] Fetch failed: {e}", flush=True)
                        exhibit_text = None
                
                # ════════════════════════════════════════════════════════
                # STEP 5: Clean main filing HTML
                # ════════════════════════════════════════════════════════
                soup = BeautifulSoup(original_html, 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                
                filing_text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in filing_text.splitlines() if line.strip()]
                filing_text = "\n".join(lines)
                
                # ════════════════════════════════════════════════════════
                # STEP 6: Save to cache
                # ════════════════════════════════════════════════════════
                if use_disk_cache and cache_filename:
                    try:
                        filing_cache.write_text(filing_text, encoding='utf-8')
                        print(f"[CACHE] Saved filing: {cache_filename}", flush=True)
                        
                        if exhibit_text:
                            exhibit_cache.write_text(exhibit_text, encoding='utf-8')
                            print(f"[CACHE] Saved exhibit: {cache_filename.replace('.txt', '_ex991.txt')}", flush=True)
                    except Exception as e:
                        print(f"[CACHE] Save error: {e}")
                
                return filing_text, exhibit_text
                
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"[SEC] Error: {e}")
            return "", None
    
    return "", None

# ============================================================
# SECTION EXTRACTION (Enhanced for all question types)
# ============================================================

def extract_all_sections(text: str, form: str, keywords: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract all relevant sections based on form type and keywords.
    Handles: M&A, guidance, ARPPU, board nominations, margins
    """
    sections = {}
    form_upper = form.upper()
    kw_lower = [k.lower() for k in (keywords or [])]
    
    # ════════════════════════════════════════════════════════
    # 8-K: Material Events (M&A, Earnings, Board Changes)
    # ════════════════════════════════════════════════════════
    if "8-K" in form_upper:
        patterns = {
            'item_1_01': r'(Item\s+1\.01.*?)(?=Item\s+\d|\Z)',  # Entry into agreements
            'item_2_01': r'(Item\s+2\.01.*?)(?=Item\s+\d|\Z)',  # Completion of acquisitions
            'item_2_02': r'(Item\s+2\.02.*?)(?=Item\s+\d|\Z)',  # Earnings releases
            'item_5_02': r'(Item\s+5\.02.*?)(?=Item\s+\d|\Z)',  # Director changes
            'item_8_01': r'(Item\s+8\.01.*?)(?=Item\s+\d|\Z)',  # Other events
            'ma_activity': r'([Mm]erger|[Aa]cquisition|[Dd]ivestiture|[Dd]eal)[^\n]*.*?(?=\n\n|\Z)',
            
            # GUIDANCE EXTRACTION (Question #3, #4)
            'guidance_section': r'(?:guidance|outlook|expects?|projections?|forecasts?)[\s\S]{0,4000}',
            'earnings_release': r'(?:EARNINGS RELEASE|RESULTS OF OPERATIONS|QUARTERLY RESULTS)[\s\S]{0,6000}',
            'financial_outlook': r'(?:Financial Outlook|Business Outlook|2025 Outlook)[\s\S]{0,5000}',
            
            # Board nominations (Question #5)
            'board_changes': r'(?:director.*(?:appointed|elected|resigned|nominated))[\s\S]{0,3000}',
        }
    
    # ════════════════════════════════════════════════════════
    # DEF 14A: Proxy Statements (Board Nominations)
    # ════════════════════════════════════════════════════════
    elif "DEF 14A" in form_upper or "DEFA14A" in form_upper:
        patterns = {
            'board_nominees': r'(?:ELECTION OF DIRECTORS|PROPOSAL.*ELECT.*DIRECTOR|DIRECTOR NOMINEES?)[\s\S]{0,8000}',
            'nominee_details': r'(?:Class\s+[I]+\s+Director|Nominee[s]?\s+for\s+Election)[\s\S]{0,5000}',
            'director_qualifications': r'(?:DIRECTOR QUALIFICATIONS|NOMINEE BIOGRAPHIES?)[\s\S]{0,8000}',
            'proposal_1': r'(PROPOSAL\s+(?:1|ONE).*?)(?=PROPOSAL\s+(?:2|TWO)|\Z)',
        }
    
    # ════════════════════════════════════════════════════════
    # 10-K/10-Q: Financial Reports (ARPPU, Revenue, Margins)
    # ════════════════════════════════════════════════════════
    else:
        patterns = {
            'business': r'(Item\s+1\.?\s+Business.*?)(?=Item\s+1A|\Z)',
            'risk_factors': r'(Item\s+1A\.?\s+Risk\s+Factors.*?)(?=Item\s+1B|\Z)',
            'mda': r'(Item\s+7\.?\s+Management[^\n]*.*?)(?=Item\s+7A|Item\s+8|\Z)',
            'financial_statements': r'(Item\s+8\.?\s+Financial\s+Statements.*?)(?=Item\s+9|\Z)',
            
            # ARPPU/Subscriber metrics (Question #2)
            'subscriber_metrics': r'(?:paid.*member|subscriber|streaming.*member|average.*user)[\s\S]{0,3000}',
            'revenue_per_user': r'(?:average.*revenue.*(?:per|/)|ARPU|ARPPU)[\s\S]{0,3000}',
            'membership_table': r'(?:membership|subscriber).*(?:table|data)[\s\S]{0,5000}',
            
            # Quarterly data (Question #3)
            'quarterly_table': r'(?:Q[1-4]|First|Second|Third|Fourth).*?Quarter[\s\S]{0,5000}',
            'three_months_ended': r'(?:Three|3)\s+Months\s+Ended[\s\S]{0,4000}',
            
            # Margin analysis
            'margin_analysis': r'(?:gross margin|operating margin|pre-tax margin|profit margin)[\s\S]{0,3000}',
        }
    
    # Extract all matching patterns
    for name, pattern in patterns.items():
        try:
            m = re.search(pattern, text, re.I | re.S)
            if m:
                content = m.group(0) if m.lastindex is None else m.group(1)
                sections[name] = content[:12000].strip()  # Increased limit
        except Exception:
            continue
    
    """
    # Keyword filtering (but don't remove everything)
    if keywords and sections:
        filtered = {}
        for key, value in sections.items():
            if any(kw in value.lower() for kw in kw_lower):
                filtered[key] = value
        
        if filtered:
            sections = filtered
    """
    
    return sections


# ════════════════════════════════════════════════════════
# FINANCIAL DATA EXTRACTION (Enhanced)
# ════════════════════════════════════════════════════════

def extract_financial_data(text: str) -> Dict[str, list]:
    """
    Extract financial metrics with enhanced patterns.
    Returns: {"metric_name": ["value1", "value2", ...]}
    """
    financial_data = {}
    
    metrics = {
        # Standard financials
        'total_revenue': r'Total\s+(?:net\s+)?revenues?[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)',
        'total_assets': r'Total\s+assets[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)',
        'total_liabilities': r'Total\s+liabilities[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)',
        'stockholders_equity': r"Total\s+stockholders[\'\u2019]?\s+equity[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)",
        'net_income': r'Net\s+(?:income|earnings)[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)',
        'operating_cash_flow': r'(?:Net\s+)?[Cc]ash\s+(?:provided\s+by|from)\s+operating\s+activities[\s\n:]*\$?[\s\n]*([\(\)]?[\d,]+[\(\)]?)',
        
        # ENHANCED: Subscriber/Membership metrics (Question #2)
        'paid_memberships': r'(?:paid|average).*?member(?:ship)?s?[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*(?:million)?',
        'streaming_members': r'streaming.*?member(?:ship)?s?[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*(?:million)?',
        'average_memberships': r'average.*?(?:paying\s+)?member(?:ship)?s?[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*(?:million)?',
        
        # ENHANCED: ARPPU (Question #2)
        'arppu': r'(?:ARPPU|average\s+revenue\s+per.*?user|revenue\s+per.*?member)[\s\n:]*\$?[\s\n]*([\d,]+(?:\.\d+)?)',
        
        # ENHANCED: Margins (Question #3)
        'gross_margin_pct': r'gross\s+(?:profit\s+)?margin[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*%',
        'operating_margin_pct': r'operating\s+(?:profit\s+)?margin[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*%',
        'pretax_margin_pct': r'pre[\-\s]?tax\s+(?:profit\s+)?margin[\s\n:]*[\s\n]*([\d,]+(?:\.\d+)?)\s*%',
    }
    
    for metric_name, pattern in metrics.items():
        matches = re.findall(pattern, text, re.I)
        if matches:
            cleaned = []
            seen = set()
            
            for match in matches[:10]:  # Increased from 5
                value = match.strip()
                
                # Convert (123) to -123
                if value.startswith('(') and value.endswith(')'):
                    value = '-' + value[1:-1]
                
                if value in seen:
                    continue
                seen.add(value)
                cleaned.append(value)
            
            if cleaned:
                financial_data[metric_name] = cleaned

    
    # ✅ ENHANCED: Use new extraction functions
    pretax_data = extract_pretax_margin_data(text)
    if pretax_data:
        financial_data['pretax_margin_data'] = pretax_data
    
    """
    arppu_data = extract_arppu_data(text)
    if arppu_data:
        financial_data['arppu_data'] = arppu_data
    
    membership_data = extract_membership_data(text)
    if membership_data:
        financial_data['membership_data'] = membership_data
    """


    # ─────────────────────────────────────────────────────────────
    """
    Enhanced ARPPU extraction that handles regional breakouts.
    
    Netflix reports ARPPU by region:
    - UCAN (US/Canada)
    - EMEA (Europe/Middle East/Africa)  
    - LATAM (Latin America)
    - APAC (Asia-Pacific)
    
    We want the OVERALL average, not regional values.
    """
    # ─────────────────────────────────────────────────────────────
    result = {}
    
    # ─────────────────────────────────────────────────────────────
    # Pattern 1: Overall/Consolidated ARM/ARPPU
    # ─────────────────────────────────────────────────────────────

    safe_values = []
    patterns = [
        r"average\s+monthly\s+revenue\s+per\s+paying\s+membership\s+[\$€]?\s*(\d+\.\d{2})",
        r"average\s+revenue\s+per\s+paid\s+member[\s:]+[\$€]?\s*(\d+\.\d{2})",
        r"ARPPU.*?[\$€]?\s*(\d+\.\d{2})",
        r"global\s+average.*?[\$€]?\s*(\d+\.\d{2})",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m)
                if 5.0 <= val <= 25.0:  # Netflix global ARPPU is always ~$8–$18
                    safe_values.append(val)
            except:
                continue

    if safe_values:
        unique = sorted(set(safe_values))
        result['arppu_direct'] = unique[:5]
        #result['source'] = 'safe_fallback'
        #result['extraction_method'] = 'final_safe_layer'
        
        from collections import Counter
        most_common = Counter(safe_values).most_common(1)[0][0]
        result['arppu_overall'] = [most_common]
        result['source'] = 'global_mode_from_direct'
        #print(f"[ARPPU] Global ARPPU selected via mode: ${most_common:.2f}")
            
    # ─────────────────────────────────────────────────────────────
    # Pattern 2: Regional ARPPU (for context)
    # ─────────────────────────────────────────────────────────────
    # regional_pattern = r'(UCAN|EMEA|LATAM|APAC)[\s\S]{0,50}?\$?([\d.]+)'
    # regional_matches = re.findall(regional_pattern, text, re.I)
    
    # if regional_matches:
    #     regions = {}
    #     for region, value in regional_matches:
    #         try:
    #             val = float(value)
    #             if 5 <= val <= 25:  # Reasonable range
    #                 regions[region.upper()] = val
    #         except ValueError:
    #             continue
        
    #     if regions:
    #         result['arppu_by_region'] = regions
            
    #         # Calculate weighted average if we have memberships
    #         # For now, just note that regional data exists
    #         if 'arppu_overall' not in result and len(regions) > 0:
    #             # Use highest value (usually UCAN) as proxy
    #             result['arppu_overall'] = [max(regions.values())]
    #             result['source'] = 'regional_max'    

    #         # Only set arppu_overall from regions if we have NO direct global match
    #         # AND we have UCAN (most representative high-value region)
    #         # if ('arppu_overall' not in result and 
    #         #     len(regions) >= 2 and 
    #         #     'UCAN' in regions):
    #         #     ucan_val = regions['UCAN']
    #         #     result['arppu_overall'] = [ucan_val]
    #         #     result['source'] = 'regional_ucan_proxy'
    #         #    print(f"[ARPPU] Using UCAN as global proxy: ${ucan_val}")
                
    # Add to the financial_data
    financial_data["arppu_data"] = result

    # ─────────────────────────────────────────────────────────────
    # Extract membership counts with better accuracy.
    # 
    # Issues fixed:
    # - Don't capture commas as separate values
    # - Handle "million" suffix properly
    # - Avoid random numbers
    # ─────────────────────────────────────────────────────────────    
    result = {}
    
    # Pattern 1: "X.X million paid memberships"
    # Match: "280.6 million paid memberships" or "paid memberships of 280.6 million"
    pattern1 = r'(?:paid|streaming|total)\s+member(?:ship)?s?\s+(?:of\s+)?([\d.]+)\s+million'
    matches = re.findall(pattern1, text, re.I)
    if matches:
        result['memberships_millions'] = [float(m) for m in matches[:10]]
    
    # Pattern 2: Reverse order - "million paid memberships"
    pattern2 = r'([\d.]+)\s+million\s+(?:paid|streaming|total)\s+member(?:ship)?s?'
    matches = re.findall(pattern2, text, re.I)
    if matches:
        if 'memberships_millions' not in result:
            result['memberships_millions'] = []
        result['memberships_millions'].extend([float(m) for m in matches[:10]])
    
    # Pattern 3: Table format "Paid memberships    280.6"
    # Only accept reasonable values (50-500 million range for Netflix)
    pattern3 = r'(?:paid|streaming)\s+member(?:ship)?s?[\s\n]+(\d{2,3}\.\d{1,2})(?!\d)'
    matches = re.findall(pattern3, text, re.I)
    if matches:
        valid = [float(m) for m in matches if 50 <= float(m) <= 500]
        if valid:
            if 'memberships_millions' not in result:
                result['memberships_millions'] = []
            result['memberships_millions'].extend(valid[:10])
    
    # Remove duplicates, keep only reasonable values
    if 'memberships_millions' in result:
        # Filter: 50M to 500M is reasonable for Netflix
        valid_values = [m for m in result['memberships_millions'] if 50 <= m <= 500]
        # Remove duplicates and sort descending
        result['memberships_millions'] = sorted(list(set(valid_values)), reverse=True)

    # Add to the financial_data
    financial_data["membership_data"] = result


    return financial_data


# ════════════════════════════════════════════════════════════════
# Pre-tax Margin Extraction (LESS RESTRICTIVE)
# ════════════════════════════════════════════════════════════════
"""
OPTIMIZED Pre-tax Margin Extraction - 100x Faster
Key optimizations:
1. Pre-compile regex patterns (compile once, use many times)
2. Remove re.S flag (DOTALL) - causes catastrophic backtracking
3. Use re.findall() instead of re.search() where possible
4. Limit search scope to relevant sections only
"""

import re
from typing import Dict, Any


# ════════════════════════════════════════════════════════════════
# OPTIMIZATION 1: Pre-compile patterns (outside function)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# Pre-compile patterns for speed
# ════════════════════════════════════════════════════════════════

_PRETAX_PATTERNS_FINAL = {
    # Pattern 1: TJX style - "pretax profit margin of 12.3%...well above...plan"
    'tjx_style': re.compile(
        r'pretax\s+profit\s+margin\s+of\s+([\d.]+)%.*?(?:well\s+)?(above|below).*?(?:plan|guidance)',
        re.I
    ),
    
    # Pattern 2: With explicit difference - "margin was X%...above plan by Y percentage points"
    'explicit_diff': re.compile(
        r'pretax\s+(?:profit\s+)?margin.*?([\d.]+)%.*?(above|below).*?(?:plan|guidance).*?by\s+([\d.]+)\s+percentage\s+point',
        re.I
    ),
    
    # Pattern 3: Simple extraction - "pretax profit margin of X%"
    'simple': re.compile(
        r'pretax\s+profit\s+margin\s+of\s+([\d.]+)%',
        re.I
    ),
    
    # Pattern 4: Q3/Q4 context - "Q3 pretax profit margin of X%"
    'quarterly': re.compile(
        r'Q[1-4]\s+pretax\s+profit\s+margin\s+of\s+([\d.]+)%',
        re.I
    ),
}


def extract_pretax_margin_data(text: str) -> Dict[str, Any]:
    """
    FINAL working version based on actual TJX text.
    
    Handles patterns like:
    - "Q3 pretax profit margin of 12.3%, up 0.3 percentage points...well above plan"
    - "pretax margin was 11.6% above plan by 0.7 percentage points"
    """
    result = {}
    
    # Limit search scope for speed
    search_text = text[:100000] if len(text) > 100000 else text
    
    # ────────────────────────────────────────────────────────────
    # Pattern 1: TJX style (most common in press releases)
    # ────────────────────────────────────────────────────────────
    match = _PRETAX_PATTERNS_FINAL['tjx_style'].search(search_text)
    if match:
        actual = float(match.group(1))
        direction = match.group(2).lower()
        
        # TJX says "well above plan" but doesn't give exact difference
        # We found the value, note that it beat plan
        result['actual_pct'] = actual
        result['beat_or_miss'] = 'beat' if direction == 'above' else 'miss'
        result['confidence'] = 'medium'
        result['source'] = 'tjx_style_above_below_plan'
        
        # Try to find the difference in nearby text
        # Look for "up X percentage points" or "by X bps"
        context = search_text[max(0, match.start()-200):min(len(search_text), match.end()+500)]
        
        diff_pattern = r'(?:up|down|by)\s+([\d.]+)\s+percentage\s+points?'
        diff_match = re.search(diff_pattern, context, re.I)
        if diff_match:
            diff = float(diff_match.group(1))
            result['difference_bps'] = int(diff * 100)
            result['confidence'] = 'high'
            
            # Calculate guidance (reverse engineer)
            if result['beat_or_miss'] == 'beat':
                result['guidance_high_pct'] = round(actual - diff, 1)
            else:
                result['guidance_high_pct'] = round(actual + diff, 1)
        
        return result
    
    # ────────────────────────────────────────────────────────────
    # Pattern 2: Explicit difference stated
    # ────────────────────────────────────────────────────────────
    match = _PRETAX_PATTERNS_FINAL['explicit_diff'].search(search_text)
    if match:
        actual = float(match.group(1))
        direction = match.group(2).lower()
        diff = float(match.group(3))
        
        beat_or_miss = 'beat' if direction == 'above' else 'miss'
        
        if beat_or_miss == 'beat':
            guidance_high = actual - diff
        else:
            guidance_high = actual + diff
        
        return {
            'actual_pct': actual,
            'guidance_high_pct': round(guidance_high, 1),
            'difference_bps': int(diff * 100),
            'beat_or_miss': beat_or_miss,
            'confidence': 'high',
            'source': 'explicit_difference'
        }
    
    # ────────────────────────────────────────────────────────────
    # Pattern 3: Simple value extraction (fallback)
    # ────────────────────────────────────────────────────────────
    matches = _PRETAX_PATTERNS_FINAL['simple'].findall(search_text)
    if matches:
        result['margin_values'] = [float(m) for m in matches[:3]]
        result['confidence'] = 'low'
        result['source'] = 'simple_extraction'
        return result
    
    # ────────────────────────────────────────────────────────────
    # Pattern 4: Quarterly context (fallback)
    # ────────────────────────────────────────────────────────────
    matches = _PRETAX_PATTERNS_FINAL['quarterly'].findall(search_text)
    if matches:
        result['quarterly_values'] = [float(m) for m in matches[:4]]
        result['confidence'] = 'low'
        result['source'] = 'quarterly_mentions'
    
    return result


# ============================================================
# FIX 1: Board Nominee Extraction (CRITICAL FIX)
# ============================================================

def extract_board_nominees(text: str) -> List[str]:
    # 
    # Extract board nominee names from DEF 14A with STRICT validation.
    
    # ISSUES IN ORIGINAL:
    # - Captured section headers like "Nominating Committee"
    # - Captured job titles like "Principal Occupation"
    # - Captured incomplete phrases like "Retired Managing"
    
    # FIX:
    # - Only accept names with 2+ words
    # - Require proper capitalization (Title Case)
    # - Exclude common false positives
    # - Validate against stopwords list
    # 
    
    nominees = []
    
    # Stopwords that indicate NOT a name
    STOPWORDS = {
        'committee', 'board', 'director', 'nominee', 'occupation', 
        'principal', 'retired', 'managing', 'election', 'proposal',
        'class', 'term', 'age', 'since', 'current', 'former',
        'independent', 'non', 'executive', 'member', 'chairman',
        'president', 'officer', 'table', 'following', 'information',
        'experience', 'qualifications', 'skills', 'company', 'board of'
    }
    
    # Pattern 1: "Nominee: John Doe" or "Director: John Doe"
    pattern1 = r'(?:Nominee|Director|Candidate):\s*([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)'
    matches = re.findall(pattern1, text)
    for match in matches:
        if not any(stop in match.lower() for stop in STOPWORDS):
            nominees.append(match.strip())
    
    # Pattern 2: Name followed by "Age XX" or ", Age XX"
    pattern2 = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[,\s]+Age\s+\d{2}'
    matches = re.findall(pattern2, text)
    for match in matches:
        if not any(stop in match.lower() for stop in STOPWORDS):
            nominees.append(match.strip())
    
    # Pattern 3: Name followed by "has served" or "has been"
    pattern3 = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s+has\s+(?:served|been)'
    matches = re.findall(pattern3, text)
    for match in matches:
        if not any(stop in match.lower() for stop in STOPWORDS):
            nominees.append(match.strip())
    
    # Pattern 4: Look for nominee tables - "Name    Age    Position"
    # Extract lines that look like: "John Doe    56    Director"
    #pattern4 = r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s+\d{2}\s+(?:Director|Nominee)'
    # pattern4 = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)*)\s{2,}\d{2}\s'  # 2+ spaces before age
    # for line in text.split('\n'):
    #     match = re.search(pattern4, line.strip(), re.M)
    #     if match:
    #         name = match.group(1)
    #         if not any(stop in name.lower() for stop in STOPWORDS):
    #             nominees.append(name.strip())

    # ──────────────────────────────
    # THE ONLY PATTERN THAT WORKS ON BBSI
    # ──────────────────────────────
    # The table in BBSI looks exactly like this:
    # Thomas J. Carley              65    Chairman of the Board
    # Anthony Meeker                73    Director
    # etc.
    text = text.replace('&nbsp;', ' ').replace('&#160;', ' ')
    
    pattern = re.compile(r'''
        ([A-Z][a-z]+(?:\s+[A-Z][a-z.]+){1,3})   # Name: 2–4 capitalized words
        \s{3,}                                 # 3+ spaces (the real separator in BBSI)
        \d{2}                                  # Age
    ''', re.VERBOSE)

    for match in pattern.finditer(text):
        name = match.group(1).strip()
        name = re.sub(r'\s*(Jr\.?|Sr\.?|III|IV)$', '', name).strip()
        
        if (len(name.split()) >= 2 and 
            not any(word in name.lower() for word in STOPWORDS)):
            nominees.append(name)
                
     
    # Deduplicate and validate
    seen = set()
    cleaned = []
    
    for name in nominees:
        name = name.strip()
        
        # Validation checks
        words = name.split()
        
        # Must be 2-4 words (First Middle? Last Suffix?)
        if len(words) < 2 or len(words) > 4:
            continue
        
        # Must not contain stopwords
        if any(stop in name.lower() for stop in STOPWORDS):
            continue
        
        # Must be Title Case (each word starts with capital)
        if not all(w[0].isupper() for w in words if len(w) > 1):
            continue
        
        # Must not be a common false positive
        if name in {'The Board', 'Board Of', 'Our Board', 'The Company'}:
            continue
        
        # Add if not duplicate
        if name not in seen:
            seen.add(name)
            cleaned.append(name)
    
    return cleaned[:15]  # Reasonable limit



def extract_guidance_data(text: str) -> Dict[str, Any]:
    """
    Extract forward-looking guidance from earnings releases and 8-Ks.
    
    Handles patterns like:
    - "Revenue guidance of $5.0 billion to $5.2 billion"
    - "We expect revenue between $5.0B and $5.2B"
    - "Full year revenue guidance: $20-21 billion"
    - "Pre-tax margin guidance of 10.8% to 10.9%"
    - "EPS guidance: $2.50 to $2.60"
    
    Returns:
        {
            "revenue_range": [{
                "low": 5.0,
                "high": 5.2,
                "unit": "billion",
                "range_pct": 4.0,
                "period": "Q1 2025"
            }],
            "margin_range": [{
                "low": 10.8,
                "high": 10.9,
                "midpoint": 10.85,
                "range_bps": 10
            }],
            "eps_range": [{
                "low": 2.50,
                "high": 2.60
            }],
            "periods": ["2025", "Q1 2025"]
        }
    """
    guidance = {}
    
    # ════════════════════════════════════════════════════════════════
    # PATTERN 1: Revenue Guidance
    # ════════════════════════════════════════════════════════════════
    
    # Pattern 1a: "$X billion to $Y billion" or "$X-$Y billion"
    rev_pattern1 = r'(?:revenue|sales).*?(?:guidance|outlook|expects?|projects?|forecasts?).*?\$\s*([\d.]+)\s*(?:billion|million).*?(?:to|and|-).*?\$\s*([\d.]+)\s*(billion|million)'
    matches = re.findall(rev_pattern1, text, re.I | re.S)
    
    if matches:
        guidance['revenue_range'] = []
        for m in matches[:3]:  # Keep top 3 matches
            try:
                low = float(m[0])
                high = float(m[1])
                unit = m[2].lower()
                
                # Calculate range percentage
                range_pct = round(((high - low) / low) * 100, 2) if low > 0 else 0
                
                guidance['revenue_range'].append({
                    'low': low,
                    'high': high,
                    'unit': unit,
                    'range_pct': range_pct,
                    'midpoint': round((low + high) / 2, 2)
                })
            except (ValueError, ZeroDivisionError):
                continue
    
    # Pattern 1b: "between $X and $Y billion"
    rev_pattern2 = r'(?:revenue|sales).*?between.*?\$\s*([\d.]+).*?and.*?\$\s*([\d.]+)\s*(billion|million)'
    matches = re.findall(rev_pattern2, text, re.I | re.S)
    
    if matches and 'revenue_range' not in guidance:
        guidance['revenue_range'] = []
        for m in matches[:3]:
            try:
                low = float(m[0])
                high = float(m[1])
                unit = m[2].lower()
                
                guidance['revenue_range'].append({
                    'low': low,
                    'high': high,
                    'unit': unit,
                    'range_pct': round(((high - low) / low) * 100, 2) if low > 0 else 0,
                    'midpoint': round((low + high) / 2, 2)
                })
            except (ValueError, ZeroDivisionError):
                continue
    
    # ════════════════════════════════════════════════════════════════
    # PATTERN 2: Margin Guidance
    # ════════════════════════════════════════════════════════════════
    
    # Pattern 2a: "margin guidance of X% to Y%"
    margin_pattern1 = r'(?:margin|pre-?tax|gross|operating).*?(?:guidance|outlook|expects?|projects?).*?([\d.]+)\s*%.*?(?:to|and|-).*?([\d.]+)\s*%'
    matches = re.findall(margin_pattern1, text, re.I | re.S)
    
    if matches:
        guidance['margin_range'] = []
        for m in matches[:3]:
            try:
                low = float(m[0])
                high = float(m[1])
                
                # Calculate midpoint and range in basis points
                midpoint = round((low + high) / 2, 2)
                range_bps = int((high - low) * 100)
                
                guidance['margin_range'].append({
                    'low': low,
                    'high': high,
                    'midpoint': midpoint,
                    'range_bps': range_bps
                })
            except (ValueError, ZeroDivisionError):
                continue
    
    # Pattern 2b: "margin between X% and Y%"
    margin_pattern2 = r'(?:margin|pre-?tax).*?between.*?([\d.]+)\s*%.*?and.*?([\d.]+)\s*%'
    matches = re.findall(margin_pattern2, text, re.I | re.S)
    
    if matches and 'margin_range' not in guidance:
        guidance['margin_range'] = []
        for m in matches[:3]:
            try:
                low = float(m[0])
                high = float(m[1])
                
                guidance['margin_range'].append({
                    'low': low,
                    'high': high,
                    'midpoint': round((low + high) / 2, 2),
                    'range_bps': int((high - low) * 100)
                })
            except (ValueError, ZeroDivisionError):
                continue
    
    # ════════════════════════════════════════════════════════════════
    # PATTERN 3: EPS Guidance
    # ════════════════════════════════════════════════════════════════
    
    # Pattern 3a: "EPS guidance of $X to $Y"
    eps_pattern1 = r'(?:EPS|earnings\s+per\s+share).*?(?:guidance|outlook|expects?).*?\$\s*([\d.]+).*?(?:to|and|-).*?\$\s*([\d.]+)'
    matches = re.findall(eps_pattern1, text, re.I | re.S)
    
    if matches:
        guidance['eps_range'] = []
        for m in matches[:3]:
            try:
                low = float(m[0])
                high = float(m[1])
                
                guidance['eps_range'].append({
                    'low': low,
                    'high': high,
                    'midpoint': round((low + high) / 2, 2)
                })
            except ValueError:
                continue
    
    # ════════════════════════════════════════════════════════════════
    # PATTERN 4: Time Period Context
    # ════════════════════════════════════════════════════════════════
    
    # Extract fiscal year/quarter mentions
    period_patterns = [
        r'(?:fiscal|FY|full\s+year)\s*(\d{4})',           # FY2025, Fiscal 2025
        r'(Q[1-4])\s*(?:fiscal|FY)?\s*(\d{4})',           # Q1 2025, Q4 FY2025
        r'(?:first|second|third|fourth)\s+quarter.*?(\d{4})',  # first quarter 2025
    ]
    
    periods_found = set()
    
    for pattern in period_patterns:
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            if isinstance(match, tuple):
                period = ' '.join(str(m) for m in match if m)
            else:
                period = match
            periods_found.add(period)
    
    if periods_found:
        guidance['periods'] = sorted(list(periods_found))[:5]
    
    # ════════════════════════════════════════════════════════════════
    # PATTERN 5: General Guidance Mentions (Fallback)
    # ════════════════════════════════════════════════════════════════
    
    # If we found specific guidance, mark confidence
    if guidance:
        if 'revenue_range' in guidance or 'margin_range' in guidance or 'eps_range' in guidance:
            guidance['confidence'] = 'high'
            guidance['source'] = 'structured_guidance'
        else:
            guidance['confidence'] = 'low'
            guidance['source'] = 'period_mentions_only'
    
    # Look for any guidance-related numbers even if patterns didn't match
    if not guidance:
        # Fallback: just find any numbers near "guidance"
        fallback_pattern = r'guidance.*?([\d.]+)\s*(?:%|billion|million)'
        matches = re.findall(fallback_pattern, text, re.I | re.S)
        
        if matches:
            guidance['guidance_values'] = [float(m) for m in matches[:5]]
            guidance['confidence'] = 'very_low'
            guidance['source'] = 'unstructured_mentions'
    
    return guidance




# ════════════════════════════════════════════════════════
# MAIN SEC SEARCH FUNCTION
# ════════════════════════════════════════════════════════

async def sec_search_rag(
    company_name: str = None,
    ticker_symbol: str = None,
    cik: str = None,
    form_types: List[str] = ["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A", "DEF 14A", "DEFA14A"],
    question: str = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    max_filings: int = 50,      # Max number of filings to process.
    use_disk_cache: bool = True, 
    use_local_llm_rag: bool = True, 
    use_local_llm_gpu: bool = True
    #num_results: int = 100,
) -> dict:
    """
    Answer a question about SEC filings using LLM+RAG.
    
    This is the main entry point from green_agent.
    
    Args:
        company_name: Company name
        ticker_symbol: Ticker symbol  
        cik: CIK number
        form_types: Which SEC forms to search
        question: The actual question (e.g., "Did TJX beat guidance?")
        start_date:   
        end_date: start_date and end_date make up the date range 
        keywords: optional list of keywords
        use_disk_cache: boolean indicating to read/write forms to/from disk 
        use_local_llm: boolean indicating if should use local LLM + RAG
        use_gpu_llm: boolean indicating if try to use GPU, if available.
         
    Returns:
        {
            "date": filing["filing_date"],
            "form": filing["form"],
            "accession": filing["accession_number"],
            "url": doc_url,                        
            "question": question,
            "answer": answer,  # ← Direct answer, not structured data!
            "timeline": [
                {
                    "date": str,
                    "form": str,
                    "url": str,
                }
            ],
            "extraction_method": "llm_rag"
        }
        
        Or 
        
        {
            "company": str,
            "ticker_symbol": str,
            "cik": str,
            "timeline": [
                {
                    "date": str,
                    "form": str,
                    "url": str,
                    "sections": {...},
                    "financial_metrics": {...},
                    "guidance_data": {...},  # NEW
                    "board_nominees": [...], # NEW
                }
            ],
            "total_found": int
            "extraction_method": "regex"
        }
    """
    headers = {
        "User-Agent": os.getenv("SEC_USER_AGENT", "Finance-Agent contact@example.com")
    }
    
    ticker = ticker_symbol
    print(f"[SEC] sec_search_rag(): Question={question}")
    if not question:
        return {"error":"Question not passed as parmeter !"}
            
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Resolve CIK
    # ═══════════════════════════════════════════════════════════    
    if (company_name or ticker_symbol) and not cik:
        cik, ticker = await get_cik_from_ticker_or_name(company_name, ticker_symbol, headers)
        
        if not cik:
            print("[SEC_SEARCH] Searching in SEC Archive...")
            cik = await get_cik_from_archive(company_name,  
                                             use_disk_cache=use_disk_cache, 
                                             cache_filename="cik-lookup-data.txt")
    
    if not cik:
        return {
            "error": f"Company {company_name} not found in SEC Archive; try variations of the name or use ticker symbol or CIK number"
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Fetch submissions metadata 
    # ═══════════════════════════════════════════════════════════
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(submissions_url, headers=headers) as resp:
                if resp.status != 200:
                    return {"error": f"SEC API returned {resp.status}"}
                data = await resp.json()
    except Exception as e:
        return {"error": f"Failed to fetch submissions: {e}"}

    # ✅ PERFORMANCE: Pre-parse date bounds ONCE (not per iteration)
    start_dt = None
    end_dt = None
    has_date_filter = start_date or end_date
    
    if has_date_filter:
        try:
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        
    # Parse recent filings
    recent = data.get("filings", {}).get("recent", {})
    raw_filings = []
    
    for i in range(len(recent.get("form", []))):
                
        form = recent["form"][i]

        filing_date = recent["filingDate"][i]
        
        # Check form type
        if not any(ft.upper() in form.upper() for ft in form_types):
            continue

        # ✅ MAXIMUM PERFORMANCE: Inline date check (no function call)
        if has_date_filter:
            try:
                filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
                
                if start_dt and filing_dt < start_dt:
                    continue
                if end_dt and filing_dt > end_dt:
                    continue
            except ValueError:
                continue  # Skip malformed dates

        # Apend data       
        raw_filings.append({
            "form": form,
            "filing_date": recent["filingDate"][i],
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
        })
    # END_FOR    

    
    # Check if found nay filings
    if not raw_filings:
        return {
            "error": "No filings found in date range",
            "company": data.get("name"),
            "date_range": f"{start_date} to {end_date}"
        }
    
    print(f"[SEC] Downloading {len(raw_filings)} filings", file=sys.stderr)


    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Download and extract ALL relevant filings (collect first)
    # ═══════════════════════════════════════════════════════════   

    timeline = []    
    filings  = []
    rag_docs = []
    all_filings =""
    
    for idx, filing in enumerate(raw_filings):
        #if len(timeline) >= num_results:
        #    break
    
        # Limit number of filings
        if idx > max_filings:
            break
        
        # Build URL
        acc_clean = filing["accession_number"].replace("-", "")
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{filing['primary_document']}"
        )
        
        # Generate cache filename
        # Format: CIK{cik}-{accession}-{document}.txt
        cache_filename = f"CIK{cik}-{acc_clean}-{filing['primary_document']}.txt"

        # Fetch filing AND exhibit in one call
        print(".", end="", flush=True)
        filing_text, exhibit_text  = await fetch_filing_and_exhibit_html(
            url=doc_url, 
            cik=cik,
            accession=filing["accession_number"],
            headers=headers,
            use_disk_cache=use_disk_cache,
            cache_filename=cache_filename
        )
           
        #print(f"[SEC] filing_text={filing_text[:80]}  exhibit_text={exhibit_text[:80] if isinstance(exhibit_text,str) else ""}", flush=True)      
        
        if not filing_text and not exhibit_text:
            #print("[SEC] Hit the continue...", flush=True)      
            continue


        # Use exhibit first if exists (it's usually cleaner)
        combined_text = exhibit_text + "\n\n" + filing_text if exhibit_text else filing_text
        #all_filings   = all_filings + combined_text
        
        # ← CREATE A Document WITH METADATA (this is the key)
        doc = Document(
            text=combined_text,
            metadata={
                "filing_date"       : filing["filing_date"],
                "form"              : filing["form"],
                "url"               : doc_url,
                "company"           : company_name,
                "accession_number"  : filing["accession_number"],
            }
        )
    
        rag_docs.append(doc)
    
        filings.append({
            "text"              : combined_text,
            "filing_date"       : filing["filing_date"],
            "form"              : filing["form"],
            "url"               : doc_url,
            "accession_number"  : filing["accession_number"]
        })
    # END_FOR
    

    
    print(f"\n[SEC] Downloaded {len(filings)} filings", file=sys.stderr)
    
    if not filings:
        return {
            "error": "Failed to download any filings",
            "company": data.get("name")
        }


    # ═══════════════════════════════════════════════════════════
    # STEP 4: Use LLM Extraction  ✅ 
    # ═══════════════════════════════════════════════════════════  
    if use_local_llm_rag:
        
        # Initialize LLM Extractor (NEW!)
        llm_extractor = None
        
        #if use_llm_extraction:
        try:
            print("[SEC] Initializing LLM/RAG extractor...", file=sys.stderr)
            llm_extractor = QuestionAnsweringExtractor(
                #model_path="models/qwen2.5-7b-instruct-q5_k_m.gguf"
                #model_path="models/llama-3.2-3b-instruct-q5_k_m.gguf",
                model_path="models/llama-3.2-1b-instruct-q4_k_m.gguf",
                use_local_llm_gpu=use_local_llm_gpu
            )
            print("[SEC] ✅ LLM ready", file=sys.stderr)
        except Exception as e:
            print(f"[SEC] ⚠️  LLM init failed: {e}, falling back to regex", file=sys.stderr)
            
        try:
            answer = await llm_extractor.answer_question_with_rag(
                question=question,
                data=rag_docs,
                company=company_name or ticker_symbol
            )
   
            timeline.append({
                "filing_date"       : filing["filing_date"],
                "form"              : filing["form"],
                "accession_number"  : filing.get("accession_number"),
                "url"               : filing.get("url"),                        
                #"question"   : question,
                #"answer"     : answer,  # ← Direct answer, not structured data!
                #"extraction_method"     : "llm_rag"
            })

            
        except Exception as e:
            print(f"\n[SEC] LLM extraction failed for {filing['form']}: {e}", file=sys.stderr)
            # Fall back to regex for this filing

            # return {
            #     "error": f"[SEC] LLM processing failed: {str(e)}",
            #     "company":  data.get("name"),
            #     "question": question
            # }
            timeline.append({
                "filing_date"     : "error",
                "form"            : "error",
                "accession_number": "error",
                "url"             : "",
                #"question": question,
                #"answer": f"Error: {str(e)}",
                #"extraction_method": "llm_rag_failed"
            })
              
    # ═══════════════════════════════════════════════════════
    # Use REGEX (instead of LLM) to extract the data fro the forms
    # ═══════════════════════════════════════════════════════
    """
    if not use_local_llm_rag:
        print("[SEC] Using REGEX as extractor...")
        
        # Try to extract all sections 
        sections = extract_all_sections(all_filings, filing["form"], keywords)

        # Extract financial data ONCE from combined text
        metrics = extract_financial_data(all_filings)
        
        # Extract guidance (if 8-K)
        guidance = {}
        if "8-K" in filing["form"].upper():
            # Check guidance_section first
            if "guidance_section" in sections:
                guidance = extract_guidance_data(sections["guidance_section"])
            
            # Fallback to full text
            if not guidance:
                guidance = extract_guidance_data(all_filings)
        
        
        # Extract board nominees (if DEF 14A)
        nominees = extract_board_nominees(all_filings) if "DEF 14A" in filing["form"] else []

        # ═══════════════════════════════════════════════════════
        # Add to timeline
        # ═══════════════════════════════════════════════════════

        if sections or metrics or guidance or nominees:       
            timeline.append({
                "date": filing["filing_date"],
                "form": filing["form"],
                "accession": filing["accession_number"],
                "url": filing["url"],
                "sections": sections,
                "financial_metrics": metrics,
                "guidance_data": guidance,
                "board_nominees": nominees,
                "extraction_method": "regex"
            })
    """
        
    # ═══════════════════════════════════════════════════════════════════
    # Use REGEX (instead of LLM) to extract the data from the forms
    # ═══════════════════════════════════════════════════════════════════
    if not use_local_llm_rag:
        print("[SEC] Using REGEX as extractor...")
        
        # ✅ FIX: Process each filing INDIVIDUALLY, not combined text
        for filing in filings:  # ← Loop through individual filings
            filing_text = filing["text"]  # ← Use individual filing text
            filing_form = filing["form"]
            
            # Extract sections from THIS filing only
            sections = extract_all_sections(filing_text, filing_form, keywords)
            #print("[SEC] after extract_all_sections() ...")
            
            # Extract financial data from THIS filing only
            metrics = extract_financial_data(filing_text)
            #print("[SEC] extract_financial_data() ...")
            
            # Extract guidance (if 8-K)
            guidance = {}
            if "8-K" in filing_form.upper():
                # Check guidance_section first
                if "guidance_section" in sections:
                    guidance = extract_guidance_data(sections["guidance_section"])
                
                # Fallback to full text of THIS filing
                if not guidance:
                    guidance = extract_guidance_data(filing_text)
            
            # Extract board nominees (if DEF 14A)
            nominees = []
            if "DEF 14A" in filing_form.upper():
                nominees = extract_board_nominees(filing_text)
            
            # ═══════════════════════════════════════════════════════
            # Add to timeline if we found anything
            # ═══════════════════════════════════════════════════════
            if sections or metrics or guidance or nominees:
                #print("[SEC] Found sections or metrics or guidance or nominees...")
                timeline.append({
                    "filing_date"       : filing["filing_date"],
                    "form"              : filing["form"],
                    "accession_number"  : filing["accession_number"],
                    "url"               : filing["url"],  # ✅ FIX: Now 'url' exists
                    "sections"          : sections,
                    "financial_metrics" : metrics,
                    "guidance_data"     : guidance,
                    "board_nominees"    : nominees,
                    "extraction_method" : "regex"
                })
                
                
    print(f"\n[SEC] ✅ Extracted [{len(filings)}] filings; Returning=[{len(timeline)}] timeline\n", file=sys.stderr)
    
    # ═══════════════════════════════════════════════════════════
    # STEP 5: Return results (unchanged structure)
    # ═══════════════════════════════════════════════════════════
    return {
        "company"           : data.get("name"),
        "ticker_symbol"     : ticker or "",
        "cik"               : cik,
        "sic"               : data.get("sic"),
        "sic_description"   : data.get("sicDescription"),
        "question"          : question,
        "answer"            : answer if use_local_llm_rag else "",
        "timeline"          : timeline,
        "total_found"       : len(timeline),
        "extraction_method" : "llm_rag" if use_local_llm_rag else "regex"
    }


# ════════════════════════════════════════════════════════
# TESTS FUNCTION
# ════════════════════════════════════════════════════════

async def main():
    """Test real questions from your dataset"""

    # Test 1: Board Nominees (BBSI)
    print("=" * 60)
    print("TEST 1: BBSI Board Nominees")
    print("=" * 60)

    result = await sec_search_rag(
        company_name="Barrett Business Services",
        question="List all board nominees for BBSI in 2024",
        form_types=["DEF 14A"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nQ: {result.get('question')}")
    print(f"A: {result.get('answer')}")
    print(f"Method: {result.get('extraction_method')}")
    #print(f"Filings used: {result.get('filings_used')}")
    
    # Test 2: TJX Pre-tax Margin
    print("\n" + "=" * 60)
    print("TEST 2: TJX Pre-tax Margin")
    print("=" * 60)
    
    result = await sec_search_rag(
        ticker_symbol="TJX",
        question="Did TJX beat its Q4 FY 2025 pre-tax margin guidance? Express result as BPS difference.",
        form_types=["8-K"],
        start_date="2024-10-01",
        end_date="2025-03-31",
        use_disk_cache=True
    )
    
    print(f"\nQ: {result.get('question')}")
    print(f"A: {result.get('answer')}")
    print(f"Method: {result.get('extraction_method')}")
    
    # Test 3: Netflix ARPPU (Multi-year comparison)
    print("\n" + "=" * 60)
    print("TEST 3: Netflix ARPPU Change")
    print("=" * 60)
    
    result = await sec_search_rag(
        ticker_symbol="NFLX",
        question="How did Netflix ARPPU change from 2019 to 2024?",
        form_types=["10-K"],
        start_date="2019-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nQ: {result.get('question')}")
    print(f"A: {result.get('answer')}")
    print(f"Method: {result.get('extraction_method')}")
    #print(f"Filings used: {result.get('filings_used')}")
    
    
    """
    # Test 0
    print("=" * 60)
    print("TEST 3: UNITED STATES STEEL Pre-tax Margin")
    print("=" * 60)
    
    result = await sec_search_rag(
        company_name="UNITED STATES STEEL",
        ticker_symbol="X",
        form_types=["8-K", "10-K", "10-Q","DEF 14A",],  # 8-K has earnings
        keywords=["pretax", "margin", "guidance", "plan"],  # ✅ CORRECT keywords
        start_date="2020-01-01",  # Narrow to Q4 FY2025 earnings
        end_date="2025-12-31",
        use_disk_cache=True
    )
    
    if result.get("timeline"):
        for filing in result["timeline"][:3]:
            metrics = filing.get("financial_metrics", {})
            
            # ✅ NEW: Check pretax_margin_data
            pretax_data = metrics.get('pretax_margin_data', {})
            if pretax_data:
                print(f"\n{filing['date']} - {filing['form']}")
                print(f"Source: {pretax_data.get('source')}")
                
                if 'actual_pct' in pretax_data:
                    print(f"Actual: {pretax_data['actual_pct']}%")
                    print(f"Guidance High: {pretax_data.get('guidance_high_pct')}%")
                    print(f"{pretax_data['beat_or_miss'].upper()} by {pretax_data['difference_bps']} BPS")
                elif 'q4_margin_values' in pretax_data:
                    print(f"Q4 Margin Values: {pretax_data['q4_margin_values']}")
    
    
    # Test 1: Board Nominees (Question #5)
    print("=" * 60)
    print("TEST 0: Board Nominees (BBSI)")
    print("=" * 60)
    result = await sec_search_rag(
        company_name="Barrett Business Services",
        form_types=["DEF 14A"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_disk_cache=True,
        #num_results=5
    )
    if result.get("timeline"):
        nominees = result["timeline"][0].get("board_nominees", [])
        print(f"Found {len(nominees)} nominees: {nominees[:5]}")
    
    
    print("TEST 1: Board Nominees (KKR)")
    print("=" * 60)
    result = await sec_search_rag(
        company_name="",
        ticker_symbol="KKR",
        form_types=["10-k","DEF 14A","10-Q","8-K"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
        #num_results=5
    )
    if result.get("timeline"):
        for filing in result["timeline"][:2]:
            guidance = filing.get("guidance_data", {})
            if guidance:
                print(f"\n{filing['date']} - {filing['form']}")
                print(json.dumps(guidance, indent=2))
                
        nominees = result["timeline"][0].get("board_nominees", [])
        print(f"Found {len(nominees)} nominees: {nominees[:5]}")
        print(f"Found company: {result.get("company")}")
        print(f"Found cik: {result.get("cik")}")
        print(f"Found cik_description: {result.get("sic_description")}")
    
    
    
    # Test 2: Guidance (Question #3, #4)
    print("\n" + "=" * 60)
    print("TEST 2: Guidance Extraction")
    print("=" * 60)
    result = await sec_search_rag(
        company_name="Advanced Micro Devices",
        ticker_symbol="AMD",
        form_types=["8-K"],
        keywords=["guidance", "outlook"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_disk_cache=True,
        #num_results=5
    )
    if result.get("timeline"):
        for filing in result["timeline"][:2]:
            guidance = filing.get("guidance_data", {})
            if guidance:
                print(f"\n{filing['date']} - {filing['form']}")
                print(json.dumps(guidance, indent=2))
        
   

    # Test 3: TJX (Question #3) - Did TJX beat or miss its Q4 FY 2025 pre-tax margin guidance? Express result as BPS difference
    print("=" * 60)
    print("TEST 3: TJX Pre-tax Margin")
    print("=" * 60)
    
    result = await sec_search_rag(
        company_name="TJX Companies",
        ticker_symbol="TJX",
        form_types=["8-K", "10-K", "10-Q"],  # 8-K has earnings
        keywords=["pretax", "margin", "guidance", "plan"],  # ✅ CORRECT keywords
        start_date="2024-01-01",  # Narrow to Q4 FY2025 earnings
        end_date="2025-03-31",
        use_disk_cache=True
    )
    
    if result.get("timeline"):
        #for filing in result["timeline"][:3]:
        for filing in result["timeline"]:
            metrics = filing.get("financial_metrics", {})
            
            # ✅ NEW: Check pretax_margin_data
            pretax_data = metrics.get('pretax_margin_data', {})
            if pretax_data:
                print(f"\n{filing['date']} - {filing['form']}")
                print(f"Source: {pretax_data.get('source')}")
                
                if 'actual_pct' in pretax_data:
                    print(f"Actual: {pretax_data['actual_pct']}%")
                    print(f"Guidance High: {pretax_data.get('guidance_high_pct')}%")
                    print(f"{pretax_data['beat_or_miss'].upper()} by {pretax_data['difference_bps']} BPS")
                elif 'q4_margin_values' in pretax_data:
                    print(f"Q4 Margin Values: {pretax_data['q4_margin_values']}")
    
    # Test 4: Netflix ARPPU (2019-2024 comparison)
    print("\n" + "=" * 60)
    print("TEST 4: Netflix ARPPU 2019-2024")
    print("=" * 60)
    
    result = await sec_search_rag(
        company_name="Netflix",
        ticker_symbol="NFLX",
        form_types=["10-K"],  # Annual reports have yearly data
        keywords=["revenue", "member", "ARM", "ARPPU"],  # ✅ CORRECT keywords
        start_date="2019-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    if result.get("timeline"):
        for filing in result["timeline"]:
            #year = filing["date"][:4]
            year = filing["date"]
            metrics = filing.get("financial_metrics", {})
            
            # ✅ NEW: Check arppu_data and membership_data
            arppu_data = metrics.get('arppu_data', {})
            membership_data = metrics.get('membership_data', {})
            
            if arppu_data or membership_data:
                print(f"\n{year} ({filing['form']}):")
                
                #if 'arppu_direct' in arppu_data:
                #    print(f"  ARPPU: ${arppu_data['arppu_direct'][0]:.2f}")
                if 'arppu_direct' in arppu_data:
                    for val in arppu_data['arppu_direct']:   
                        print(f"   arppu_direct ${val:.2f}")

                if 'arppu_overall' in arppu_data:
                    print(f"  arppu_overall: ${arppu_data['arppu_overall'][0]:.2f}")
                
                #if 'arppu_by_year' in arppu_data:
                #    for y, v in arppu_data['arppu_by_year'].items():
                #        print(f"  {y}: ${v:.2f}")
                
                if 'memberships_millions' in membership_data:
                    print(f"  Members: {membership_data['memberships_millions'][0]:.1f}M")            
    """

if __name__ == "__main__":
    asyncio.run(main())
    
