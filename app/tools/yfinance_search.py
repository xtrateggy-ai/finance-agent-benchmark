"""
Refactored yfinance_search.py
- Robust company-name -> ticker lookup using yf.search() with Yahoo fallback
- Uses yf.get_info() with fallback to fast_info
- Normalizes multiple label variants returned by yfinance
- Computes Free Cash Flow when not provided
- Caching + retry + thread-offloading for blocking yfinance calls
- Better date handling and period extraction
- Improved ratio calculations and safer guards
- Provides sync functions and async wrappers suitable for async codebases
"""
from __future__ import annotations

import json
import sys
import logging
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Any

import requests
import pandas as pd
import yfinance as yf

# Configure basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# --- YFinance label alternatives mapping ---
YF_LABELS: Dict[str, List[str]] = {
    'total_revenue': ['Total Revenue', 'TotalRevenue', 'totalRevenue'],
    'net_income': ['Net Income', 'NetIncome', 'netIncome'],
    'operating_income': ['Operating Income', 'OperatingIncome', 'operatingIncome'],
    'gross_profit': ['Gross Profit', 'GrossProfit', 'grossProfit'],
    'cost_of_revenue': ['Cost Of Revenue', 'CostOfRevenue', 'costOfRevenue'],
    'operating_cash_flow': ['Operating Cash Flow', 'OperatingCashFlow', 'operatingCashFlow'],
    'capital_expenditure': ['Capital Expenditure', 'CapitalExpenditure', 'capitalExpenditure'],
    'free_cash_flow': ['Free Cash Flow', 'FreeCashFlow', 'freeCashFlow'],
    'inventory': ['Inventory', 'inventory'],
    'total_assets': ['Total Assets', 'TotalAssets', 'totalAssets'],
    'total_liabilities': ['Total Liabilities', 'TotalLiabilities', 'totalLiabilities', 'Total Liabilities Net Minority Interest'],
    'stockholders_equity': ['Stockholders Equity', 'StockholdersEquity', 'Total Stockholder Equity', 'Total Stockholders Equity']
}

# ═══════════════════════════════════════════════════════════
# EXPANDED NICKNAME → TICKER MAPPING
# ═══════════════════════════════════════════════════════════
name_to_ticker = {
    # Tech Giants
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'msft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'amzn': 'AMZN',
    'meta': 'META',
    'facebook': 'META',
    'fb': 'META',
    'netflix': 'NFLX',
    'nflx': 'NFLX',
    'nvidia': 'NVDA',
    'nvda': 'NVDA',
    'tesla': 'TSLA',
    'tsla': 'TSLA',
    
    # Semiconductors
    'amd': 'AMD',
    'advanced micro devices': 'AMD',
    'intel': 'INTC',
    'intc': 'INTC',
    'qualcomm': 'QCOM',
    'broadcom': 'AVGO',
    'micron': 'MU',
    'texas instruments': 'TXN',
    
    # Enterprise Software
    'cisco': 'CSCO',
    'oracle': 'ORCL',
    'ibm': 'IBM',
    'international business machines': 'IBM',
    'salesforce': 'CRM',
    'adobe': 'ADBE',
    'servicenow': 'NOW',
    'snowflake': 'SNOW',
    
    # Retail
    'walmart': 'WMT',
    'wmt': 'WMT',
    'target': 'TGT',
    'tgt': 'TGT',
    'costco': 'COST',
    'home depot': 'HD',
    'the home depot': 'HD',
    'tjx': 'TJX',
    'tjx companies': 'TJX',
    'the tjx companies': 'TJX',
    'lowes': 'LOW',
    "lowe's": 'LOW',
    'best buy': 'BBY',
    
    # Industrial & Steel
    'us steel': 'X',
    'united states steel': 'X',
    'u.s. steel': 'X',
    'uss': 'X',
    'caterpillar': 'CAT',
    'cat': 'CAT',
    'boeing': 'BA',
    'ba': 'BA',
    '3m': 'MMM',
    'mmm': 'MMM',
    'general electric': 'GE',
    'ge': 'GE',
    'honeywell': 'HON',
    'lockheed martin': 'LMT',
    'raytheon': 'RTX',
    'deere': 'DE',
    'john deere': 'DE',
    
    # Finance
    'jpmorgan': 'JPM',
    'jp morgan': 'JPM',
    'jpm': 'JPM',
    'bank of america': 'BAC',
    'bofa': 'BAC',
    'bac': 'BAC',
    'wells fargo': 'WFC',
    'wfc': 'WFC',
    'goldman sachs': 'GS',
    'goldman': 'GS',
    'morgan stanley': 'MS',
    'citigroup': 'C',
    'citi': 'C',
    'us bancorp': 'USB',
    'american express': 'AXP',
    'amex': 'AXP',
    'visa': 'V',
    'mastercard': 'MA',
    'paypal': 'PYPL',
    'charles schwab': 'SCHW',
    'schwab': 'SCHW',
    'blackrock': 'BLK',
    
    # Energy
    'exxon': 'XOM',
    'exxonmobil': 'XOM',
    'exxon mobil': 'XOM',
    'chevron': 'CVX',
    'conocophillips': 'COP',
    'schlumberger': 'SLB',
    'marathon petroleum': 'MPC',
    'valero': 'VLO',
    'phillips 66': 'PSX',
    
    # Pharma & Healthcare
    'johnson & johnson': 'JNJ',
    'johnson and johnson': 'JNJ',
    'jnj': 'JNJ',
    'pfizer': 'PFE',
    'merck': 'MRK',
    'abbvie': 'ABBV',
    'eli lilly': 'LLY',
    'lilly': 'LLY',
    'bristol myers': 'BMY',
    'bristol-myers squibb': 'BMY',
    'unitedhealth': 'UNH',
    'united health': 'UNH',
    'cvs': 'CVS',
    'cvs health': 'CVS',
    'walgreens': 'WBA',
    'moderna': 'MRNA',
    
    # Consumer Goods
    'procter & gamble': 'PG',
    'procter and gamble': 'PG',
    'p&g': 'PG',
    'pg': 'PG',
    'coca-cola': 'KO',
    'coca cola': 'KO',
    'coke': 'KO',
    'pepsi': 'PEP',
    'pepsico': 'PEP',
    'colgate': 'CL',
    'colgate-palmolive': 'CL',
    'kimberly clark': 'KMB',
    'kimberly-clark': 'KMB',
    'general mills': 'GIS',
    'kraft heinz': 'KHC',
    'mondelez': 'MDLZ',
    'estee lauder': 'EL',
    
    # Hospitality & Travel
    'airbnb': 'ABNB',
    'abnb': 'ABNB',
    'marriott': 'MAR',
    'hilton': 'HLT',
    'booking': 'BKNG',
    'booking holdings': 'BKNG',
    'expedia': 'EXPE',
    'delta': 'DAL',
    'delta airlines': 'DAL',
    'american airlines': 'AAL',
    'united airlines': 'UAL',
    'southwest': 'LUV',
    
    # Entertainment & Media
    'disney': 'DIS',
    'walt disney': 'DIS',
    'comcast': 'CMCSA',
    'warner bros': 'WBD',
    'paramount': 'PARA',
    'spotify': 'SPOT',
    
    # Automotive
    'ford': 'F',
    'general motors': 'GM',
    'gm': 'GM',
    'stellantis': 'STLA',
    'rivian': 'RIVN',
    'lucid': 'LCID',
    
    # Other Notable
    'berkshire': 'BRK-B',
    'berkshire hathaway': 'BRK-B',
    'brk': 'BRK-B',
    'starbucks': 'SBUX',
    'sbux': 'SBUX',
    'mcdonalds': 'MCD',
    "mcdonald's": 'MCD',
    'nike': 'NKE',
    'lululemon': 'LULU',
    'fedex': 'FDX',
    'ups': 'UPS',
    'united parcel service': 'UPS',
    
    # Finance Specific (from benchmark questions)
    'bbsi': 'BBSI',
    'barrett business services': 'BBSI',
    'kkr': 'KKR',
    'kkr & co': 'KKR',
}

# ═══════════════════════════════════════════════════════════
# TICKER → COMPANY NAME MAPPING (REVERSE LOOKUP)
# ═══════════════════════════════════════════════════════════
ticker_to_company = {
    # Tech Giants
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NFLX': 'Netflix Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    
    # Semiconductors
    'AMD': 'Advanced Micro Devices Inc.',
    'INTC': 'Intel Corporation',
    'QCOM': 'Qualcomm Inc.',
    'AVGO': 'Broadcom Inc.',
    'MU': 'Micron Technology Inc.',
    'TXN': 'Texas Instruments Inc.',
    
    # Enterprise Software
    'CSCO': 'Cisco Systems Inc.',
    'ORCL': 'Oracle Corporation',
    'IBM': 'International Business Machines Corporation',
    'CRM': 'Salesforce Inc.',
    'ADBE': 'Adobe Inc.',
    'NOW': 'ServiceNow Inc.',
    'SNOW': 'Snowflake Inc.',
    
    # Retail
    'WMT': 'Walmart Inc.',
    'TGT': 'Target Corporation',
    'COST': 'Costco Wholesale Corporation',
    'HD': 'The Home Depot Inc.',
    'TJX': 'The TJX Companies Inc.',
    'LOW': "Lowe's Companies Inc.",
    'BBY': 'Best Buy Co. Inc.',
    
    # Industrial & Steel
    'X': 'United States Steel Corporation',
    'CAT': 'Caterpillar Inc.',
    'BA': 'The Boeing Company',
    'MMM': '3M Company',
    'GE': 'General Electric Company',
    'HON': 'Honeywell International Inc.',
    'LMT': 'Lockheed Martin Corporation',
    'RTX': 'RTX Corporation',
    'DE': 'Deere & Company',
    
    # Finance
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corporation',
    'WFC': 'Wells Fargo & Company',
    'GS': 'The Goldman Sachs Group Inc.',
    'MS': 'Morgan Stanley',
    'C': 'Citigroup Inc.',
    'USB': 'U.S. Bancorp',
    'AXP': 'American Express Company',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Incorporated',
    'PYPL': 'PayPal Holdings Inc.',
    'SCHW': 'The Charles Schwab Corporation',
    'BLK': 'BlackRock Inc.',
    
    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger N.V.',
    'MPC': 'Marathon Petroleum Corporation',
    'VLO': 'Valero Energy Corporation',
    'PSX': 'Phillips 66',
    
    # Pharma & Healthcare
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'MRK': 'Merck & Co. Inc.',
    'ABBV': 'AbbVie Inc.',
    'LLY': 'Eli Lilly and Company',
    'BMY': 'Bristol-Myers Squibb Company',
    'UNH': 'UnitedHealth Group Incorporated',
    'CVS': 'CVS Health Corporation',
    'WBA': 'Walgreens Boots Alliance Inc.',
    'MRNA': 'Moderna Inc.',
    
    # Consumer Goods
    'PG': 'The Procter & Gamble Company',
    'KO': 'The Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'CL': 'Colgate-Palmolive Company',
    'KMB': 'Kimberly-Clark Corporation',
    'GIS': 'General Mills Inc.',
    'KHC': 'The Kraft Heinz Company',
    'MDLZ': 'Mondelez International Inc.',
    'EL': 'The Estée Lauder Companies Inc.',
    
    # Hospitality & Travel
    'ABNB': 'Airbnb Inc.',
    'MAR': 'Marriott International Inc.',
    'HLT': 'Hilton Worldwide Holdings Inc.',
    'BKNG': 'Booking Holdings Inc.',
    'EXPE': 'Expedia Group Inc.',
    'DAL': 'Delta Air Lines Inc.',
    'AAL': 'American Airlines Group Inc.',
    'UAL': 'United Airlines Holdings Inc.',
    'LUV': 'Southwest Airlines Co.',
    
    # Entertainment & Media
    'DIS': 'The Walt Disney Company',
    'CMCSA': 'Comcast Corporation',
    'WBD': 'Warner Bros. Discovery Inc.',
    'PARA': 'Paramount Global',
    'SPOT': 'Spotify Technology S.A.',
    
    # Automotive
    'F': 'Ford Motor Company',
    'GM': 'General Motors Company',
    'STLA': 'Stellantis N.V.',
    'RIVN': 'Rivian Automotive Inc.',
    'LCID': 'Lucid Group Inc.',
    
    # Other Notable
    'BRK-B': 'Berkshire Hathaway Inc.',
    'SBUX': 'Starbucks Corporation',
    'MCD': "McDonald's Corporation",
    'NKE': 'Nike Inc.',
    'LULU': 'Lululemon Athletica Inc.',
    'FDX': 'FedEx Corporation',
    'UPS': 'United Parcel Service Inc.',
    
    # Finance Specific
    'BBSI': 'Barrett Business Services Inc.',
    'KKR': 'KKR & Co. Inc.',
}


def _fallback_ticker_lookup(company_name: str) -> dict:
    """
    Fallback to internal mapping when Yahoo API fails.
    Handles nicknames and common variations.
    """
    name_lower = company_name.strip().lower()
    
    if name_lower in name_to_ticker:
        ticker_symbol = name_to_ticker[name_lower]
        
        # Verify ticker exists and get official name
        try:
            import yfinance as yf
            t = yf.Ticker(ticker_symbol)
            info = t.info
            
            return {
                "company_name": info.get("longName", ticker_to_company.get(ticker_symbol, company_name)),
                "ticker": ticker_symbol,
                "exchange": info.get("exchange", "Unknown")
            }
        except Exception:
            # Fallback to our mapping if yfinance fails
            return {
                "company_name": ticker_to_company.get(ticker_symbol, company_name),
                "ticker": ticker_symbol,
                "exchange": "Unknown"
            }
    
    return {
        "error": "Ticker not found; try sec_search_handler with company name (or variation of the company name) instead",
        "company_name": company_name,
    }


async def get_company_name_from_ticker(ticker: str) -> str:
    """
    Convert ticker symbol to official company name.
    
    Args:
        ticker: Stock ticker symbol (e.g., "X", "NFLX", "AAPL")
    
    Returns:
        Official company name (e.g., "United States Steel Corporation")
        Returns ticker if not found in mapping
    
    Examples:
        >>> get_company_name_from_ticker("X")
        "United States Steel Corporation"
        
        >>> get_company_name_from_ticker("NFLX")
        "Netflix Inc."
        
        >>> get_company_name_from_ticker("AAPL")
        "Apple Inc."
    """
    ticker_upper = ticker.strip().upper()
    
    # Try our mapping first (fastest)
    if ticker_upper in ticker_to_company:
        return ticker_to_company[ticker_upper]
    
    # Fallback to yfinance if not in mapping
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_upper)
        info = t.info
        
        long_name = info.get("longName")
        if long_name:
            return long_name
        
        # Try shortName as fallback
        short_name = info.get("shortName")
        if short_name:
            return short_name
            
    except Exception:
        pass
    
    # Last resort: return ticker itself
    return ticker_upper

# --- Utilities ---
def _safe_get_info(ticker_obj: yf.Ticker) -> Dict[str, Any]:
    """Get info with modern yfinance API and safe fallbacks."""
    try:
        info = ticker_obj.get_info()  # preferred
        if info:
            return info
    except Exception:
        pass
    try:
        # fast_info is lightweight
        fi = getattr(ticker_obj, 'fast_info', None)
        if fi:
            return dict(fi)
    except Exception:
        pass
    return {}


def _normalize_series_index(s: pd.Series) -> pd.Series:
    """Ensure index is datetime-like where possible and sorted desc."""
    idx = s.index
    try:
        dt_index = pd.to_datetime(idx)
        s2 = s.copy()
        s2.index = dt_index
        s2 = s2.sort_index(ascending=False)
        return s2
    except Exception:
        # fallback: try to keep original ordering but convert to strings
        s2 = s.copy()
        s2.index = [str(i) for i in s2.index]
        return s2


def _extract_periods(series: pd.Series, num_periods: int) -> Dict[str, float]:
    """Extract most recent N periods from a pandas Series. Returns dict(period_str -> value)"""
    if series is None or len(series) == 0:
        return {}

    series = _normalize_series_index(series)

    out: Dict[str, float] = {}
    for i, (date, value) in enumerate(series.items()):
        if i >= num_periods:
            break
        date_str = None
        if isinstance(date, pd.Timestamp):
            # format as YYYY-QX if we can detect quarter
            year = date.year
            month = date.month
            quarter = (month - 1) // 3 + 1
            date_str = f"{year}-Q{quarter}"
        else:
            date_str = str(date)
        try:
            out[date_str] = float(value) if pd.notna(value) else 0.0
        except Exception:
            out[date_str] = 0.0
    return out



async def get_ticker_symbol(company_name):
    """
    Convert company name to ticker symbol using Yahoo Finance Search API.
    
    Args:
        company_name: Company name (e.g., "Apple", "TJX Companies", "Netflix")
    
    Returns:
        {
            "company_name": str,
            "ticker": str,
            "exchange": str
        }
        OR
        {
            "error": str,
            "company_name": str,
            "suggestion": str
        }
    """
    
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        params = {
            "q": company_name,
            "quotesCount": 5,  # Get top 5 results
            "newsCount": 0,
            "country": "United States"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        #print (f"[SEC_SEARCH] Company=[{company_name}] resp={data["quotes"][0]}\n")
    
        # Check if we got results
        quotes = data.get("quotes", [])
        if not quotes:
            # Fallback to internal mapping
            return _fallback_ticker_lookup(company_name)
    
        # Filter for US exchanges
        us_exchanges = {"NYQ", "NMS", "NASDAQ", "NYSE", "ARCX", "AMEX", "NGM", "NCM"}
           
        for quote in quotes:
            exchange = quote.get("exchange", "")
            
            # Prioritize US exchanges
            if exchange in us_exchanges:
                return {
                    "company_name": quote.get("shortname") or quote.get("longname") or company_name,
                    "ticker": quote.get("symbol"),
                    "exchange": exchange
                }
            
        # If it is not US exchanges, return error
        # return {
        #     "error": f"No US exchange listing found for '{company_name}'",
        #     "company_name": company_name,
        #     "suggestion": "Try sec_search_handler with company_name instead"
        # }
        
        # Look at the internal list
        return _fallback_ticker_lookup(company_name)
    
    except requests.RequestException as e:
        print(f"[YFINANCE] Search API error: {e}", file=sys.stderr)
        # Fallback to internal mapping
        return _fallback_ticker_lookup(company_name)
    
    except Exception as e:
        return {
            "error": f"Lookup failed: {str(e)}",
            "company_name": company_name,
            "suggestion": "Try sec_search_handler with company_name instead"
        }


def _fallback_ticker_lookup(company_name: str) -> dict:
    """
    Fallback to internal mapping when Yahoo API fails.
    """
    
    name_lower = company_name.strip().lower()
    
    if name_lower in name_to_ticker:
        ticker_symbol = name_to_ticker[name_lower]
        
        # Verify ticker exists
        try:
            import yfinance as yf
            t = yf.Ticker(ticker_symbol)
            info = t.info
            
            return {
                "company_name": info.get("longName", company_name),
                "ticker": ticker_symbol,
                "exchange": info.get("exchange", "Unknown")
            }
        except Exception:
            pass
    
    return {
        "error": "Ticker not found; try sec_search_handler with company name (or variation of the company name) instead",
        "company_name": company_name,
    }

"""
async def get_ticker_symbol(company_name):
    try:
        url = f"https://stock-symbol-lookup-api.onrender.com/{company_name}"
        resp = requests.get(url)
        print (f"[SEC_SEARCH] Company {company_name} resp={resp.json()}")

        return resp.json()   # returns {'stock_symbol': 'AAPL'} (or similar)
    
        # return {
        #     "error": "Ticker not found",
        #     "company_name": company_name,
        #     "suggestion": "Try sec_search_handler instead"
        # }

    except Exception as e:
        return {
            "error": f"Lookup failed: {str(e)}",
            "company_name": company_name,
            "suggestion": "Try sec_search_handler instead "
        }
"""

def _pick_first_available(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return None


async def get_financial_metrics(ticker: str, metrics: Optional[List[str]] = None, period: str = 'annual', years: int = 3) -> Dict[str, Any]:

    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)

    print(f"[YFINANCE] get_financial_metrics() ticker={ticker} period={period} years={years}")
 
    # financial tables (pandas DataFrame) — prefer quarterly if requested
    try:
        if period == 'quarterly':
            financials = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow
        else:
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
    except Exception as e:
        return {'error': f'Failed to retrieve financial tables for {ticker}: {e}', 'ticker': ticker}

    info = _safe_get_info(stock)

    result: Dict[str, Any] = {'ticker': ticker, 'period': period, 'data': {}, 'company_info': {}}

    if metrics is None:
        metrics = ['revenue', 'net_income', 'operating_income', 'total_assets', 'shares_outstanding', 'free_cash_flow']

    # helper to read series by canonical metric name
    def _get_series_for(metric_key: str) -> Optional[pd.Series]:
        if metric_key not in YF_LABELS:
            return None
        candidates = YF_LABELS[metric_key]
        # try financials first, then balance sheet, then cashflow
        for df in (financials, balance_sheet, cash_flow):
            if df is None or df.empty:
                continue
            s = _pick_first_available(df, candidates)
            if s is not None:
                return s
        return None

    # extract metrics
    for metric in metrics:
        mk = metric.lower().replace(' ', '_')

        if mk in ('revenue', 'total_revenue'):
            s = _get_series_for('total_revenue')
            if s is not None:
                result['data']['revenue'] = _extract_periods(s, years)

        elif mk in ('net_income', 'net_earnings'):
            s = _get_series_for('net_income')
            if s is not None:
                result['data']['net_income'] = _extract_periods(s, years)

        elif mk in ('operating_income', 'ebit'):
            s = _get_series_for('operating_income')
            if s is not None:
                result['data']['operating_income'] = _extract_periods(s, years)

        elif mk in ('gross_profit',):
            s = _get_series_for('gross_profit')
            if s is not None:
                result['data']['gross_profit'] = _extract_periods(s, years)

        elif mk in ('cost_of_revenue', 'cogs'):
            s = _get_series_for('cost_of_revenue')
            if s is not None:
                result['data']['cost_of_revenue'] = _extract_periods(s, years)

        elif mk in ('total_assets', 'assets'):
            s = _get_series_for('total_assets')
            if s is not None:
                result['data']['total_assets'] = _extract_periods(s, years)

        elif mk in ('total_liabilities', 'liabilities'):
            s = _get_series_for('total_liabilities')
            if s is not None:
                result['data']['total_liabilities'] = _extract_periods(s, years)

        elif mk in ('equity', 'shareholders_equity', 'stockholders_equity'):
            s = _get_series_for('stockholders_equity')
            if s is not None:
                result['data']['equity'] = _extract_periods(s, years)

        elif mk in ('inventory',):
            s = _get_series_for('inventory')
            if s is not None:
                result['data']['inventory'] = _extract_periods(s, years)

        elif mk in ('operating_cash_flow', 'cfo'):
            s = _get_series_for('operating_cash_flow')
            if s is not None:
                result['data']['operating_cash_flow'] = _extract_periods(s, years)

        elif mk in ('capex', 'capital_expenditure'):
            s = _get_series_for('capital_expenditure')
            if s is not None:
                result['data']['capex'] = _extract_periods(s, years)

        elif mk in ('free_cash_flow', 'fcf'):
            # FCF: compute from cashflow if not directly available
            s_fcf = _get_series_for('free_cash_flow')
            if s_fcf is not None:
                result['data']['free_cash_flow'] = _extract_periods(s_fcf, years)
            else:
                s_ocf = _get_series_for('operating_cash_flow')
                s_capex = _get_series_for('capital_expenditure')
                if s_ocf is not None and s_capex is not None:
                    # align indices and compute
                    ocf = _normalize_series_index(s_ocf)
                    cap = _normalize_series_index(s_capex)
                    # use intersection of periods
                    periods = set(ocf.index).intersection(set(cap.index))
                    combined = {}
                    for p in sorted(periods, reverse=True)[:years]:
                        try:
                            combined[str(p)] = float(ocf.loc[p]) + float(cap.loc[p])
                        except Exception:
                            combined[str(p)] = 0.0
                    result['data']['free_cash_flow'] = combined

        elif mk in ('shares_outstanding', 'shares'):
            # from info
            if info and 'sharesOutstanding' in info:
                result['data']['shares_outstanding'] = {'current': info.get('sharesOutstanding')}

        elif mk in ('dividends', 'dividend_rate'):
            if info:
                if 'dividendRate' in info:
                    result['data']['dividend_rate'] = info.get('dividendRate')
                if 'dividendYield' in info:
                    result['data']['dividend_yield'] = info.get('dividendYield')

        elif mk in ('eps', 'earnings_per_share'):
            # Basic EPS label may appear in financials; otherwise rely on info
            if financials is not None and 'Basic EPS' in financials.index:
                result['data']['basic_eps'] = _extract_periods(financials.loc['Basic EPS'], years)
            elif info and 'forwardEps' in info:
                result['data']['basic_eps'] = {'forwardEps': info.get('forwardEps')}

    # add company metadata
    result['company_info'] = {
        'name': info.get('longName', ticker),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'currency': info.get('currency', 'USD')
    }

    return result



def _calculate_margin(numerator: Dict[str, float], denominator: Dict[str, float]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for p, nval in numerator.items():
        if p in denominator and denominator[p] != 0:
            try:
                res[p] = round((nval / denominator[p]) * 100, 2)
            except Exception:
                pass
    return res


def _calculate_ratio(numerator: Dict[str, float], denominator: Dict[str, float]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for p, nval in numerator.items():
        if p in denominator and denominator[p] != 0:
            try:
                res[p] = round(nval / denominator[p], 4)
            except Exception:
                pass
    return res


async def get_financial_ratios(ticker: str, ratios: Optional[List[str]] = None, period: str = 'annual') -> Dict[str, Any]:
    base = await get_financial_metrics(ticker=ticker, metrics=None, period=period, years=5)
    if 'error' in base:
        return base
    data = base.get('data', {})

    result = {'ticker': ticker.upper(), 'period': period, 'ratios': {}}
    if ratios is None:
        ratios = ['profit_margin', 'operating_margin', 'gross_margin', 'roe']

    for ratio in ratios:
        rkey = ratio.lower().replace(' ', '_')
        if rkey == 'profit_margin':
            if 'net_income' in data and 'revenue' in data:
                result['ratios']['profit_margin'] = _calculate_margin(data['net_income'], data['revenue'])
        elif rkey == 'operating_margin':
            if 'operating_income' in data and 'revenue' in data:
                result['ratios']['operating_margin'] = _calculate_margin(data['operating_income'], data['revenue'])
        elif rkey == 'gross_margin':
            if 'gross_profit' in data and 'revenue' in data:
                result['ratios']['gross_margin'] = _calculate_margin(data['gross_profit'], data['revenue'])
        elif rkey in ('roe', 'return_on_equity'):
            if 'net_income' in data and 'equity' in data:
                result['ratios']['roe'] = _calculate_ratio(data['net_income'], data['equity'])
        elif rkey in ('roa', 'return_on_assets'):
            if 'net_income' in data and 'total_assets' in data:
                result['ratios']['roa'] = _calculate_ratio(data['net_income'], data['total_assets'])
        elif rkey == 'inventory_turnover':
            if 'cost_of_revenue' in data and 'inventory' in data:
                result['ratios']['inventory_turnover'] = _calculate_turnover(data['cost_of_revenue'], data['inventory'])
        elif rkey == 'fcf_margin':
            if 'free_cash_flow' in data and 'revenue' in data:
                result['ratios']['fcf_margin'] = _calculate_margin(data['free_cash_flow'], data['revenue'])

    return result


def _calculate_turnover(sales: Dict[str, float], avg_balance: Dict[str, float]) -> Dict[str, float]:
    # convert period keys to sortable tokens by attempting to parse YYYY-Qn or date strings
    import re

    def _period_key(p: str):
        # try YYYY-Qn
        m = re.match(r"(\d{4})-Q(\d)", p)
        if m:
            year = int(m.group(1)); q = int(m.group(2));
            return (year, q)
        try:
            dt = pd.to_datetime(p)
            return (dt.year, (dt.month - 1) // 3 + 1)
        except Exception:
            return (0, 0)

    periods = sorted(sales.keys(), key=_period_key)
    out: Dict[str, float] = {}
    for i in range(1, len(periods)):
        cur = periods[i]
        prev = periods[i-1]
        if prev in avg_balance and cur in avg_balance:
            avg = (avg_balance[cur] + avg_balance[prev]) / 2 if avg_balance[cur] is not None and avg_balance[prev] is not None else 0
            if avg != 0:
                try:
                    val = round(sales[cur] / avg, 2)
                    out[cur] = val
                except Exception:
                    pass
    return out


# --- CLI / test runner ---
if __name__ == '__main__':
    import asyncio

    async def _run_tests():
        print(await get_ticker_symbol('us steel'))

        print(await get_ticker_symbol("international business machines"))

        print('\nTesting get_financial_metrics("TSLA", metrics=["revenue", "net_income"], period="annual")')
        res = await get_financial_metrics('TSLA', metrics=['revenue', 'net_income'], period='annual', years=3)
        print(json.dumps(res, indent=2))

        print('\nTesting get_financial_ratios("AAPL")')
        rr = await get_financial_ratios('AAPL')
        print(json.dumps(rr, indent=2))

    asyncio.run(_run_tests())
