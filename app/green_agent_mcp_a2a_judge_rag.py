# app/green_agent_mcp_a2a.py
"""
Finance-Green Agent for AgentBeats
A2A + MCP server with proper /reset support
"""

import asyncio
import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm as _tqdm
import uvicorn
import litellm
import httpx
import tomllib
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastmcp import FastMCP


# Import tools
#from tools.google_search import google_search
#from tools.serp_search import serp_search
#from tools.edgar_search import edgar_search
from tools.sec_search_rag import sec_search_rag # Using LLM and RAG
from tools.parse_html import parse_html
from tools.today_date import get_today_date
from tools.retrieve_information import retrieve_information
from tools.yfinance_search import get_financial_metrics, get_financial_ratios, get_ticker_symbol, get_company_name_from_ticker

from utils.llm_judge import LLMJudge
from utils.llm_manager import safe_llm_call
from utils.env_setup import init_environment

init_environment()


class GreenAgent:
    """
    Finance-Green Agent with stateful dataset management.
    Supports /reset to restart dataset reading from beginning.
    """

    def __init__(self):
        self.agent_host = os.getenv("GREEN_AGENT_HOST", "0.0.0.0")
        self.agent_port = int(os.getenv("GREEN_AGENT_PORT", 9000))
        self.mcp_port = int(os.getenv("MCP_PORT", 9001))
        self.name = "finance-green-agent"
        self.verbose = bool(int(os.getenv("VERBOSE", 1))) # 1=True 0=False 
        
        # Load agent card
        self.card_path = os.getenv(
            "GREEN_CARD", 
            Path(__file__).parent / "cards" / "green_card.toml"
        )
        if not os.path.exists(self.card_path):
            raise FileNotFoundError(f"Agent card {self.card_path} not found")
        
        with open(self.card_path, "rb") as f:
            self.agent_card = tomllib.load(f)

        # Environment and model setup
        self.safety_check = bool(int(os.getenv("SAFETY_CHECK", 0))) # 1=True 0=False
        self.safety_model = os.getenv("LLM_SAFETY", "gemini/gemini-2.5-flash-lite")

        self.llm_model         = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key       = os.getenv("LLM_API_KEY")
        self.use_local_llm_rag = bool(int(os.getenv("USE_LOCAL_LLM_RAG", 1))) # 1=True 0=False
        self.use_local_llm_gpu = bool(int(os.getenv("USE_LOCAL_LLM_GPU", 1))) # 1=True 0=False

        self.dataset_path = os.getenv("DATASET", "data/public.csv")
        self.use_disk_cache = bool(int(os.getenv("USE_DISK_CACHE", 1)))  # 1=True 0=False - Save SEC filings to disk.
        
        # === STATE MANAGEMENT ===
        self.dataset_df = None  # Will be loaded on first use
        self.current_task_index = 0  # Track which task we're on
        self.data_storage = {}  # For parse_html_page + retrieve_info
        self.assessment_history = []  # Track assessment results
        self.max_filings        = int(os.getenv("MAX_FILINGS_PER_QUESTION", 50))   # Max number of filings to process per question.
        
        
        # Initialize LLM Judge for answer evaluation
        self.judge = LLMJudge(model=self.llm_model, api_key=self.llm_api_key)
        
        # Create FastAPI app for A2A
        self.app = FastAPI(title="Finance Green Agent")
        
        # Create FastMCP server for tools
        self.mcp_server = FastMCP("finance-tools")
        
        # Setup routes and tools
        self._setup_a2a_routes()
        self._register_mcp_tools()

    def _load_dataset(self):
        """Load or reload dataset from CSV"""
        try:
            self.dataset_df = pd.read_csv(self.dataset_path)
            self.dataset_df.columns = self.dataset_df.columns.str.lower()
            print(f"[GREEN] Loaded dataset: {len(self.dataset_df)} questions")
        except Exception as e:
            print(f"[GREEN] Error loading dataset: {e}")
            self.dataset_df = None

    def reset_state(self):
        """Reset all agent state - called by /reset endpoint"""
        print(f"[GREEN] Resetting agent state...")
        
        # Reset dataset reading position
        self.current_task_index = 0
        
        # Reload dataset from file (start fresh)
        self._load_dataset()
        
        # Clear data storage (parsed HTML, etc.)
        self.data_storage.clear()
        
        # Clear assessment history
        self.assessment_history.clear()
        
        print(f"[GREEN] State reset complete")
        return {
            "status": "reset_complete",
            "dataset_loaded": self.dataset_df is not None,
            "dataset_size": len(self.dataset_df) if self.dataset_df is not None else 0,
            "current_task_index": self.current_task_index
        }

    def _setup_a2a_routes(self):
        """Setup A2A protocol endpoints"""
        
        @self.app.get("/card")
        @self.app.get("/.well-known/agent-card.json")
        async def get_card():
            """Return agent card"""
            return JSONResponse(self.agent_card)

        @self.app.post("/reset")
        async def reset_agent():
            """
            Reset agent state (AgentBeats requirement)
            Clears all state and restarts dataset from beginning
            """
            print(f"[GREEN] Resetting agent state...", file=sys.stderr)

            result = self.reset_state()
            return JSONResponse(result)

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "ok",
                "agent": self.name,
                "dataset_loaded": self.dataset_df is not None,
                "current_task_index": self.current_task_index
            }

        @self.app.post("/a2a")
        async def handle_a2a_message(request: Request):
            """Handle A2A messages from AgentBeats"""
            try:
                payload = await request.json()
                
                if self.verbose:
                    print(f"[GREEN] Received A2A message: {payload}",file=sys.stderr)
                
                method = payload.get("method")
                args = payload.get("args", {})
                
                if method == "run_assessment":
                    result = await self.run_assessment(
                        white_agent_address=args.get("white_address"),
                        config=args
                    )
                    return JSONResponse({
                        "status": "completed",
                        "result": result
                    })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": f"Unknown method: {method}"
                    }, status_code=400)
                    
            except Exception as e:
                print(f"[GREEN] Error handling A2A message: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "status": "error",
                    "message": str(e)
                }, status_code=500)

    def _register_mcp_tools(self):
        """Register tools with FastMCP server"""

        @self.mcp_server.tool()
        async def sec_search_handler(
            company_name: str = None,
            ticker_symbol: str = None,
            cik: str = None,
            question: str = None,
            start_date: str = None, 
            end_date: str = None,
            keywords: List[str] = None,  
            #num_results: int = 100,  # - num_results (int): Max filings to analyze (default: 100)
        ) -> dict:
            """
            Search SEC filings (10-K, 10-Q, 8-K, DEF-14A) for company financial data and events.
            
            ═══════════════════════════════════════════════════════════════
            WHEN TO USE THIS TOOL:
            ═══════════════════════════════════════════════════════════════
            ✓ Questions about mergers, acquisitions, divestitures
            ✓ Questions about specific SEC filings or annual reports
            ✓ Questions about risk factors, business operations, MD&A
            ✓ Questions requiring official/audited financial data
            ✓ Questions about corporate events (CEO changes, lawsuits, etc.)
            ✓ Multi-year trend questions (revenue growth over 5 years)
            
            For QUICK financial metrics (revenue, assets, ratios), prefer:
            → get_financial_metrics (faster, uses Yahoo Finance)
            → get_financial_ratios (for margins, ROE, etc.)
            
            
            ═══════════════════════════════════════════════════════════════════════
            PARAMETER PRIORITY (Use in this order for best results):
            ═══════════════════════════════════════════════════════════════════════
            
            1. ✅ BEST: ticker_symbol (if known) - Fastest, most reliable
               Example: ticker_symbol="NFLX"
            
            2. ⚠️ FALLBACK: company_name (if ticker unknown) - Slower, may fail
               Example: company_name="Netflix"
            
            3. ⚠️ ADVANCED: cik (rarely needed) - For specific CIK lookups
               Example: cik="0001065280"
            
            ⚠️ IMPORTANT: If you called get_ticker_symbol_handler and got a ticker,
            ALWAYS use ticker_symbol parameter instead of company_name!
            
            WRONG:
            {
              "company_name": "Barrett Business Services, Inc.",  # ❌ Slow
              "ticker_symbol": None  # ❌ You had the ticker but didn't use it!
            }
            
            RIGHT:
            {
              "company_name": "Barrett Business Services",  # Optional (for context)
              "ticker_symbol": "BBSI",  # ✅ Use the ticker you looked up!
            }
            
            ═══════════════════════════════════════════════════════════════════════
            PARAMETERS:
            ═══════════════════════════════════════════════════════════════════════
            
            - company_name (str, OPTIONAL): Use common name OR official name
                Examples: "Apple", "Netflix", "US Steel", "United States Steel"
                
            - ticker_symbol (str, OPTIONAL but PREFERRED): Stock ticker
                Examples: "AAPL", "NFLX", "X", "BBSI"
                ⚠️ Use this if you got ticker from get_ticker_symbol_handler!

            - question (str, REQUIRED): Question for the LLM to provide an answer.
                
            - start_date (str, REQUIRED): Format "YYYY-MM-DD"
                ⚠️ USE WIDE DATE RANGES for better results:
                - For recent events: last 2 years (e.g., "2023-01-01")
                - For mergers/acquisitions: 5+ years (e.g., "2020-01-01")
                - For historical trends: 10 years (e.g., "2015-01-01")
                
            - end_date (str, REQUIRED): Format "YYYY-MM-DD"
                Usually today or recent: "2025-12-31"
                ⚠️ Use get_today_date_handler to get current date!
                
            - keywords (List[str], OPTIONAL): Single words only, NOT phrases
                Good: ["merger", "acquisition", "Nippon"]
                Bad: ["merger with Nippon Steel"]  ← Won't work!
                
            
            
            ═══════════════════════════════════════════════════════════════════════
            EXAMPLE USAGE:
            ═══════════════════════════════════════════════════════════════════════
            
            Scenario: "What board members were nominated in 2024 for BBSI?"
            
            Step 1: Get ticker (if not known)
            → get_ticker_symbol_handler(company_name="BBSI")
            → Returns: {"ticker": "BBSI", "company_name": "Barrett Business Services, Inc."}
            
            Step 2: Search SEC filings with TICKER
            → sec_search_handler(
                ticker_symbol="BBSI",  # ✅ Use the ticker!
                start_date="2024-01-01",
                end_date="2024-12-31",
                keywords=["board", "director", "nomination"]
            
            ═══════════════════════════════════════════════════════════════
            RETURNS:
            ═══════════════════════════════════════════════════════════════
            {
                    "date": filing["filing_date"],
                    "form": filing["form"],
                    "accession": filing["accession_number"],
                    "url": doc_url,                        
                    "question": question,
                    "answer": answer,  # ← Direct answer, not structured data!
                    "method": "llm_rag_extraction"
            }
                OR
                
            {
              "company": str,               // canonical company name
              "ticker_symbol": str,         // company's ticker symbol
              "cik": str,                   // company’s CIK
              "sic": str,                   // company's sic
              "sic_description": str,       / company's sic desciption 
              "timeline": [
                {
                "date": str,
                "form": str,
                "accession": str,
                "url": str,
                "financial_metrics": [
                    {
                        'total_revenue': ["12,345"],      ← Values in MILLIONS USD
                        'total_assets': ["20,451"],
                        'total_liabilities': str',
                        'stockholders_equity': str",
                        'net_income': ["500", "-125"],    ← Negative = loss
                        'operating_cash_flow': ["2,100"]
                    }
                ],
                "sections": [
                    # 10-K 
                    {
                    'business': "Company description text...",
                    'risk_factors': "Risk factors text...",
                    'mda': "Management discussion text...",
                    'quantitative_qualitative': str,
                    'financial_statements_item': str,
                    'balance_sheet': str,
                    'income_statement': str',
                    'cash_flow': str',
                    'stockholders_equity': str',
                    'financial_notes': str',
                    'controls': str',
                    'paid_memberships': str,
                    'streaming_members': str,
                    'average_memberships': str,
                    'arppu': str,
                    'gross_margin_pct': str,
                    'operating_margin_pct': str,
                    'pretax_margin_pct': str,
                    'arppu_calculated': str
                    },
                
                    #8-K
                    {
                    'item_1_01': str,
                    'item_2_01': str,
                    'item_2_02': str,
                    'item_5_02': str,
                    'item_8_01': str,
                    'ma_activity': str,
                    }
 
                ],
                "guidance_data":  List,
                "board_nominees": List,
                "sections_found": list(sections.keys()),
                "metrics_found":  list(financial_metrics.keys())    
                },
              "total_found": int,
            }
 
              OR
            
            {  "error": str,   // Description of the error, if error found.
               "company": str  // Company name
            }  

            ═══════════════════════════════════════════════════════════════
            HOW TO EXTRACT ANSWERS:
            ═══════════════════════════════════════════════════════════════
            
            For NUMERIC answers (revenue, assets, etc.):
            → Look in: timeline[0]["financial_metrics"]["total_revenue"][0]
            → Values are in MILLIONS (e.g., "20,451" = $20.451 billion)
            → Negative shown as "-125" (indicates a loss)
            
            For TEXT answers (mergers, events, operations):
            → Look in: timeline[0]["sections"]["ma_activity"] for M&A
            → Look in: timeline[0]["sections"]["business"] for operations
            → Look in: timeline[0]["sections"]["risk_factors"] for risks
            
            ═══════════════════════════════════════════════════════════════
            EXAMPLES:
            ═══════════════════════════════════════════════════════════════
            
            Q: "How has US Steel addressed its merger with Nippon Steel?"
            → sec_search_handler(
                  company_name="US Steel",
                  ticker_symbol="X",
                  question="How has US Steel addressed its merger with Nippon Steel?",
                  start_date="2020-01-01",     ← Wide range for M&A history
                  end_date="2025-12-31",
                  keywords=["merger", "Nippon", "acquisition", "transaction"]
              )
            → Answer in: sections["ma_activity"] or sections["business"]
            
            Q: "What was Netflix revenue in 2023?"
            → sec_search_handler(
                  company_name="Netflix",
                  ticker_symbol="NFLX",
                  question="What was Netflix revenue in 2023?",
                  start_date="2023-01-01",
                  end_date="2023-12-31",
                  keywords=["revenue"]
              )
            → Answer in: financial_metrics["total_revenue"][0]
            
            Q: "How has Apple's revenue changed from 2019 to 2024?"
            → sec_search_handler(
                  company_name="Apple",
                  ticker_symbol="AAPL",
                  question="How has Apple's revenue changed from 2019 to 2024?",
                  start_date="2019-01-01",
                  end_date="2024-12-31",
                  keywords=["revenue"]
              )
            → Compare: financial_metrics["total_revenue"] across timeline entries


            """
            try:
                #from tools.sec_search import sec_search
                
                wcompany_name = company_name
                #print(f"[GREEN] Fetching {form_types} for {company_name} start={start_date} end={end_date}")
                print(f"[GREEN] Fetching {wcompany_name} ticker={ticker_symbol} start={start_date} end={end_date}")
                
                # Look for ticker symbol, if none was passed as parameter
                if ticker_symbol == None and wcompany_name != None:
                    result = await get_ticker_symbol_handler(wcompany_name)
                    wticker = result.get("ticker")
                    if len(wticker != 0):
                        ticker_symbol = wticker
                        print(f"[GREEN] Found ticker={wticker} for companu={company_name}")
                else:
                    if ticker_symbol != None and wcompany_name != None:
                        # Adjust company name, if required (e.g. change US Stell to United States Steel)
                        wcompany_name = await get_company_name_from_ticker(ticker_symbol)
                        print(f"[GREEN] Company name received ={company_name} found={wcompany_name}")
                        
                        # Company not found, so stick to what was received as parameter.
                        if not wcompany_name:
                            wcompany_name = company_name 
                        
                try: 
                    # Search for filings
                    search_result = await asyncio.wait_for( sec_search_rag(
                        company_name      = wcompany_name, 
                        ticker_symbol     = ticker_symbol,
                        cik               = cik,
                        form_types        = ["10-K", "10-K/A", "10-KT", "10-KT/A","10-Q", "10-Q/A","8-K", "8-K/A","DEF 14A","DEFA14A" ],
                        question          = question,
                        start_date        = start_date,
                        end_date          = end_date,
                        keywords          = keywords,
                        max_filings       = self.max_filings,
                        use_disk_cache    = self.use_disk_cache,
                        use_local_llm_rag = self.use_local_llm_rag,
                        use_local_llm_gpu = self.use_local_llm_gpu
                        #num_results=num_results
                    ),
                    timeout=300.0 #5 minutes
                    )
                except asyncio.TimeoutError:
                    return {
                        "error": "SEC search timed out after 5 minutes",
                        "company": company_name
                    }
                                
                
                if isinstance(search_result, dict) and search_result.get("error"):
                    return search_result
                
                total_found = int(search_result.get("total_found"))
                print(f"[GREEN] total_found={total_found} search_result={str(search_result)[:200]}")
                answer = search_result.get("answer")
                if answer:
                    print(f"[GREEN] Answer={answer}")
                    
                if total_found == 0:
                    return {
                        "error": f"No filings found in the SEC website",
                        "company": company_name
                        }                
              
                # Return the result from the sec_search().
                return search_result
                
            except Exception as e:
                return {
                    "error": f"Failed to fetch SEC data: {str(e)}",
                    "company": company_name
                }
        
        
        # ------------------------------------------------------------
        # Ticker Lookup (helps white agent use yfinance tools)
        # ------------------------------------------------------------
        @self.mcp_server.tool()
        async def get_ticker_symbol_handler(
            company_name: str
        ) -> dict:
            """
            Convert a company name to its US stock ticker symbol.
            
            ⚠️ ONLY RETURNS US EXCHANGE TICKERS (NYSE, NASDAQ, AMEX, etc.)
            
            USE THIS TOOL FIRST when you need to call get_financials_metrics or 
            get_financial_ratios, which require ticker symbols.
            
            Args:
                company_name: Company name, e.g., "Apple", "Netflix", "US Steel", "Airbnb"
            
            Returns:
                SUCCESS (US company):
                {
                    "company_name": "Netflix Inc",
                    "ticker": "NFLX",
                    "exchange": "NASDAQ"
                }
                
                ERROR (Non-US company):
                {
                    "error": "No US exchange listing found for 'Nestle'",
                    "company_name": "Nestle",
                    "non_us_results": {
                        "tickers": ["NESN"],
                        "exchanges": ["SWX"]
                    },
                    "suggestion": "Try sec_search_handler with company_name instead"
                }
                
                ERROR (Not found):
                {
                    "error": "Ticker not found",
                    "company_name": "Unknown Corp",
                    "suggestion": "Try sec_search_handler with company_name instead"
                }
            
            COMMON US TICKERS:
            - Apple → AAPL
            - Netflix → NFLX  
            - US Steel / United States Steel → X
            - Airbnb → ABNB
            - Microsoft → MSFT
            - Amazon → AMZN
            - Tesla → TSLA
            - Google/Alphabet → GOOGL
            
            NON-US COMPANIES (will return error):
            - Nestle → Swiss (SWX exchange)
            - Toyota → Japanese (JPX exchange)
            - HSBC → UK (LSE exchange)
            
            For non-US companies, use sec_search_handler instead.
            """
            try:
                print(f"[GREEN] Fetching ticker symbols for company={company_name}")
                
                # Search for filings
                result = await get_ticker_symbol(
                    company_name=company_name,
                )
                
                return result                
               
            except Exception as e:
                return {
                    "error": f"Lookup failed: {str(e)}",
                    "company_name": company_name
                }
            

        @self.mcp_server.tool()
        async def parse_html_handler(url: str, key: str) -> dict:
            """Parse HTML page and store in data_storage."""
            try:
                result = await parse_html(url)
                if "text" in result:
                    self.data_storage[key] = result["text"]
                    return {
                        "success": True,
                        "message": f"Saved to key '{key}'",
                        "available_keys": list(self.data_storage.keys())
                    }
                return result
            except Exception as e:
                return {"error": str(e)}
     
        
        @self.mcp_server.tool()
        async def retrieve_info_handler(
            prompt: str,
            input_character_ranges: dict = None
        ) -> dict:
            """Retrieve information from stored documents."""
            try:
                return await retrieve_information(
                    prompt=prompt,
                    input_character_ranges=input_character_ranges or {},
                    data_storage=self.data_storage,
                    llm_model=self.llm_model,
                    llm_api_key=self.safety_key
                )
            except Exception as e:
                return {"error": str(e)}

        # ========== NEW: YFINANCE TOOLS ==========        
        @self.mcp_server.tool()
        async def get_financial_metrics_handler(
            ticker: str,
            metrics: list = None,
            period: str = "annual",
            years: int = 3
        ) -> dict:
            """
            Get financial metrics from Yahoo Finance (FREE, FAST, no SEC parsing needed).
            
            ═══════════════════════════════════════════════════════════════
            ⚠️ REQUIRES TICKER SYMBOL - Use get_ticker_symbol first if needed!
            ═══════════════════════════════════════════════════════════════
            
            WHEN TO USE:
            ✓ Quick revenue, income, assets lookups
            ✓ Multi-year financial comparisons
            ✓ When you know the ticker symbol
            ✓ Faster than SEC filing parsing
            
            WHEN TO USE sec_search_handler INSTEAD:
            ✗ Need official/audited numbers
            ✗ Need specific SEC form data
            ✗ Questions about mergers, events, risk factors
            
            ═══════════════════════════════════════════════════════════════
            PARAMETERS:
            ═══════════════════════════════════════════════════════════════
            
            - ticker (str, REQUIRED): Stock ticker symbol
                Examples: "AAPL", "NFLX", "X" (US Steel), "ABNB"
                
                💡 Don't know the ticker? Call get_ticker_symbol first!
                
            - metrics (list): What to retrieve. Options:
                Revenue:     "revenue", "total_revenue"
                Income:      "net_income", "operating_income", "gross_profit"
                Balance:     "total_assets", "total_liabilities", "equity"
                Cash Flow:   "operating_cash_flow", "free_cash_flow", "capex"
                Per Share:   "eps", "shares_outstanding"
                
                If None, returns: revenue, net_income, operating_income,
                                  total_assets, shares_outstanding, free_cash_flow
                
            - period (str): "annual" or "quarterly"
            
            - years (int): Number of periods to retrieve (default: 3)
            
            ═══════════════════════════════════════════════════════════════
            RETURNS:
            ═══════════════════════════════════════════════════════════════
            {
                "ticker": "NFLX",
                "period": "annual",
                "data": {
                    "revenue": {
                        "2024-Q4": 9246000000,    ← Values in actual dollars
                        "2024-Q3": 8500000000,
                        "2024-Q2": 7685000000
                    },
                    "net_income": {
                        "2024-Q4": 1500000000,
                        "2024-Q3": 1200000000
                    }
                },
                "company_info": {
                    "name": "Netflix, Inc.",
                    "sector": "Communication Services",
                    "currency": "USD"
                }
            }
            
            ═══════════════════════════════════════════════════════════════
            EXAMPLES:
            ═══════════════════════════════════════════════════════════════
            
            Q: "What was Netflix revenue in 2023?"
            → get_financials_metrics(ticker="NFLX", metrics=["revenue"], years=3)
            → Answer: data["revenue"]["2023-Q4"] (or sum quarters)
            
            Q: "Compare Apple and Microsoft revenue"
            → Call twice:
               get_financials_metrics(ticker="AAPL", metrics=["revenue"])
               get_financials_metrics(ticker="MSFT", metrics=["revenue"])
            
            Q: "What are Airbnb's total assets?"
            → First: get_ticker_symbol("Airbnb") → "ABNB"
            → Then: get_financials_metrics(ticker="ABNB", metrics=["total_assets"])
            """
            try:
                return await get_financial_metrics(
                    ticker=ticker,
                    metrics=metrics,
                    period=period,
                    years=years
                )
            except Exception as e:
                return {"error": f"YFinance error: {str(e)}"}
        
        
        @self.mcp_server.tool()
        async def get_financial_ratios_handler(
            ticker: str,
            ratios: list = None,
            period: str = "annual"
        ) -> dict:
            """
            Calculate financial ratios from Yahoo Finance data.
            
            ═══════════════════════════════════════════════════════════════
            ⚠️ REQUIRES TICKER SYMBOL - Use get_ticker_symbol first if needed!
            ═══════════════════════════════════════════════════════════════
            
            WHEN TO USE:
            ✓ Profit margin questions
            ✓ Return on equity (ROE) / Return on assets (ROA)
            ✓ Efficiency ratios
            ✓ Comparative ratio analysis
            
            ═══════════════════════════════════════════════════════════════
            PARAMETERS:
            ═══════════════════════════════════════════════════════════════
            
            - ticker (str, REQUIRED): Stock ticker symbol
                💡 Don't know the ticker? Call get_ticker_symbol first!
            
            - ratios (list): Which ratios to calculate. Options:
                Profitability:  "profit_margin", "operating_margin", "gross_margin"
                Returns:        "roe" (return on equity), "roa" (return on assets)
                Efficiency:     "inventory_turnover", "asset_turnover"
                Cash:           "fcf_margin" (free cash flow margin)
                
            - period (str): "annual" or "quarterly"
            
            ═══════════════════════════════════════════════════════════════
            RETURNS:
            ═══════════════════════════════════════════════════════════════
            {
                "ticker": "NFLX",
                "period": "annual",
                "ratios": {
                    "profit_margin": {
                        "2024-Q4": 15.5,     ← Percentage values
                        "2024-Q3": 14.2
                    },
                    "roe": {
                        "2024-Q4": 0.25,     ← Decimal (25%)
                        "2024-Q3": 0.22
                    }
                }
            }
            
            ═══════════════════════════════════════════════════════════════
            EXAMPLES:
            ═══════════════════════════════════════════════════════════════
            
            Q: "What is Netflix's profit margin?"
            → get_financial_ratios(ticker="NFLX", ratios=["profit_margin"])
            
            Q: "What is Tesla's ROE?"
            → get_financial_ratios(ticker="TSLA", ratios=["roe"])
            """
            try:
                return await get_financial_ratios(
                    ticker=ticker,
                    ratios=ratios,
                    period=period
                )
            except Exception as e:
                return {"error": f"Ratio calculation error: {str(e)}"}

        # ------------------------------------------------------------------
        # Get Today's Date
        # ------------------------------------------------------------------
        @self.mcp_server.tool()
        async def get_today_date_handler(
            date_format: str = "iso",
            timezone: str = "UTC"
        ) -> dict:
            """
            CRITICAL TOOL: Returns today's real-world date.
            
            Use this tool IMMEDIATELY at the start of any task involving dates, fiscal years,
            filing deadlines, or "as of today" comparisons.
            
        
            Why you MUST know the current date:
            • There are questions in the database related to financial metrics and you must know if you need Q4 FY of the current year. 
            • Dataset questions are from real 10-K, 10-Q, 8-K, etc. filings 
            • You are running in November 2025 (or later)
            • Without this tool, you will hallucinate the current year and fail questions like:
                - "What is the most recent fiscal year reported?" → 2024, not 2025
                - "Has the 2024 10-K been filed?" → Yes (filed early 2025)
                - "What was revenue in FY2023 vs FY2024?" → needs to know 2024 is complete
        
            Always call this first if the question mentions:
            "latest", "most recent", "current", "as of", "fiscal year ended", etc.
            
            Returns today's date in various formats.
            
            Args:
                format (str): 
                    "iso"      → 2025-11-23
                    "full"     → Sunday, November 23, 2025
                    "ymd"      → 20251123
                    "mdy"      → 11/23/2025
                    "timestamp"→ 2025-11-23T14:30:22.123456+00:00
                timezone (str): IANA timezone, e.g. "America/New_York", "Europe/London", "UTC"
            
            Returns:
                dict with all formats + metadata
            """
            return get_today_date(
                date_format=date_format,
                timezone=timezone
            )

    # ------------------------------------------------------------------
    # Validate if the query is safe
    # ------------------------------------------------------------------
    async def validate_query(self, query: str) -> dict:
        """Validate if a finance query is safe."""
        
        if self.safety_check == False:  # 0=True 1=False
            return {"valid": True, "reason": "Safety check disabled"}
        
        prompt = f"""
        Classify this finance query as SAFE or UNSAFE.
        UNSAFE: non-public info, trading signals, PII.
        Query: "{query}"
        Respond JSON: {{"safe": true/false, "reason": "explanation"}}
        """
        try:
            #resp = litellm.completion(
            #    model=self.llm_model,
            #    messages=[{"role": "user", "content": prompt}],
            #    api_key=self.llm_api_key,
            #    response_format={"type": "json_object"},
            #)
            response = await safe_llm_call(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.llm_api_key,
                response_format={"type": "json_object"},
                temperature=0.1  # Lower temp for more focused decisions
                )      
            
            result = json.loads(response.choices[0].message.content)

            if self.verbose:
                print(f"[GREEN] Validate query={result.get('safe', False)}",file=sys.stderr)

            return {
                "valid": result.get("safe", False), 
                "reason": result.get("reason", "")
            }
        except Exception as e:
            return {"valid": False, "reason": f"Error: {str(e)}"}


    async def run_assessment(self, white_agent_address: str, config: dict):
        """
        Main assessment logic with stateful dataset reading.
        Continues from current_task_index (supports resuming after /reset).
        """
        num_tasks = config.get("num_tasks", 5)
        mcp_url = f"http://{self.agent_host}:{self.mcp_port}"
        
        # Load dataset if not already loaded
        if self.dataset_df is None:
            self._load_dataset()
        
        if self.dataset_df is None:
            return {
                "metric": "accuracy",
                "value": 0.0,
                "error": "Failed to load dataset",
                "total_tasks": 0,
                "correct_tasks": 0
            }
        
        # Determine which rows to process
        start_idx = self.current_task_index
        end_idx = min(start_idx + num_tasks, len(self.dataset_df))
        
        print(f"[GREEN] Running tasks {start_idx} to {end_idx-1} from {self.dataset_path}")
        print(f"[GREEN] MCP URL for white agent: {mcp_url}")

        results = []
        correct_count = 0
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx in range(start_idx, end_idx):
                row = self.dataset_df.iloc[idx]
                question = row["question"]
                expected_answer = row["answer"].strip().lower()
                
                # Prepare task for white agent via A2A
                task_payload = {
                    "question": question,
                    "mcp_url": mcp_url
                }
                
                # Call validation, if configured in the env file. 
                if self.safety_check == 0:  # 0=True 1=False
                    validation = await self.validate_query(question)
                
                    if not validation.get("valid", False):
                        print(f"[GREEN] Question {idx} considered unsafe, skipping it...",file=sys.stderr)
                        continue
                else:
                    print(f"[GREEN] Skipping safety check...",file=sys.stderr)
                
                
                
                try:
                    response = await client.post(
                        f"{white_agent_address}/a2a",
                        json=task_payload,
                        timeout=200.0
                    )
                    response.raise_for_status()
                    
                    white_response = response.json()
                    predicted_answer = white_response.get("answer", "").strip().lower()
                    
                    # Use LLM Judge for evaluation (instead of exact match)
                    evaluation = await self.judge.evaluate(
                        question=question,
                        expected_answer=expected_answer,
                        predicted_answer=predicted_answer
                    )
                    
                    is_correct = evaluation["correct"]
                    score = evaluation["score"]
                    
                    if is_correct:
                        correct_count += 1
                    
                    results.append({
                        "correct": is_correct,
                        "score": score,
                        "evaluation": evaluation
                    })
                    
                    # Store in history
                    self.assessment_history.append({
                        "task_index": idx,
                        "question": question,
                        "expected": expected_answer,
                        "predicted": predicted_answer,
                        "correct": is_correct,
                        "score": score,
                        "match_type": evaluation.get("match_type", "unknown"),
                        "reasoning": evaluation.get("reasoning", "")
                    })
                    
                    status = "✓" if is_correct else "✗"
                    print(f"[GREEN] Task {idx+1}/{num_tasks}: {status} "
                          f"(score: {score:.2f}) "
                          f"Expected: '{expected_answer[:30]}' "
                          f"Got: '{predicted_answer[:30]}'")
                    print(f"[GREEN]   → {evaluation.get('match_type', 'unknown')}: {evaluation.get('reasoning', '')[:80]}")
                    
                except Exception as e:
                    print(f"[GREEN] Error on task {idx+1}: {e}")
                    results.append({
                        "correct": False,
                        "score": 0.0,
                        "evaluation": {"error": str(e)}
                    })
                    self.assessment_history.append({
                        "task_index": idx,
                        "question": question,
                        "error": str(e),
                        "correct": False,
                        "score": 0.0
                    })
        
        # Update current position for next assessment
        self.current_task_index = end_idx
        
        # Calculate metrics
        correct_count_total = sum(1 for r in results if r["correct"])
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        accuracy = correct_count_total / len(results) if results else 0.0
        
        filename = self.save_to_csv("eval_result.csv")
        
        print(f"\n[GREEN] ═══════════════════════════════")
        print(f"[GREEN] Assessment Complete!")
        print(f"[GREEN] Accuracy: {accuracy:.3f} ({correct_count_total}/{len(results)})")
        print(f"[GREEN] Average Score: {avg_score:.3f}")
        print(f"[GREEN] Next task will start at index: {self.current_task_index}")
        print(f"[GREEN] Dataset: {self.dataset_path}")
        print(f"[GREEN] Result saved to file: {filename}")
        print(f"[GREEN] ═══════════════════════════════")
        
        return {
            "metric": "accuracy",
            "value": accuracy,
            "average_score": avg_score,
            "total_tasks": len(results),
            "correct_tasks": correct_count_total,
            "start_index": start_idx,
            "end_index": end_idx,
            "next_index": self.current_task_index,
            "description": "LLM-judged evaluation on dataset"
        }


    def save_to_csv(self, filename: str = None) -> str:
        """Save assessment history to CSV."""
        import csv
        from datetime import datetime
        
        if not self.assessment_history:
            print("No data to save.")
            return
        
        if filename is None:
            filename = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.assessment_history[0].keys())
            writer.writeheader()
            writer.writerows(self.assessment_history)
        
        #print(f"[GREEN] ═══════════════════════════════")
        #print(f"[GREEN] Assessment saved to file {filename}")
        #print(f"[GREEN] ═══════════════════════════════")
        return filename
        
    
    
    
    def run(self):
        """Start A2A + MCP server"""
        print(f"[GREEN] ═══════════════════════════════")
        print(f"[GREEN] Starting Finance Green Agent")
        print(f"[GREEN] A2A: {self.agent_host}:{self.agent_port}")
        print(f"[GREEN] MCP: {self.agent_host}:{self.mcp_port}")
        print(f"[GREEN] Dataset: {self.dataset_path}")
        print(f"[GREEN] ═══════════════════════════════")
        
        # Load dataset on startup
        self._load_dataset()
        
        # Start MCP server in background thread
        import threading
        mcp_thread = threading.Thread(
            target=lambda: self.mcp_server.run(
                transport="sse",
                host=self.agent_host,
                port=self.mcp_port
            ),
            daemon=True
        )
        mcp_thread.start()
        
        # Give MCP time to start
        import time
        time.sleep(2)
        
        # Run FastAPI (A2A) - blocking
        uvicorn.run(
            self.app,
            host=self.agent_host,
            port=self.agent_port,
            log_level="info"
        )


if __name__ == "__main__":
    agent = GreenAgent()
    agent.run()
