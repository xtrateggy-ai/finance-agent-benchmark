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
from tools import *

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

        self.llm_model   = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key = os.getenv("LLM_API_KEY")

        self.dataset_path = os.getenv("DATASET", "data/public.csv")
        self.use_disk_cache = bool(int(os.getenv("USE_DISK_CACHE", 1)))  # 1=True 0=False - Save SEC filings to disk.
        
        # === STATE MANAGEMENT ===
        self.dataset_df = None  # Will be loaded on first use
        self.current_task_index = 0  # Track which task we're on
        self.data_storage = {}  # For parse_html_page + retrieve_info
        self.assessment_history = []  # Track assessment results
        
        
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

        # -------------------------------------------------------------------
        # Handler for MCP/A2A agent tool usage
        # -------------------------------------------------------------------
        
        # company_name to CIK resolver
        @self.mcp_server.tool()
        async def cik_resolver_handler(company_name: str) -> dict:
            """Resolve company name to CIK using official SEC ticker mapping."""

            if self.verbose:
                print(f"[GREEN] Resolving CIK for: {company_name}", file=sys.stderr)

            try:
                return await resolve_cik(company_name)
            except Exception as e:
                return {"error": str(e)}
        
        # -----------------------
        # fetch company submissions
        # -----------------------
        @self.mcp_server.tool()
        async def submissions_handler(cik: str) -> dict:
            """
            Tool entrypoint for MCP Agent.
            Accepts CIK (any length), normalizes it, fetches submissions.
            """
            if self.verbose:
                print(f"[GREEN] submissions_handler received CIK: {cik}", file=sys.stderr)

            try:
                return await submissions_tool(cik)
            except Exception as e:
                return {"error": str(e)}

        # return submissions_handler
    

        # ---------------------------
        # fetch xbrl company concepts
        # ---------------------------
        @self.mcp_server.tool()
        async def xbrl_companyconcept_handler(
        cik: int,
        taxonomy: str,
        concept: str,
        ) -> dict:
            """
                XBRL Company-Concept Tool:
                Fetch all XBRL facts for a CIK + taxonomy + concept.
            """

            if self.verbose:
                print("[GREEN] Calling xbrl_companyconcept...", file=sys.stderr)

            try:
                return await fetch_company_concept(
                    cik10=str(cik),
                    taxonomy=taxonomy,
                    concept=concept,
                )
            except Exception as e:
                return {"error": str(e)}

        # --------------------------
        # fetch xbrl company facts
        # --------------------------
        @self.mcp_server.tool()
        async def companyfacts_handler(cik: int) -> dict:
            """
            Tool entrypoint for MCP/LLM:
            Fetch all XBRL company facts for a CIK.
            """
            if self.verbose:
                print(f"[GREEN] Calling companyfacts_handler for CIK: {cik}", file=sys.stderr)

            try:
                return await fetch_companyfacts(str(cik))
            except Exception as e:
                return {"error": str(e)}
            
        # --------------------------
        # fetch xbrl frames
        # --------------------------
        @self.mcp_server.tool()
        async def frames_handler(
            taxonomy: str,
            concept: str,
            unit: str,
            period: str,
        ) -> dict:
            """
            XBRL Frames Tool
            Fetch the most recent fact for a reporting entity based on:
            - taxonomy (us-gaap, ifrs-full, etc.)
            - concept (AccountsPayableCurrent, NetIncomeLoss, etc.)
            - unit (USD, USD-per-shares, pure)
            - period (CY2023, CY2023Q2, CY2023Q2I)
            """

            if verbose:
                print(f"[GREEN] Calling frames_handler for {taxonomy}/{concept}/{unit}/{period}", file=sys.stderr)

            try:
                return await fetch_frames(taxonomy, concept, unit, period)
            except Exception as e:
                return {"error": str(e)}

        # --------------------------
        # Google Web Search Tool
        # --------------------------
        @self.mcp_server.tool()
        async def google_search_handler(query: str, verbose: bool = False) -> dict:
            """
            MCP/LLM handler for Google Custom Search API.
            Performs a web search and returns top 5 results snippets.
            """
            if verbose:
                print(f"[GOOGLE_SEARCH] Query received: {query}", file=sys.stderr)

            try:
                return await google_search(query=query, verbose=verbose)
            except Exception as e:
                return {"error": str(e)}

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
