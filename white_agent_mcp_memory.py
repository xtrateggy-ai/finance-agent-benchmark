# app/white_agent_mcp_memory.py
"""
Finance-White Agent - Improved Tool Selection & Memory
WITH REDUNDANT TOOL CALL PREVENTION
"""

import os
import sys
import json
import litellm
import tomllib
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx
from typing import List, Dict, Any, Union

from mcp.client.sse import sse_client
from mcp import ClientSession

from utils.llm_manager import safe_llm_call
from utils.local_llm_wrapper import safe_local_llm_call
from utils.env_setup import init_environment

init_environment()


class ConversationMemory:
    """
    Enhanced conversation memory with tool call tracking
    Prevents redundant tool calls
    """
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.tool_call_count = {}  # Track how many times each tool was called
        self.successful_tools = set()  # Track which tools returned useful data
    
    def add_tool_call(self, tool: str, params: dict, result: Any):
        """Add a tool call to memory and track usage"""
        # Track call count
        self.tool_call_count[tool] = self.tool_call_count.get(tool, 0) + 1
        
        # Mark as successful if result has useful data
        if self._is_useful_result(result):
            self.successful_tools.add(tool)
        
        self.history.append({
            "type": "tool_call",
            "tool": tool,
            "params": params,
            "result": result,
            "timestamp": self._get_timestamp(),
            "call_number": self.tool_call_count[tool]
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def _is_useful_result(self, result: Any) -> bool:
        """Check if result contains useful information"""
        if isinstance(result, dict):
            # FIRST: Check for error - if error exists, NOT useful
            if "error" in result:
                return False
            
            # Has timeline with data = useful
            if result.get("timeline") and len(result.get("timeline", [])) > 0:
                return True
            
            # Has financial data = useful
            if result.get("data") and len(result.get("data", {})) > 0:
                return True
            
            # Has sections = useful
            if result.get("sections") and len(result.get("sections", {})) > 0:
                return True
            
            # Has company info = useful
            if result.get("company") and not result.get("error"):
                return True
        
        # Has substantial text = useful
        if isinstance(result, str) and len(result) > 100:
            return True
        
        return False
    
    def should_try_tool(self, tool: str) -> tuple[bool, str]:
        """
        Determine if we should try calling this tool again.
        
        Returns:
            (should_call, reason)
        """
        # Never called before = definitely try
        if tool not in self.tool_call_count:
            return True, "First time calling this tool"
        
        # Already got useful results = don't call again
        if tool in self.successful_tools:
            return False, f"Already got useful data from {tool}"
        
        # Called once and failed = try once more (maybe with different params)
        if self.tool_call_count[tool] == 1:
            return True, "Retrying with different parameters"
        
        # Called twice or more = stop
        if self.tool_call_count[tool] >= 3:
            return False, f"{tool} already tried {self.tool_call_count[tool]} times"
        
        return True, ""
    
    def get_tool_usage_summary(self) -> str:
        """Get summary of which tools were called and results"""
        if not self.tool_call_count:
            return "No tools called yet."
        
        lines = ["Tool Usage Summary:"]
        for tool, count in self.tool_call_count.items():
            status = "✓ Got useful data" if tool in self.successful_tools else "✗ No useful data"
            lines.append(f"  - {tool}: {count} call(s) → {status}")
        
        return "\n".join(lines)
    
    def add_reasoning(self, thought: str):
        """Add reasoning step to memory"""
        self.history.append({
            "type": "reasoning",
            "thought": thought,
            "timestamp": self._get_timestamp()
        })
    
    def get_summary(self, last_n: int = 3) -> str:
        """Get summary of recent history (reduced from 5 to 3)"""
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history
        
        summary = []
        for item in recent:
            if item["type"] == "tool_call":
                result_preview = str(item['result'])[:150]
                summary.append(
                    f"[Call #{item['call_number']}] {item['tool']}({item['params']})\n"
                    f"Result: {result_preview}..."
                )
            elif item["type"] == "reasoning":
                summary.append(f"Thought: {item['thought']}")
        
        return "\n\n".join(summary)
    
    def clear(self):
        """Clear memory"""
        self.history = []
        self.tool_call_count = {}
        self.successful_tools = set()
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()


class WhiteAgent:
    """
    Improved Finance White Agent with:
    - Better tool selection logic
    - Conversation memory
    - Redundant tool call prevention
    """

    def __init__(self):
        self.agent_host = os.getenv("WHITE_AGENT_HOST", "0.0.0.0")
        self.agent_port = int(os.getenv("WHITE_AGENT_PORT", 8000))
        self.name = "finance-white-agent"
        
        # Load agent card
        self.card_path = os.getenv(
            "WHITE_CARD",
            Path(__file__).parent / "cards" / "white_card.toml"
        )
        if not os.path.exists(self.card_path):
            raise FileNotFoundError(f"Agent card {self.card_path} not found")
        
        with open(self.card_path, "rb") as f:
            self.agent_card = tomllib.load(f)
        
        # LLM configuration
        self.llm_model   = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key = os.getenv("LLM_API_KEY")

        # Check if we should use local LLM
        self.llm_use_local = bool(int(os.getenv("USE_LOCAL_LLM_WHITE", "1")))

        self.max_iterations = int(os.getenv("WHITE_AGENT_MAX_ITER", 6))
        

        # Memory
        self.memory = ConversationMemory(max_history=10)
        
        # Logging
        self.log_file = self.start_new_log()
        
        # Create FastAPI app
        self.app = FastAPI(title="Finance White Agent")
        self._setup_routes()


    def start_new_log(self) -> str:
        """Create fresh log file"""
        self.log_file = "eval_white.txt"
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('')
        print(f"[WHITE] New evaluation log: {self.log_file}")
        return self.log_file

    def log_result(self, result: Dict[str, Any]):
        """Append result to log"""
        if self.log_file is None:
            raise RuntimeError("Log file not initialized")
        line = json.dumps(result, ensure_ascii=False, default=str)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def log_separator(self, text: str = ""):
        """Add visual separator"""
        separator = "=" * 80
        self.log_result({
            "separator": separator,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })

    def _setup_routes(self):
        """Setup A2A protocol endpoints"""
        
        @self.app.get("/card")
        @self.app.get("/.well-known/agent-card.json")
        async def get_card():
            return JSONResponse(self.agent_card)
        
        @self.app.post("/a2a")
        async def handle_task(request: Request):
            try:
                payload = await request.json()
                question = payload.get("question")
                mcp_url = payload.get("mcp_url")
                
                print(f"[WHITE] ═══════════════════════════════════")
                print(f"[WHITE] Q: {question[:80]}...")
                print(f"[WHITE] MCP: {mcp_url}")
                print(f"[WHITE] ═══════════════════════════════════")
                
                # Clear memory for new question
                self.memory.clear()
                
                # Generate answer
                answer = await self.answer_question(question, mcp_url)
                
                if isinstance(answer, str):
                    print(f"[WHITE] A: {answer[:80]}...")
                if isinstance(answer, dict):
                    print(f"[WHITE] A: {str(answer)[:80]}...")
                
                return JSONResponse({
                    "status": "completed",
                    "answer": answer
                })
                
            except Exception as e:
                print(f"[WHITE] Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "status": "error",
                    "message": str(e),
                    "answer": "ERROR"
                }, status_code=500)
        
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "agent": self.name}

    async def answer_question(self, question: str, mcp_url: str) -> str:
        """
        Reasoning loop with redundant tool call prevention
        """
        
        # Parse MCP URL
        import re
        match = re.match(r"http://([^:]+):(\d+)", mcp_url)
        if not match:
            return "ERROR: Invalid MCP URL"
        
        mcp_host = match.group(1)
        mcp_port = int(match.group(2))
        
        sse_url = f"http://{mcp_host}:{mcp_port}/sse" if not mcp_url.endswith("/sse") else mcp_url
        
        print(f"[WHITE] Connecting to: {sse_url}", file=sys.stderr)
        
        
        try:
            async with sse_client(sse_url, timeout=600.0) as (read, write):  # timeout=10 min
                async with ClientSession(read, write) as session:
                    # Initialize
                    await session.initialize()
                    print(f"[WHITE] MCP initialized", file=sys.stderr)
                    
                    # Discover tools
                    tools_result = await session.list_tools()
                    available_tools = tools_result.tools
                    
                    # Cap iterations at reasonable number
                    available_tool_count = len(available_tools)
                    self.max_iterations = min(available_tool_count, 6)
                    
                    # Display tool names cleanly
                    tool_names = [t.name for t in available_tools]
                    print(f"[WHITE] {available_tool_count} tools: {', '.join(tool_names)}", file=sys.stderr)
                    print(f"[WHITE] max_iterations={self.max_iterations}", file=sys.stderr)
                    
                    # Track success/failure
                    successful_calls = 0
                    failed_calls = 0
                    
                    # Multi-turn reasoning
                    for iteration in range(self.max_iterations):
                        print(f"\n[WHITE] --- Iteration {iteration+1}/{self.max_iterations} ---", file=sys.stderr)
                        print(f"[WHITE] Q: {question[:80]}...")

                        # Emergency break if too many failures
                        if failed_calls >= 3:
                            print(f"[WHITE] ⚠️ Too many failures ({failed_calls}), generating answer", file=sys.stderr)
                            return await self._generate_final_answer(question)
                        
                        # Build context-aware prompt
                        if iteration == 0:
                            system_prompt = self._build_initial_prompt(question, available_tools)
                        else:
                            system_prompt = self._build_followup_prompt(question, available_tools)
                        
                        # Get LLM decision
                        try:
                            if self.llm_use_local:
                                response = await safe_local_llm_call(
                                    messages=[
                                        {
                                            "role": "system", 
                                            "content": "You are a precise JSON-only assistant. Respond ONLY with valid JSON."
                                        },
                                        {
                                            "role": "user", 
                                            "content": system_prompt + "\n\nRespond with ONLY JSON, nothing else."
                                        }
                                    ],
                                    response_format={"type": "json_object"},
                                    temperature=0.1,
                                    component="white" 
                                )
                            else:
                                response = await safe_llm_call(
                                    model=self.llm_model,
                                    messages=[
                                        {
                                            "role": "system", 
                                            "content": "You are a precise JSON-only assistant. Respond ONLY with valid JSON."
                                        },
                                        {
                                            "role": "user", 
                                            "content": system_prompt + "\n\nRespond with ONLY JSON, nothing else."
                                        }
                                    ],
                                    api_key=self.llm_api_key,
                                    response_format={"type": "json_object"},
                                    temperature=0.1
                                )
                            
                            if hasattr(response, 'choices') and response.choices:
                                # Parse LLM response
                                response_text = response.choices[0].message.content
                                print(f"[WHITE] LLM response preview: {response_text[:200]}...", file=sys.stderr)
                            else:
                                # Malformed success response → structured error JSON
                                response_text = json.dumps({
                                    "error": "malformed_response",
                                    "details": f"Unexpected response format; could be due to exceed rate limit: {response}",
                                    "predicted_answer": "error_generating_answer"
                                })
                                print(f"[WHITE] LLM error response: {response}")


                            try:
                                decision = json.loads(response_text)
                                
                                # Log decision
                                self.log_result(decision)
                                self.log_separator()
                                
                            except json.JSONDecodeError as e:
                                print(f"[WHITE] JSON parse error: {e}", file=sys.stderr)
                                failed_calls += 1
                                continue
                            
                            action = decision.get("action")
                            
                            if not action:
                                print(f"[WHITE] No action in decision", file=sys.stderr)
                                failed_calls += 1
                                continue
                            
                            print(f"[WHITE] Decision: {action}", file=sys.stderr)
                            
                            # Handle decision
                            if action == "answer":
                                final_answer = decision.get("answer", "")
                                reasoning = decision.get("reasoning", "")
                                self.memory.add_reasoning(reasoning)
                                return final_answer
                            
                            elif action == "tool_call":
                                tool = decision.get("tool")
                                params = decision.get("params", {})
                                reasoning = decision.get("reasoning", "")
                                
                                # ═══ CRITICAL: CHECK IF WE SHOULD CALL THIS TOOL ═══
                                should_call, check_reason = self.memory.should_try_tool(tool)
                                
                                if not should_call:
                                    print(f"[WHITE] ⚠️ Skipping {tool}: {check_reason}", file=sys.stderr)
                                    self.memory.add_reasoning(
                                        f"Refused to call {tool} again: {check_reason}. "
                                        "Must try DIFFERENT tool or provide final answer."
                                    )
                                    failed_calls += 1
                                    continue
                                
                                # Verify tool exists
                                if not any(t.name == tool for t in available_tools):
                                    print(f"[WHITE] Tool '{tool}' not found", file=sys.stderr)
                                    failed_calls += 1
                                    continue
                                
                                # Normalize parameter names      
                                if "search_term" in params and "query" not in params:
                                    params["query"] = params.pop("search_term")
                                if "search_query" in params and "query" not in params:
                                    params["query"] = params.pop("search_query")

                                attempt_num = self.memory.tool_call_count.get(tool, 0) + 1
                                print(f"[WHITE] ✓ Calling {tool} (attempt #{attempt_num})", file=sys.stderr)
                                print(f"[WHITE]   Params: {params}", file=sys.stderr)
                                print(f"[WHITE]   Reason: {reasoning}", file=sys.stderr)
                                
                                self.memory.add_reasoning(reasoning)
                                
                                # Call tool
                                tool_result = await session.call_tool(tool, arguments=params)
                                result_data = self._extract_text_from_tool_result(tool_result)
                                
                                print(f"[WHITE] Result preview: {str(result_data)[:5000]}...", file=sys.stderr)
                                
                                # Log result
                                self.log_result({"tool_result": result_data})
                                self.log_separator()

                                # Check if tool call failed
                                if isinstance(result_data, dict) and result_data.get("error"):
                                    print(f"[WHITE] ✗ Tool failed: {result_data.get('error')}", file=sys.stderr)
                                    # Store as failed call (will NOT mark as successful)
                                    self.memory.add_tool_call(tool, params, result_data)
                                    self.memory.add_reasoning(f"Tool {tool} failed: {result_data.get('error')}. Should retry with different parameters or try different tool.")
                                    failed_calls += 1
                                    continue
                                
                                # Success - store in memory (will mark as successful)
                                self.memory.add_tool_call(tool, params, result_data)
                                successful_calls += 1
                                print(f"[WHITE] ✓ Tool succeeded (got useful data)", file=sys.stderr)
                                
                                # If we got useful data and tried multiple times, encourage answer
                                if successful_calls >= 1 and iteration >= 2:
                                    print(f"[WHITE] ℹ️ Have data, LLM should consider answering", file=sys.stderr)
                                
                        except json.JSONDecodeError as e:
                            print(f"[WHITE] JSON error: {e}", file=sys.stderr)
                            failed_calls += 1
                            continue
                        except Exception as e:
                            print(f"[WHITE] Error: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc()
                            failed_calls += 1
                            continue
                    
                    # Max iterations reached
                    print(f"[WHITE] Max iterations reached, generating final answer", file=sys.stderr)
                    return await self._generate_final_answer(question)
            
        except Exception as e:
            print(f"[WHITE] MCP error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"

    def _build_initial_prompt(self, question: str, tools: List) -> str:
        """Build initial reasoning prompt"""
        tools_desc = "\n".join([
            f"- {t.name}: {t.description or 'No description'}"
            for t in tools if t.name != "validate_query"
        ])
        
        return f"""PERSONA: You are a precise financial analyst. Respond ONLY with valid JSON.

QUESTION: {question}

AVAILABLE TOOLS:
{tools_desc}

═══════════════════════════════════════════════════════════════════════
CRITICAL DECISION TREE (Follow this order):
═══════════════════════════════════════════════════════════════════════

Step 1: Does question involve TIME? (Q4 FY 2025, "latest", "recent", "2024")
├─ YES → Call get_today_date_handler FIRST to check if data exists
│         Example: Q4 FY 2025 filings won't exist in Nov 2024!
│
└─ NO → Continue to Step 2

Step 2: Does question need COMPANY TICKER?
├─ Company name not standard (e.g., "BBSI", abbreviation)
│  └─ Call get_ticker_symbol_handler first
│     Then use ticker_symbol in sec_search_handler
│
└─ Company name is clear (e.g., "Apple", "Netflix")
   └─ Go directly to sec_search_handler

Step 3: Which data source?
├─ Need OFFICIAL FILINGS (mergers, board changes, guidance)
│  └─ sec_search_handler (use ticker_symbol if you have it!)
│
├─ Need QUICK METRICS (revenue, assets, ratios)
│  └─ get_financial_metrics_handler or get_financial_ratios_handler
│
└─ Need SPECIFIC DOCUMENT PARSING
   └─ parse_html_handler + retrieve_info_handler

═══════════════════════════════════════════════════════════════════════

ALWAYS include as parameter to the tool, if listed as parameter:
    - question: str (mandatory)

RULES:
1. Call ONLY ONE tool per response
2. If you get ticker from get_ticker_symbol_handler, USE IT in next call:
   ✅ sec_search_handler(ticker_symbol="BBSI", ...)
   ❌ sec_search_handler(company_name="Barrett Business...", ...)
3. For SEC filings, use WIDE date ranges (e.g., 2018-2025)
4. For future periods (Q4 FY 2025), check today's date FIRST

RESPOND WITH:
{{
    "action": "tool_call",
    "tool": "tool_name",
    "params": {{"param": "value"}},
    "reasoning": "why this tool"
}}

OR:
{{
    "action": "answer",
    "answer": "your answer",
    "reasoning": "why correct"
}}
"""

    def _build_followup_prompt(self, question: str, tools: List) -> str:
        """Build follow-up prompt with tool usage awareness"""
        memory_summary = self.memory.get_summary(last_n=3)
        tool_usage = self.memory.get_tool_usage_summary()
        
        # Check if we're stuck calling same tool
        stuck_on_tool = None
        if self.memory.tool_call_count:
            max_calls = max(self.memory.tool_call_count.values())
            if max_calls >= 2:
                for tool, count in self.memory.tool_call_count.items():
                    if count >= 2 and tool in self.memory.successful_tools:
                        stuck_on_tool = tool
                        break
        
        tools_desc = "\n".join([
            f"- {t.name}: {t.description or 'No description'}"
            for t in tools if t.name != "validate_query"
        ])
        
        stuck_warning = ""
        if stuck_on_tool:
            stuck_warning = f"""
⚠️⚠️⚠️ CRITICAL WARNING ⚠️⚠️⚠️
You keep trying to call {stuck_on_tool} but it already succeeded!
DO NOT call {stuck_on_tool} again. It will be BLOCKED.
You MUST either:
1. Provide final answer based on data from {stuck_on_tool}
2. Call a DIFFERENT tool if more info needed

If you call {stuck_on_tool} again, you're wasting iterations!
"""
        
        return f"""QUESTION: {question}

AVAILABLE TOOLS:
{tools_desc}

═══════════════════════════════════════════════
WHAT HAPPENED:
═══════════════════════════════════════════════
{memory_summary}

═══════════════════════════════════════════════
{tool_usage}
═══════════════════════════════════════════════
{stuck_warning}

YOUR OPTIONS:

Option A: PROVIDE FINAL ANSWER (if you have enough data)
{{
    "action": "answer",
    "answer": "Based on data from [tool], the answer is...",
    "reasoning": "I have sufficient information from previous tool calls"
}}

Option B: CALL DIFFERENT TOOL (if you need more data)
{{
    "action": "tool_call",
    "tool": "DIFFERENT_tool_name",  ← Must be different!
    "params": {{}},
    "reasoning": "Need additional data that [previous_tool] didn't provide"
}}

ALWAYS include as parameter to the tool, if listed as parameter:
    - question: str (mandatory)

⚠️ RULES:
1. DO NOT call tools marked "✓ Got useful data"
2. DO NOT call same tool more than 2 times
3. If blocked, you MUST try different approach or answer
4. Calling blocked tool again = wasted iteration

RESPOND WITH ONLY JSON:
"""

    async def _generate_final_answer(self, question: str) -> str:
        """Generate final answer from memory"""
        memory_summary = self.memory.get_summary()
        
        if not memory_summary:
            return "NO_ANSWER_FOUND"
        
        prompt = f"""Question: {question}

Research completed:
{memory_summary}

Provide concise final answer based on data obtained.
"""
        
        try:
            if self.llm_use_local:
                response = await safe_local_llm_call(
                    messages=[
                        {"role": "system", "content": "You are a precise assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
            else:
                response = await safe_llm_call(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    api_key=self.llm_api_key,
                    temperature=0.1
                )            

            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[WHITE] Final answer error: {e}", file=sys.stderr)
            return "ERROR_GENERATING_ANSWER"

    """
    def _extract_text_from_tool_result(self, result) -> Any:
        
        # Extract data from MCP tool result.
        # Returns the actual data structure (dict/str), NOT just text.
        
        if result.content:
            texts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    text = content.text
                    
                    # Try to parse as JSON first
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, keep as text
                        texts.append(text)
            
            # Return joined text if multiple items
            if texts:
                return "\n".join(texts)
        
        return ""
    """
    
    
    """
    def _extract_text_from_tool_result(self, result) -> Union[str, dict]:
        
        # Universal generic parser for ALL green-agent tools.
        # - Returns str on success
        # - Returns {"error": "..."} dict on failure (for white-agent error handling)
        # - Handles: plain text, JSON strings, dicts, lists, malformed output
        
        if not result.content:
            return {"error": "Tool returned no content"}
    
        collected_texts = []
    
        for content in result.content:
            if not hasattr(content, "text") or not content.text:
                continue
    
            raw = content.text.strip()
            if not raw:
                continue
    
            # 1. First: try to parse as JSON (most tools return JSON string)
            try:
                data = json.loads(raw)
    
                # Case A: Tool explicitly returned an error dict
                if isinstance(data, dict) and data.get("error"):
                    return {"error": data["error"]}
    
                # Case B: sec_search_rag success → extract the 'answer' field
                if isinstance(data, dict) and "answer" in data:
                    answer = data["answer"]
                    if answer and str(answer).strip():
                        collected_texts.append(str(answer).strip())
                        continue
                    else:
                        # answer field exists but is empty → treat as soft error
                        return {"error": "Tool returned empty answer"}
    
                # Case C: Other valid JSON (e.g. get_ticker_symbol returns {"ticker": "NFLX"})
                if isinstance(data, (dict, list)):
                    collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                    continue
    
            except (json.JSONDecodeError, TypeError, ValueError):
                # Not JSON → fall through to plain text handling
                pass
    
            # 2. Plain text fallback (parse_html_handler, etc.)
            collected_texts.append(raw)
    
        # Final: join all collected parts
        final_answer = "\n".join(collected_texts).strip()
    
        if not final_answer:
            return {"error": "Tool returned empty or unparseable result"}
    
        return final_answer
    """

    def _extract_text_from_tool_result(self, result) -> Union[str, dict]:
        """
        Universal generic parser for ALL green-agent tools.
        - Returns str on success
        - Returns {"error": "..."} dict on failure (for white-agent error handling)
        - Handles: plain text, JSON strings, dicts, lists, malformed output
        - Handles both LLM+RAG and REGEX extraction modes from sec_search_rag
        """
        if not result.content:
            return {"error": "Tool returned no content"}
    
        collected_texts = []
    
        for content in result.content:
            if not hasattr(content, "text") or not content.text:
                continue
    
            raw = content.text.strip()
            if not raw:
                continue
    
            # 1. First: try to parse as JSON (most tools return JSON string)
            try:
                data = json.loads(raw)
    
                # Case A: Tool explicitly returned an error dict
                if isinstance(data, dict) and data.get("error"):
                    return {"error": data["error"]}
    
                # Case B: sec_search_rag with REGEX extraction
                # Return full structured data so LLM can analyze timeline
                if (isinstance(data, dict) and 
                    data.get("extraction_method") == "regex" and
                    data.get("timeline")):
                    
                    # Check if timeline has data
                    if len(data.get("timeline", [])) > 0:
                        collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                        continue
                    else:
                        return {"error": "REGEX extraction returned empty timeline"}
    
                # Case C: sec_search_rag with LLM+RAG extraction
                # Extract just the 'answer' field
                if (isinstance(data, dict) and 
                    data.get("extraction_method") == "llm_rag" and
                    "answer" in data):
                    
                    answer = data["answer"]
                    if answer and str(answer).strip():
                        collected_texts.append(str(answer).strip())
                        continue
                    else:
                        # LLM+RAG should always return an answer
                        return {"error": "LLM+RAG returned empty answer"}
    
                # Case D: Other tools with simple dict responses
                # (e.g. get_ticker_symbol returns {"ticker": "NFLX"})
                if isinstance(data, dict):
                    # Check if it has useful data (not just metadata)
                    has_data = any(
                        key not in ["company", "ticker_symbol", "cik", "sic", "sic_description", "question", "extraction_method"]
                        for key in data.keys()
                    )
                    
                    if has_data:
                        collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                        continue
    
                # Case E: Lists (might be returned by some tools)
                if isinstance(data, list):
                    collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                    continue
    
            except (json.JSONDecodeError, TypeError, ValueError):
                # Not JSON → fall through to plain text handling
                pass
    
            # 2. Plain text fallback (parse_html_handler, etc.)
            collected_texts.append(raw)
    
        # Final: join all collected parts
        final_answer = "\n".join(collected_texts).strip()
    
        if not final_answer:
            return {"error": "Tool returned empty or unparseable result"}
    
        return final_answer



    def run(self):
        """Start agent server"""
        print(f"[WHITE] ═══════════════════════════════════")
        print(f"[WHITE] Finance White Agent (Fixed)")
        print(f"[WHITE] Server: {self.agent_host}:{self.agent_port}")
        print(f"[WHITE] Memory: Enabled with tool tracking")
        print(f"[WHITE] ═══════════════════════════════════")
        
        uvicorn.run(
            self.app,
            host=self.agent_host,
            port=self.agent_port,
            log_level="info"
        )


if __name__ == "__main__":
    agent = WhiteAgent()
    agent.run()
