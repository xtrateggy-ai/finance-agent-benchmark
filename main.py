#!/usr/bin/env python3
"""
AgentBeats-Compatible Launcher (main.py)
Implements A2A protocol for agent lifecycle management

Usage:
    python main.py run                    # Start launcher (NO auto-start, wait for AgentBeats)
    python main.py run --auto_start       # Start launcher AND agents (for local testing)
    python main.py run --port 7000        # Custom port
"""

import os
import sys
import asyncio
import signal
import subprocess
import argparse
import json
import time
import tomllib
from datetime import datetime
from pathlib import Path
import multiprocessing
from typing import Optional
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_setup import init_environment
init_environment()

from contextlib import asynccontextmanager

"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifespan events for FastAPI
    # Startup
    launcher = app.state.launcher
    if launcher.auto_start:
        print("[MAIN]Auto-starting agents...")
        # Wait longer for server to be fully ready
        await asyncio.sleep(5)  # Increased from 3
        
        # Verify server is ready first
        max_retries = 10
        for i in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    health_check = await client.get(
                        f"http://127.0.0.1:{launcher.launcher_port}/health",
                        timeout=2.0
                    )
                    if health_check.status_code == 200:
                        print(f"[MAIN]Launcher ready, starting agents...")
                        break
            except:
                print(f"[MAIN]Waiting for launcher to be ready... ({i+1}/{max_retries})")
                await asyncio.sleep(1)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{launcher.launcher_port}/start",
                    timeout=60.0
                )
                result = response.json()
                print(f"[MAIN]Auto-start result: {result}")
        except Exception as e:
            print(f"[MAIN]Auto-start failed: {e}")
            import traceback
            traceback.print_exc()
    
    yield
    
    # Shutdown
    if launcher.green_process:
        launcher.green_process.terminate()
    if launcher.white_process:
        launcher.white_process.terminate()
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI"""
    launcher = app.state.launcher
    
    # Auto-start agents if requested
    if launcher.auto_start:
        print("[MAIN]Auto-starting agents...")
        
        # Start green agent
        if not launcher.green_process or launcher.green_process.poll() is not None:
            print("[MAIN]Starting green agent...")
            launcher.green_process = subprocess.Popen(
                [sys.executable, "green_agent_mcp_a2a_judge_rag.py"],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            await asyncio.sleep(3)
        
        # Start white agent
        if not launcher.white_process or launcher.white_process.poll() is not None:
            print("[MAIN]Starting white agent...")
            launcher.white_process = subprocess.Popen(
                [sys.executable, "white_agent_mcp_memory.py"],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            await asyncio.sleep(2)
        
        # Wait for agents to be ready
        print("[MAIN]Waiting for agents to be ready...")
        green_ready = await launcher._wait_agent_ready(
            f"http://{launcher.green_host}:{launcher.green_port}",
            timeout=30
        )
        white_ready = await launcher._wait_agent_ready(
            f"http://{launcher.white_host}:{launcher.white_port}",
            timeout=30
        )
        
        if green_ready and white_ready:
            print("[MAIN]✅ All agents started successfully")
        else:
            print("[MAIN]⚠️  Warning: Some agents may not have started")
            if not green_ready:
                print("[MAIN]   Green agent not responding")
            if not white_ready:
                print("[MAIN]   White agent not responding")
    else:
        print("[MAIN]Auto-start disabled - agents will start when /start is called")
    
    yield
    
    # Shutdown - cleanup agents
    print("[MAIN]Shutting down...")
    if launcher.green_process:
        print("[MAIN]Terminating green agent...")
        launcher.green_process.terminate()
        try:
            launcher.green_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            launcher.green_process.kill()
    
    if launcher.white_process:
        print("[MAIN]Terminating white agent...")
        launcher.white_process.terminate()
        try:
            launcher.white_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            launcher.white_process.kill()


class AgentBeatsLauncher:
    """
    AgentBeats-compatible launcher.
    
    This is NOT an agent - it's a controller that:
    1. Starts/stops the actual agents (green + white)
    2. Provides discovery endpoints for AgentBeats
    3. Proxies agent information to AgentBeats
    """
    def __init__(self, port: int = 7000, auto_start: bool = False):
        # ✅ FIX: Read port from AgentBeats environment variable
        # AgentBeats sets AGENT_PORT when running through run_ctrl
        agentbeats_port = os.getenv("AGENT_PORT")
        if agentbeats_port:
            self.launcher_port = int(agentbeats_port)
            print(f"[MAIN]Using AgentBeats port: {self.launcher_port}")
        else:
            self.launcher_port = port
            print(f"[MAIN]Using default port: {self.launcher_port}")
        
        # ✅ FIX: Read host from AgentBeats environment variable
        agentbeats_host = os.getenv("HOST")
        if agentbeats_host:
            self.launcher_host = agentbeats_host
            print(f"[MAIN]Using AgentBeats host: {self.launcher_host}")
        else:
            self.launcher_host = "0.0.0.0"
        
        self.auto_start = auto_start
        
        # Agent URLs (from env)
        self.green_host = os.getenv("GREEN_AGENT_HOST", "127.0.0.1")
        self.green_port = int(os.getenv("GREEN_AGENT_PORT", 9000))
        
        self.white_host = os.getenv("WHITE_AGENT_HOST", "127.0.0.1")
        self.white_port = int(os.getenv("WHITE_AGENT_PORT", 8000))
        
        # Load agent cards from TOML files
        self.green_card = self._load_card("cards/green_card.toml")
        self.white_card = self._load_card("cards/white_card.toml")
        
        # Process tracking
        self.green_process: Optional[subprocess.Popen] = None
        self.white_process: Optional[subprocess.Popen] = None
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(title="AgentBeats Launcher", lifespan=lifespan)
        self.app.state.launcher = self
        self._setup_routes()
  
    
    def _load_card(self, card_path: str) -> dict:
        """Load agent card from TOML file"""
        try:
            full_path = Path(__file__).parent / card_path
            if full_path.exists():
                with open(full_path, "rb") as f:
                    card = tomllib.load(f)
                print(f"[MAIN]Loaded card: {full_path}")
                return card
            else:
                print(f"[MAIN]Warning: Card not found: {full_path}")
                return {}
        except Exception as e:
            print(f"[MAIN]Error loading card {card_path}: {e}")
            return {}
                        

    def _setup_routes(self):
        """Setup A2A protocol endpoints"""
        
        @self.app.get("/")
        async def dashboard():
            """Management dashboard"""
            green_status = await self._check_agent_health(
                f"http://{self.green_host}:{self.green_port}"
            )
            white_status = await self._check_agent_health(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AgentBeats Launcher</title>
                <meta http-equiv="refresh" content="10">
                <style>
                    body {{ font-family: Arial; margin: 40px; background: #1a1a1a; color: #fff; }}
                    .container {{ max-width: 900px; margin: 0 auto; background: #2d2d2d; padding: 30px; border-radius: 10px; }}
                    .status {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
                    .running {{ background: #1e4620; border: 2px solid #28a745; }}
                    .stopped {{ background: #4a1f1f; border: 2px solid #dc3545; }}
                    button {{ padding: 12px 24px; margin: 5px; cursor: pointer; border: none; border-radius: 5px; font-size: 14px; font-weight: bold; }}
                    .btn-start {{ background: #28a745; color: white; }}
                    .btn-stop {{ background: #dc3545; color: white; }}
                    .btn-reset {{ background: #ffc107; color: black; }}
                    .btn-card {{ background: #17a2b8; color: white; }}
                    .btn-eval {{ background: #6f42c1; color: white; }}    
                    h1 {{ color: #fff; }}
                    h2 {{ color: #aaa; margin-bottom: 10px; }}
                    .info {{ color: #aaa; font-size: 14px; }}
                    .endpoint {{ background: #1a1a1a; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; color: #0f0; }}
                </style>
                <script>
                    async function sendCommand(endpoint) {{
                        const btn = event.target;
                        btn.disabled = true;
                        btn.textContent = 'Processing...';
                        try {{
                            const response = await fetch(endpoint, {{method: 'POST'}});
                            const data = await response.json();
                            alert(JSON.stringify(data, null, 2));
                            setTimeout(() => location.reload(), 500);
                        }} catch(e) {{
                            alert('Error: ' + e.message);
                        }} finally {{
                            btn.disabled = false;
                            btn.textContent = btn.className.includes('start') ? 'Start' : 
                                             btn.className.includes('stop') ? 'Stop' : 'Reset';
                        }}
                    }}
                    
                    async function showCard() {{
                        const display = document.getElementById('cardDisplay');
                        const btn = event.target;
                        btn.disabled = true;
                        btn.textContent = 'Loading...';
                        try {{
                            const response = await fetch('/card');
                            const card = await response.json();
                            display.textContent = JSON.stringify(card, null, 2);
                            display.style.display = 'block';
                        }} catch(e) {{
                            display.textContent = 'Error: ' + e.message;
                            display.style.display = 'block';
                        }} finally {{
                            btn.disabled = false;
                            btn.textContent = 'Show Agent Card';
                        }}
                    }}
                            
                    async function runEvaluation() {{
                        const resultDiv = document.getElementById('evalResult');
                        const btn = event.target;
                        const numTasks = document.getElementById('numTasks').value || 5;
                        
                        btn.disabled = true;
                        btn.textContent = 'Running...';
                        resultDiv.textContent = 'Starting evaluation...';
                        resultDiv.style.display = 'block';
                        
                        try {{
                            const response = await fetch('/run-eval', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{ num_tasks: parseInt(numTasks) }})
                            }});
                            
                            const data = await response.json();
                            resultDiv.textContent = JSON.stringify(data, null, 2);
                            
                            if (data.status === 'success') {{
                                alert('Evaluation completed! Check results below.');
                            }} else {{
                                alert('Evaluation failed: ' + data.message);
                            }}
                        }} catch(e) {{
                            resultDiv.textContent = 'Error: ' + e.message;
                            alert('Error: ' + e.message);
                        }} finally {{
                            btn.disabled = false;
                            btn.textContent = '🚀 Run Evaluation';
                        }}
                    }}
        
                            
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 AgentBeats Finance Benchmark</h1>
                    <p>Auto-refresh every 10 seconds</p>
                    
                    <div class="status {'running' if green_status else 'stopped'}">
                        <h2>Green Agent (Assessor)</h2>
                        <p>Port: {self.green_port} | Status: {'🟢 Running' if green_status else '🔴 Stopped'}</p>
                    </div>
                    
                    <div class="status {'running' if white_status else 'stopped'}">
                        <h2>White Agent (Executor)</h2>
                        <p>Port: {self.white_port} | Status: {'🟢 Running' if white_status else '🔴 Stopped'}</p>
                    </div>
                    
                    <div style="margin: 30px 0;">
                        <button class="btn-start" onclick="sendCommand('/start')">▶️ Start Agents</button>
                        <button class="btn-stop" onclick="sendCommand('/stop')">⏹️ Stop Agents</button>
                        <button class="btn-reset" onclick="sendCommand('/reset')">🔄 Reset Agents</button>
                        <button class="btn-card" onclick="showCard()">📋 Show Agent Card</button>
                    </div>
            
                    <div style="margin: 30px 0; padding: 20px; background: #1e1e1e; border-radius: 5px;">
                        <h3>Run Evaluation</h3>
                        <label for="numTasks">Number of tasks:</label>
                        <input type="number" id="numTasks" value="5" min="1" max="100">
                        <button class="btn-eval" onclick="runEvaluation()">🚀 Run Evaluation</button>
                    </div>
                    
                    <div id="cardDisplay"></div>
                    <div id="evalResult"></div>
            
                    <div>
                        <h3>Launcher Endpoints (A2A Protocol):</h3>
                        <div class="endpoint">POST /start  → Start agents</div>
                        <div class="endpoint">POST /stop   → Stop agents</div>
                        <div class="endpoint">POST /reset  → Reset & restart agents</div>
                        <div class="endpoint">GET  /health → Health check</div>
                        <div class="endpoint">GET  /status → Detailed status</div>
                        <div class="endpoint">GET  /card  → Get green agent card</div>
                        <div class="endpoint">GET  /.well-known/agent-card.json  → Also get green agent card</div>
                        <div class="endpoint">GET  /run-eval  → Ask green agent to initiate the evaluation/assessment</div>
                    </div>
                    
                    <div id="cardDisplay" style="display: none;"></div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(html)

        """        
        @self.app.get("/white/card")
        async def get_white_card():
            # Return white agent's card
            card = dict(self.white_card)
            
            white_running = await self._check_agent_health(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            card["url"] = f"http://{self.white_host}:{self.white_port}"
            card["status"] = "running" if white_running else "stopped"
            
            return JSONResponse(card)
        """
        
        @self.app.get("/card")
        @self.app.get("/.well-known/agent-card.json")
        async def get_card():
            """
            Return green agent's card from TOML file.
            AgentBeats uses this to discover the agent.
            """
            # Start with the card from TOML file
            card = dict(self.green_card)
            
            # Check if green agent is running
            green_running = await self._check_agent_health(
                f"http://{self.green_host}:{self.green_port}"
            )
            
            # Add runtime information
            card["url"] = f"http://{self.green_host}:{self.green_port}"
            card["status"] = "running" if green_running else "stopped"
            card["launcher"] = {
                "host": self.launcher_host,
                "port": self.launcher_port
            }
            
            # Add endpoints for AgentBeats to call
            card["endpoints"] = {
                "a2a": f"http://{self.green_host}:{self.green_port}/a2a",
                "reset": f"http://{self.green_host}:{self.green_port}/reset",
                "health": f"http://{self.green_host}:{self.green_port}/health"
            }
            
            return JSONResponse(card)
        
        @self.app.post("/start")
        async def start_agents():
            """Start both agents"""
            print("[MAIN]Starting agents...")
            
            # Start green agent
            if not self.green_process or self.green_process.poll() is not None:
                print("[MAIN]Starting green agent...")
                self.green_process = subprocess.Popen(
                    [sys.executable, "green_agent_mcp_a2a_judge_rag.py"],
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                await asyncio.sleep(3)
            
            # Start white agent
            if not self.white_process or self.white_process.poll() is not None:
                print("[MAIN]Starting white agent...")
                self.white_process = subprocess.Popen(
                    [sys.executable, "white_agent_mcp_memory.py"],
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                await asyncio.sleep(3)
            
            # Wait for agents to be ready
            green_ready = await self._wait_agent_ready(
                f"http://{self.green_host}:{self.green_port}",
                timeout=60
            )
            white_ready = await self._wait_agent_ready(
                f"http://{self.white_host}:{self.white_port}",
                timeout=60
            )
            
            return JSONResponse({
                "status": "started",
                "green_agent": {
                    "running": green_ready,
                    "url": f"http://{self.green_host}:{self.green_port}",
                    "pid": self.green_process.pid if self.green_process else None
                },
                "white_agent": {
                    "running": white_ready,
                    "url": f"http://{self.white_host}:{self.white_port}",
                    "pid": self.white_process.pid if self.white_process else None
                }
            })
        
        @self.app.post("/stop")
        async def stop_agents():
            """Stop both agents"""
            print("[MAIN]Stopping agents...")
            
            stopped = []
            
            if self.green_process and self.green_process.poll() is None:
                self.green_process.terminate()
                try:
                    self.green_process.wait(timeout=10)
                    stopped.append("green")
                except subprocess.TimeoutExpired:
                    self.green_process.kill()
                    stopped.append("green (force killed)")
            
            if self.white_process and self.white_process.poll() is None:
                self.white_process.terminate()
                try:
                    self.white_process.wait(timeout=10)
                    stopped.append("white")
                except subprocess.TimeoutExpired:
                    self.white_process.kill()
                    stopped.append("white (force killed)")
            
            return JSONResponse({
                "status": "stopped",
                "stopped_agents": stopped
            })
        
        @self.app.post("/reset")
        async def reset_agents():
            """Reset agents (stop, restart, clear state)"""
            print("[MAIN]========================================")
            print("[MAIN]RESET CALLED - Starting clean assessment")
            print("[MAIN]========================================")
            
            # Stop agents
            await stop_agents()
            await asyncio.sleep(2)
            
            # Restart agents
            return await start_agents()
        
        @self.app.get("/health")
        async def health():
            """Health check"""
            green_ok = await self._check_agent_health(
                f"http://{self.green_host}:{self.green_port}"
            )
            white_ok = await self._check_agent_health(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            return {
                "status": "ok" if (green_ok and white_ok) else "degraded",
                "launcher": "running",
                "green_agent": "running" if green_ok else "stopped",
                "white_agent": "running" if white_ok else "stopped"
            }
        
        @self.app.get("/status")
        async def status():
            """Detailed status"""
            green_ok = await self._check_agent_health(
                f"http://{self.green_host}:{self.green_port}"
            )
            white_ok = await self._check_agent_health(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            return JSONResponse({
                "launcher": {
                    "port": self.launcher_port,
                    "status": "running"
                },
                "green_agent": {
                    "url": f"http://{self.green_host}:{self.green_port}",
                    "status": "running" if green_ok else "stopped",
                    "pid": self.green_process.pid if self.green_process else None
                },
                "white_agent": {
                    "url": f"http://{self.white_host}:{self.white_port}",
                    "status": "running" if white_ok else "stopped",
                    "pid": self.white_process.pid if self.white_process else None
                }
            })
    
        @self.app.post("/run-eval")
        async def run_evaluation(request: Request):
            """
            Trigger evaluation/assessment on green agent.
            Optionally accepts num_tasks in request body.
            """
            try:
                # Get num_tasks from request body if provided
                body = await request.json() if request.headers.get("content-type") == "application/json" else {}
                num_tasks = body.get("num_tasks", 5)  # Default to 5 for testing
                
                print(f"[LAUNCHER] Starting evaluation with {num_tasks} tasks...")
                
                # Check if green agent is running
                green_ok = await self._check_agent_health(
                    f"http://{self.green_host}:{self.green_port}"
                )
                
                if not green_ok:
                    return JSONResponse({
                        "status": "error",
                        "message": "Green agent not running. Start agents first.",
                        "green_agent_url": f"http://{self.green_host}:{self.green_port}"
                    }, status_code=400)
                
                # Call green agent's run_assessment endpoint
                async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout
                    response = await client.post(
                        f"http://{self.green_host}:{self.green_port}/a2a",
                        json={
                            "method": "run_assessment",
                            "args": {
                                "white_address": f"http://{self.white_host}:{self.white_port}",
                                "num_tasks": num_tasks
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"[LAUNCHER] Evaluation completed: {result}")
                        return JSONResponse({
                            "status": "success",
                            "result": result
                        })
                    else:
                        return JSONResponse({
                            "status": "error",
                            "message": f"Green agent returned {response.status_code}",
                            "details": response.text
                        }, status_code=response.status_code)
            
            except Exception as e:
                print(f"[LAUNCHER] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "status": "error",
                    "message": str(e)
                }, status_code=500)
    
    
    async def _wait_agent_ready(self, url: str, timeout: int = 30) -> bool:
        """Wait for agent to respond"""
        async with httpx.AsyncClient() as client:
            for i in range(timeout):
                try:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    if response.status_code == 200:
                        print(f"[MAIN]✅ Agent ready: {url}")
                        return True
                except Exception:
                    try:
                        response = await client.get(f"{url}/card", timeout=5.0)
                        if response.status_code == 200:
                            print(f"[MAIN]✅ Agent ready: {url}")
                            return True
                    except:
                        pass
                
                await asyncio.sleep(1)
            
            print(f"[MAIN]❌ Timeout waiting for: {url}")
            return False
    
    async def _check_agent_health(self, url: str) -> bool:
        """Check if agent is responding"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                return response.status_code == 200
        except:
            return False
    
    def run(self):
        """Start launcher server"""
        print("=" * 60)
        print("AgentBeats Finance Benchmark Launcher")
        print("=" * 60)
        print(f"Launcher: http://{self.launcher_host}:{self.launcher_port}")
        print(f"Green:    http://{self.green_host}:{self.green_port}")
        print(f"White:    http://{self.white_host}:{self.white_port}")
        print(f"Auto-start: {self.auto_start}")
        print("=" * 60)
        
        # Check for port conflict
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', self.launcher_port))
        sock.close()
        
        if result == 0:
            print(f"[MAIN]⚠️  Port {self.launcher_port} already in use!")
            print(f"[MAIN]Attempting to kill existing process...")
            try:
                subprocess.run(f"lsof -ti:{self.launcher_port} | xargs kill -9", 
                              shell=True, check=False)
                time.sleep(2)
            except:
                pass
        
        # Handle shutdown
        def signal_handler(sig, frame):
            print("\n[MAIN]Shutting down...")
            if self.green_process:
                self.green_process.terminate()
            if self.white_process:
                self.white_process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        """
        # Auto-start if requested (for local testing)
        if self.auto_start:
            print("\n[MAIN]Auto-starting agents in 3 seconds...")
            async def delayed_start():
                await asyncio.sleep(3)  # Wait for uvicorn to start
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"http://{self.launcher_host}:{self.launcher_port}/start",
                            timeout=60.0
                        )
                        result = response.json()
                        print(f"[MAIN]Auto-start result: {result}")
                except Exception as e:
                    print(f"[MAIN]Auto-start failed: {e}")
            
            # Run in background
            asyncio.create_task(delayed_start())
        else:
            print("\n[MAIN]Waiting for AgentBeats to call /start...")
        """
        
        # Run launcher server (blocking)
        uvicorn.run(
            self.app,
            host=self.launcher_host,
            port=self.launcher_port,
            log_level="info"
        )


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="AgentBeats Finance Benchmark")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    run_parser = subparsers.add_parser("run", help="Start launcher")
    run_parser.add_argument("--port", type=int, default=7000, help="Launcher port")
    # Fix: Use hyphen in the argument name for consistency
    run_parser.add_argument("--auto-start", "--auto_start", 
                           action="store_true",
                           dest="auto_start",  # Store as auto_start internally
                           help="Auto-start agents (for local testing)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        launcher = AgentBeatsLauncher(
            port=args.port,
            auto_start=args.auto_start
        )
        launcher.run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
