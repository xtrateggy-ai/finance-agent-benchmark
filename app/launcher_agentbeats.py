#!/usr/bin/env python
"""
AgentBeats-Compatible Launcher
Implements A2A protocol for agent lifecycle management

This launcher:
- Receives commands from AgentBeats platform
- Manages green and white agent processes
- Provides health monitoring and logging
- Implements /reset, /start, /stop, /status endpoints
"""

import os
import sys
import asyncio
import signal
import multiprocessing
import json
import time
import httpx
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.env_setup import init_environment

init_environment()


def run_green_agent():
    """Standalone function to run green agent (picklable)"""
    from green_agent_mcp_a2a_judge import GreenAgent
    agent = GreenAgent()
    agent.run()


def run_white_agent():
    """Standalone function to run white agent (picklable)"""
    from white_agent_mcp_memory import WhiteAgent
    agent = WhiteAgent()
    agent.run()


class AgentBeatsLauncher:
    """
    AgentBeats-compatible launcher that manages agent lifecycle.
    
    Implements A2A protocol endpoints:
    - POST /reset - Reset and restart agents
    - POST /start - Start agents
    - POST /stop  - Stop agents
    - GET  /health - Health check
    - GET  /status - Detailed status
    - GET  /card - Launcher card
    """
    
    def __init__(self):
        self.launcher_host = os.getenv("LAUNCHER_HOST", "0.0.0.0")
        self.launcher_port = int(os.getenv("LAUNCHER_PORT", 7000))
        
        self.green_host = os.getenv("GREEN_AGENT_HOST", "127.0.0.1")
        self.green_port = int(os.getenv("GREEN_AGENT_PORT", 9000))
        
        self.white_host = os.getenv("WHITE_AGENT_HOST", "127.0.0.1")
        self.white_port = int(os.getenv("WHITE_AGENT_PORT", 8000))
        
        # Process tracking
        self.green_process: Optional[multiprocessing.Process] = None
        self.white_process: Optional[multiprocessing.Process] = None
        
        # Create FastAPI app
        self.app = FastAPI(title="AgentBeats Launcher")
        self._setup_routes()
    
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
                <style>
                    body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                    .status {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
                    .running {{ background: #d4edda; border: 1px solid #28a745; }}
                    .stopped {{ background: #f8d7da; border: 1px solid #dc3545; }}
                    button {{ padding: 12px 24px; margin: 5px; cursor: pointer; border: none; border-radius: 5px; font-size: 14px; }}
                    .btn-start {{ background: #28a745; color: white; }}
                    .btn-stop {{ background: #dc3545; color: white; }}
                    .btn-reset {{ background: #ffc107; color: black; }}
                    .btn-refresh {{ background: #007bff; color: white; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    .endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }}
                </style>
                <script>
                    async function sendCommand(endpoint) {{
                        try {{
                            const response = await fetch(endpoint, {{method: 'POST'}});
                            const data = await response.json();
                            alert(JSON.stringify(data, null, 2));
                            setTimeout(() => location.reload(), 1000);
                        }} catch(e) {{
                            alert('Error: ' + e.message);
                        }}
                    }}
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 AgentBeats Launcher</h1>
                    
                    <div class="status {'running' if green_status else 'stopped'}">
                        <h2>Green Agent (Assessor)</h2>
                        <p>Status: {'🟢 Running' if green_status else '🔴 Stopped'}</p>
                        <p>URL: http://{self.green_host}:{self.green_port}</p>
                        <p>Endpoints: /a2a, /reset, /health, /card</p>
                    </div>
                    
                    <div class="status {'running' if white_status else 'stopped'}">
                        <h2>White Agent (Executor)</h2>
                        <p>Status: {'🟢 Running' if white_status else '🔴 Stopped'}</p>
                        <p>URL: http://{self.white_host}:{self.white_port}</p>
                        <p>Endpoints: /a2a, /health, /card</p>
                    </div>
                    
                    <div style="margin: 30px 0;">
                        <button class="btn-start" onclick="sendCommand('/start')">▶️ Start Agents</button>
                        <button class="btn-stop" onclick="sendCommand('/stop')">⏹️ Stop Agents</button>
                        <button class="btn-reset" onclick="sendCommand('/reset')">🔄 Reset Agents</button>
                        <button class="btn-refresh" onclick="location.reload()">🔃 Refresh</button>
                    </div>
                    
                    <div>
                        <h3>Launcher Endpoints (A2A Protocol):</h3>
                        <div class="endpoint">POST /start - Start agents</div>
                        <div class="endpoint">POST /stop - Stop agents</div>
                        <div class="endpoint">POST /reset - Reset and restart agents</div>
                        <div class="endpoint">GET /health - Health check</div>
                        <div class="endpoint">GET /status - Detailed status</div>
                        <div class="endpoint">GET /card - Launcher card</div>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(html)
        
        @self.app.get("/card")
        @self.app.get("/.well-known/agent-card.json")
        async def get_card():
            """Return launcher card (A2A requirement)"""
            return JSONResponse({
                "name": "Finance Agent Launcher",
                "description": "Launcher for finance agent benchmark",
                "version": "1.0.0",
                "capabilities": {
                    "agent_management": True,
                    "lifecycle_control": True,
                    "health_monitoring": True
                },
                "agents": {
                    "green": f"http://{self.green_host}:{self.green_port}",
                    "white": f"http://{self.white_host}:{self.white_port}"
                }
            })
        
        @self.app.post("/start")
        async def start_agents():
            """Start both agents"""
            print("[LAUNCHER] Starting agents...")
            
            # Start green agent
            if not self.green_process or not self.green_process.is_alive():
                self.green_process = multiprocessing.Process(
                    target=run_green_agent,
                    daemon=True
                )
                self.green_process.start()
                await asyncio.sleep(3)  # Wait for green to start
            
            # Start white agent
            if not self.white_process or not self.white_process.is_alive():
                self.white_process = multiprocessing.Process(
                    target=run_white_agent,
                    daemon=True
                )
                self.white_process.start()
                await asyncio.sleep(2)  # Wait for white to start
            
            # Verify agents started
            green_ready = await self._wait_agent_ready(
                f"http://{self.green_host}:{self.green_port}"
            )
            white_ready = await self._wait_agent_ready(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            return JSONResponse({
                "status": "started",
                "green_agent": {
                    "running": green_ready,
                    "url": f"http://{self.green_host}:{self.green_port}"
                },
                "white_agent": {
                    "running": white_ready,
                    "url": f"http://{self.white_host}:{self.white_port}"
                }
            })
        
        @self.app.post("/stop")
        async def stop_agents():
            """Stop both agents"""
            print("[LAUNCHER] Stopping agents...")
            
            stopped = []
            
            if self.green_process and self.green_process.is_alive():
                self.green_process.terminate()
                self.green_process.join(timeout=10)
                stopped.append("green")
            
            if self.white_process and self.white_process.is_alive():
                self.white_process.terminate()
                self.white_process.join(timeout=10)
                stopped.append("white")
            
            return JSONResponse({
                "status": "stopped",
                "stopped_agents": stopped
            })
        
        @self.app.post("/reset")
        async def reset_agents():
            """
            Reset agents (AgentBeats requirement)
            
            This is called by AgentBeats before each assessment to ensure
            clean state. Steps:
            1. Stop all agents
            2. Send /reset to green agent endpoint (when restarted)
            3. Restart agents
            """
            print("[LAUNCHER] Resetting agents...")
            
            # Stop agents
            await stop_agents()
            await asyncio.sleep(1)
            
            # Start agents
            start_result = await start_agents()
            
            # Send reset to green agent
            if start_result["green_agent"]["running"]:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    try:
                        response = await client.post(
                            f"http://{self.green_host}:{self.green_port}/reset"
                        )
                        reset_data = response.json()
                        
                        return JSONResponse({
                            "status": "reset_complete",
                            "agents_restarted": True,
                            "green_agent_reset": reset_data
                        })
                    except Exception as e:
                        return JSONResponse({
                            "status": "reset_partial",
                            "agents_restarted": True,
                            "green_agent_reset": False,
                            "error": str(e)
                        })
            
            return JSONResponse({
                "status": "reset_failed",
                "error": "Failed to start agents"
            })
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
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
            """Detailed status information"""
            green_ok = await self._check_agent_health(
                f"http://{self.green_host}:{self.green_port}"
            )
            white_ok = await self._check_agent_health(
                f"http://{self.white_host}:{self.white_port}"
            )
            
            return JSONResponse({
                "launcher": {
                    "host": self.launcher_host,
                    "port": self.launcher_port,
                    "status": "running"
                },
                "green_agent": {
                    "url": f"http://{self.green_host}:{self.green_port}",
                    "status": "running" if green_ok else "stopped",
                    "process_alive": self.green_process.is_alive() if self.green_process else False
                },
                "white_agent": {
                    "url": f"http://{self.white_host}:{self.white_port}",
                    "status": "running" if white_ok else "stopped",
                    "process_alive": self.white_process.is_alive() if self.white_process else False
                }
            })
    
    def _start_green_agent(self):
        """Start green agent process - REMOVED, using standalone function"""
        pass
    
    def _start_white_agent(self):
        """Start white agent process - REMOVED, using standalone function"""
        pass
    
    async def _wait_agent_ready(self, url: str, timeout: int = 30) -> bool:
        """Wait for agent to be ready"""
        async with httpx.AsyncClient() as client:
            for i in range(timeout):
                try:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    if response.status_code == 200:
                        print(f"[LAUNCHER] ✓ Agent ready: {url}")
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1)
            print(f"[LAUNCHER] ✗ Timeout waiting for: {url}")
            return False
    
    async def _check_agent_health(self, url: str) -> bool:
        """Check if agent is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                return response.status_code == 200
        except:
            return False
    
    def run(self):
        """Start launcher server"""
        print("=" * 60)
        print("AgentBeats Launcher")
        print("=" * 60)
        print(f"Launcher URL: http://{self.launcher_host}:{self.launcher_port}")
        print(f"Green Agent:  http://{self.green_host}:{self.green_port}")
        print(f"White Agent:  http://{self.white_host}:{self.white_port}")
        print("=" * 60)
        
        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            print("\n[LAUNCHER] Shutting down...")
            if self.green_process:
                self.green_process.terminate()
            if self.white_process:
                self.white_process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Auto-start agents
        print("\n[LAUNCHER] Auto-starting agents...")
        asyncio.run(self._auto_start())
        
        # Run launcher server (blocking)
        uvicorn.run(
            self.app,
            host=self.launcher_host,
            port=self.launcher_port,
            log_level="info"
        )
    
    async def _auto_start(self):
        """Auto-start agents on launcher startup"""
        # Start green agent
        self.green_process = multiprocessing.Process(
            target=run_green_agent,
            daemon=True
        )
        self.green_process.start()
        await asyncio.sleep(3)
        
        # Start white agent
        self.white_process = multiprocessing.Process(
            target=run_white_agent,
            daemon=True
        )
        self.white_process.start()
        await asyncio.sleep(2)
        
        # Verify
        green_ready = await self._wait_agent_ready(
            f"http://{self.green_host}:{self.green_port}"
        )
        white_ready = await self._wait_agent_ready(
            f"http://{self.white_host}:{self.white_port}"
        )
        
        if green_ready and white_ready:
            print("\n[LAUNCHER] ✓ All agents started successfully\n")
        else:
            print("\n[LAUNCHER] ⚠ Some agents failed to start\n")


if __name__ == "__main__":
    # CRITICAL: This guard is required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    launcher = AgentBeatsLauncher()
    launcher.run()
