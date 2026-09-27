# This is the only file that knows about:

# FastAPI

# A2A

# MCP

# Agent card

# src/server.py

import os
import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import json
import uuid

import uvicorn
from fastapi import FastAPI

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import RequestHandler

from a2a.types import (
    MessageSendParams,
    Message,
    Task,
    TaskStatus,
    TaskState,
    Artifact,
    AgentCard,
    AgentCapabilities,
    AgentSkill
)
from a2a.utils.errors import ServerError

from src.executer import Executer
from fastmcp import FastMCP
from src.mcp_tools import register_mcp_tools


# ============================================================
# Request Handler (JSON-RPC)
# ============================================================

class GreenRequestHandler(RequestHandler):

    def __init__(self, executer: Executer):
        self.executer = executer

    async def on_message_send(
        self,
        params: MessageSendParams,
        context=None):
        # ----------------------------------------------------
        # 1. Read incoming assessment request
        # ----------------------------------------------------
        part = params.message.parts[0].root
        text = part.text

        try:
            payload = json.loads(text)
        except Exception:
            payload = {}

        # ----------------------------------------------------
        # 2. Resolve Purple/White agent address
        # ----------------------------------------------------
        white_address = payload.get("white_address")
        num_tasks = payload.get("num_tasks", 1)

        if not white_address and context:
            white_address = getattr(
                context,
                "white_address",
                None,
            )

        if not white_address:
            white_address = os.getenv(
                "WHITE_ADDRESS",
                "http://finance-purple-agent:9009",
            )

        if not white_address:
            raise ValueError(
                f"Missing white_address. Payload received: {text}"
            )

        # ----------------------------------------------------
        # 3. Run assessment
        # ----------------------------------------------------
        result = await self.executer.run_assessment(
            white_address=white_address,
            num_tasks=num_tasks,
        )


        print(
            f"[GREEN][ASSESSMENT] Result: "
            f"{json.dumps(result)}")
        
        return Message(
            role="assistant",
            content=result
        )

    # Required by abstract base class
    async def on_message_send_stream(
        self,
        params,
        context=None
    ) -> AsyncGenerator:

        raise ServerError(error=UnsupportedOperationError())
    
    async def on_get_task(self, params, context=None):
        return None


    async def on_cancel_task(self, params, context=None):
        return None


    async def on_set_task_push_notification_config(self, params, context=None):
        return params


    async def on_get_task_push_notification_config(self, params, context=None):
        return None


    async def on_list_task_push_notification_config(self, params, context=None):
        return []


    async def on_delete_task_push_notification_config(self, params, context=None):
        return None


    async def on_resubscribe_to_task(self, params, context=None):
        raise ServerError(error=UnsupportedOperationError())

# ============================================================
# GreenAgent Wrapper (AgentBeats Compatible)
# ============================================================

class GreenAgent:

    def __init__(self):
        self.agent_host = "0.0.0.0"
        self.agent_port = 9009
        self.mcp_port = 9001
        self.card_url = None
        


    def create_agent_card(self):

        skill = AgentSkill(
            id="finance-analysis",
            name="Finance Analysis",
            description="Analyze financial filings using MCP tools",
            input_modes=["text/plain"],
            output_modes=["text/plain"],
            tags = ["AI Benchmark", "Evaluating", "mcp", "finance", "tool calling"]
        )

        return AgentCard(
            name="Green Finance Agent",
            description="Finance benchmark agent",
            version="1.0.0",

            # REQUIRED
            url=self.card_url or f"http://localhost:{self.agent_port}/",
            # REQUIRED
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],

            capabilities=AgentCapabilities(
                streaming=False,
                push_notifications=False
            ),

            # REQUIRED
            skills=[skill]
        )

    def create_mcp(self):

        mcp = FastMCP("finance-tools")
        register_mcp_tools(mcp)
        # @mcp.tool()
        # async def health_check():
        #     return {"status": "ok"}

        return mcp

    def create_app(self):

        executer = Executer(
            mcp_url=os.getenv("MCP_URL", f"http://green-agent:{self.mcp_port}/sse")
        )

        handler = GreenRequestHandler(executer)

        mcp = self.create_mcp()

        @asynccontextmanager
        async def lifespan(app: FastAPI):

            mcp_task = asyncio.create_task(
                mcp.run_async(
                    transport="sse",
                    host=self.agent_host,
                    port=self.mcp_port,
                )
            )

            print(f"A2A running on {self.agent_host}:{self.agent_port}")
            print(f"MCP running on {self.agent_host}:{self.mcp_port}")

            yield

            mcp_task.cancel()

        # create base app
        app = FastAPI(lifespan=lifespan)
        agent_card = self.create_agent_card()
        
        a2a_app = A2AStarletteApplication(
            agent_card = agent_card,
            http_handler=handler,
        )

        a2a_app.add_routes_to_app(app)

        return app

    def run(self):

        app = self.create_app()

        uvicorn.run(
            app,
            host=self.agent_host,
            port=self.agent_port,
        )


# ============================================================
# CLI ENTRYPOINT (AgentBeats Requirement)
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finance Green Agent")

    parser.add_argument(
        "--host",
        default=os.getenv("GREEN_AGENT_HOST", "0.0.0.0"),
        help="Host to bind the A2A server"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GREEN_AGENT_PORT", 9009)),
        help="Port for the A2A server"
    )

    parser.add_argument(
        "--mcp-port",
        type=int,
        default=int(os.getenv("MCP_PORT", 9001)),
        help="Port for the MCP server"
    )

    parser.add_argument(
        "--card-url",
        type=str,
        help="Public URL where this agent's card is served"
    )

    args = parser.parse_args()

    agent = GreenAgent()

    # 🔥 REQUIRED FOR AGENTBEATS
    agent.agent_host = args.host
    agent.agent_port = args.port
    agent.mcp_port = args.mcp_port
    agent.card_url = args.card_url

    agent.run()