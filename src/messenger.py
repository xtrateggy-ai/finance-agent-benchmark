# Handles communication (white agent + MCP endpoint passing)
# src/messenger.py

import httpx
import uuid
import json

class Messenger:

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url

    async def ask_white_agent(self, white_address: str, question: str):

        payload_text = json.dumps(
            {
                "question": question,
                "mcp_url": self.mcp_url,
            }
        )

        request_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": payload_text,
                        }
                    ],
                }
            },
        }
        
        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                white_address.rstrip("/"),
                json=request_payload,
            )
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                raise Exception(f"A2A error: {data['error']}")

            return data.get("result", {})