# Bridges A2A → agent logic
# src/executer.py

import os
from src.agent import GreenAgent
from src.messenger import Messenger


class Executer:

    def __init__(self, mcp_url: str | None = None):
        dataset = os.getenv("DATASET", "data/public.csv")
        host = os.getenv("HOST", "0.0.0.0")
        mcp_port = os.getenv("MCP_PORT", "9001")

        if mcp_url is None:
            host = os.getenv("HOST", "0.0.0.0")
            mcp_port = os.getenv("MCP_PORT", "9001")
            mcp_url = f"http://{host}:{mcp_port}"

        self.mcp_url = mcp_url

        self.agent = GreenAgent(dataset_path=dataset)
        self.messenger = Messenger(self.mcp_url)

        self.agent.load_dataset()

    async def run_assessment(self, white_address: str, num_tasks: int = 1):

        results = []
        correct = 0

        start = self.agent.current_task_index
        end = min(start + num_tasks, len(self.agent.dataset_df))

        for idx in range(start, end):
            row = self.agent.dataset_df.iloc[idx]

            question = row["question"]
            expected = row["answer"].strip().lower()

            white_response = await self.messenger.ask_white_agent(
                white_address,
                question
            )

            predicted = white_response.get("answer", "").strip().lower()

            evaluation = await self.agent.evaluate(
                question,
                expected,
                predicted
            )

            if evaluation["correct"]:
                correct += 1

            results.append(evaluation)

        self.agent.current_task_index = end

        accuracy = correct / len(results) if results else 0.0

        return {
            "metric": "accuracy",
            "value": accuracy,
            "total_tasks": len(results),
            "correct_tasks": correct
        }