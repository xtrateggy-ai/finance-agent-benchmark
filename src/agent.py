# src/agent.py
# Business logic only

import os
import pandas as pd
from utils.llm_judge import LLMJudge


class GreenAgent:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_df = None
        self.current_task_index = 0
        self.assessment_history = []

        self.llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key = os.getenv("LLM_API_KEY")

        self.judge = LLMJudge(
            model=self.llm_model,
            api_key=self.llm_api_key
        )

    def load_dataset(self):
        self.dataset_df = pd.read_csv(self.dataset_path)
        self.dataset_df.columns = self.dataset_df.columns.str.lower()

    def reset(self):
        self.current_task_index = 0
        self.assessment_history.clear()
        self.load_dataset()

    async def evaluate(self, question, expected, predicted):
        return await self.judge.evaluate(
            question=question,
            expected_answer=expected,
            predicted_answer=predicted
        )