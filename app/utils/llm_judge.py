# app/utils/llm_judge.py
"""
LLM Judge for Answer Evaluation
Handles semantic similarity, numerical equivalence, and partial matches
"""
import os
import json
import litellm
from typing import Dict, Any
from utils.llm_manager import safe_llm_call

class LLMJudge:
    """
    LLM-based judge to evaluate if predicted answer matches expected answer.
    Handles:
    - Semantic equivalence (different wording, same meaning)
    - Numerical values (with tolerance)
    - Partial matches
    - Currency, dates, percentages
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        self.llm_model   = model   or os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key = api_key or os.getenv("LLM_API_KEY")
    
    async def evaluate(
        self, 
        question: str, 
        expected_answer: str, 
        predicted_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if predicted answer matches expected answer.
        
        Returns:
            {
                "correct": bool,
                "score": float (0.0 to 1.0),
                "reasoning": str,
                "match_type": str  # "exact", "semantic", "numerical", "partial", "incorrect"
            }
        """
        
        # Quick exact match check (case-insensitive)
        if expected_answer.strip().lower() == predicted_answer.strip().lower():
            return {
                "correct": True,
                "score": 1.0,
                "reasoning": "Exact string match",
                "match_type": "exact"
            }
        
        # Use LLM for semantic evaluation
        prompt = f"""You are an expert judge evaluating financial question-answering systems.

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

PREDICTED ANSWER: {predicted_answer}

TASK: Determine if the PREDICTED ANSWER is correct compared to the EXPECTED ANSWER.

EVALUATION CRITERIA:
1. **Exact Match**: Same text (case-insensitive) → Score: 1.0
2. **Semantic Match**: Different wording but same meaning → Score: 0.9-1.0
   - Example: "US Steel rejected the merger" ≈ "The merger was rejected by US Steel"
3. **Numerical Match**: Same number (allow ±2% tolerance for rounding) → Score: 0.9-1.0
   - Example: "$3.25 billion" ≈ "$3.25B" ≈ "3250 million"
4. **Partial Match**: Contains key information but incomplete → Score: 0.5-0.8
   - Example: Expected "Merger blocked", Predicted "Nippon Steel merger" (missing key fact)
5. **Incorrect**: Wrong information or missing critical facts → Score: 0.0-0.4

IMPORTANT:
- For numerical answers: Check if numbers are equivalent (consider units: billion vs million)
- For text answers: Check if core facts are present (not just keywords)
- "Not found" or "ERROR" should score 0.0

Respond in JSON format:
{{
    "correct": true/false,
    "score": 0.95,
    "reasoning": "Brief explanation of why correct/incorrect",
    "match_type": "exact|semantic|numerical|partial|incorrect"
}}

Consider the answer CORRECT (true) if score >= 0.8, otherwise INCORRECT (false).
"""
        
        try:
            """
            response = litellm.completion(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise evaluator. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                api_key=self.llm_api_key,
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent evaluation
            )
            """
            response = await safe_llm_call(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise evaluator. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                api_key=self.llm_api_key,
                response_format={"type": "json_object"},
                temperature=0.1  # Lower temp for more focused decisions
                )

            result = json.loads(response.choices[0].message.content)

            # Ensure required fields
            if "correct" not in result:
                result["correct"] = result.get("score", 0.0) >= 0.8
            
            return result
            
        except Exception as e:
            print(f"[JUDGE] LLM evaluation error: {e}")
            # Fallback: strict string comparison
            exact_match = expected_answer.strip().lower() == predicted_answer.strip().lower()
            return {
                "correct": exact_match,
                "score": 1.0 if exact_match else 0.0,
                "reasoning": f"LLM judge failed, used exact match. Error: {str(e)}",
                "match_type": "exact" if exact_match else "incorrect"
            }
    
    def batch_evaluate(
        self, 
        evaluations: list
    ) -> Dict[str, Any]:
        """
        Evaluate multiple question-answer pairs.
        
        Args:
            evaluations: List of dicts with keys: question, expected, predicted
            
        Returns:
            {
                "accuracy": float,
                "total": int,
                "correct": int,
                "scores": [float],
                "details": [dict]
            }
        """
        import asyncio
        
        async def evaluate_all():
            tasks = [
                self.evaluate(
                    item["question"],
                    item["expected"],
                    item["predicted"]
                )
                for item in evaluations
            ]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(evaluate_all())
        
        correct_count = sum(1 for r in results if r["correct"])
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "accuracy": correct_count / len(results) if results else 0.0,
            "average_score": avg_score,
            "total": len(results),
            "correct": correct_count,
            "scores": scores,
            "details": results
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    judge = LLMJudge()
    
    async def test():
        # Test 1: Semantic match
        result1 = await judge.evaluate(
            question="How has US Steel addressed its planned merger with Nippon Steel?",
            expected_answer="The merger was blocked by executive order.",
            predicted_answer="US Steel's merger with Nippon Steel was blocked."
        )
        print("Test 1 (Semantic):", result1)
        
        # Test 2: Numerical match
        result2 = await judge.evaluate(
            question="What was the total consideration cost?",
            expected_answer="$3.25 Billion",
            predicted_answer="3.25 billion dollars"
        )
        print("Test 2 (Numerical):", result2)
        
        # Test 3: Incorrect
        result3 = await judge.evaluate(
            question="What was the merger cost?",
            expected_answer="$3.25 Billion",
            predicted_answer="$5 Billion"
        )
        print("Test 3 (Incorrect):", result3)
    
    asyncio.run(test())
