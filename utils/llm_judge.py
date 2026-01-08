# app/utils/llm_judge.py
"""
LLM Judge for Answer Evaluation
Handles semantic similarity, numerical equivalence, and partial matches
"""
import os
import json
import sys  # ← ADDED THIS IMPORT
import litellm
from typing import Dict, Any
from utils.llm_manager import safe_llm_call
from utils.local_llm_wrapper import safe_local_llm_call 

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
        
        # Check if we should use local LLM: 1=True 0=False
        self.llm_use_local = bool(int(os.getenv("USE_LOCAL_LLM_JUDGE", "1")))

    async def evaluate(
        self,
        question: str,
        expected_answer: str,
        predicted_answer: str
    ) -> dict:
        """
        Evaluate if predicted answer matches expected answer.
        
        Returns:
            {
                "correct": bool,
                "score": float (0.0 to 1.0),
                "match_type": str,
                "reasoning": str
            }
        """
        
        # Handle error cases
        if predicted_answer.startswith("Error:") or predicted_answer == "error_generating_answer":
            return {
                "correct": False,
                "score": 0.0,
                "match_type": "error",
                "reasoning": "White agent failed to generate answer"
            }
        
        # Normalize answers
        expected_clean = expected_answer.strip().lower()
        predicted_clean = predicted_answer.strip().lower()
        
        # Exact match
        if expected_clean == predicted_clean:
            return {
                "correct": True,
                "score": 1.0,
                "match_type": "exact",
                "reasoning": "Exact match"
            }
        
        # Fuzzy match (80%+ similarity)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, expected_clean, predicted_clean).ratio()
        
        if similarity >= 0.8:
            return {
                "correct": True,
                "score": similarity,
                "match_type": "fuzzy",
                "reasoning": f"High similarity ({similarity:.2%})"
            }
        
        # LLM-based semantic evaluation
        prompt = f"""Compare these two answers and determine if they are semantically equivalent.

    Question: {question}

    Expected Answer: {expected_answer}
    Predicted Answer: {predicted_answer}

    Respond with ONLY a JSON object:
    {{
        "equivalent": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }}"""

        try:
            if self.llm_use_local:
                response = await safe_local_llm_call(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    component="judge" 
                )
            else:
                response = await safe_llm_call(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.llm_api_key,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            
            # ✅ CHECK: Is this an error dict or real response?
            if isinstance(response, dict) and "error" in response:
                # LLM call failed, fall back to exact match
                print(f"[JUDGE] LLM evaluation failed: {response['error']}", file=sys.stderr)
                print(f"[JUDGE] Falling back to exact match", file=sys.stderr)
                
                return {
                    "correct": expected_clean == predicted_clean,
                    "score": 1.0 if expected_clean == predicted_clean else 0.0,
                    "match_type": "exact_fallback",
                    "reasoning": f"LLM judge failed ({response['error']}), used exact match"
                }
            
            # ✅ CHECK: Does response have 'choices' attribute?
            if not hasattr(response, 'choices'):
                print(f"[JUDGE] Invalid response format: {type(response)}", file=sys.stderr)
                
                return {
                    "correct": expected_clean == predicted_clean,
                    "score": 1.0 if expected_clean == predicted_clean else 0.0,
                    "match_type": "exact_fallback",
                    "reasoning": "LLM response invalid, used exact match"
                }
            
            # Parse LLM response
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                "correct": result.get("equivalent", False),
                "score": result.get("confidence", 0.0),
                "match_type": "llm_semantic",
                "reasoning": result.get("reasoning", "LLM evaluation")
            }
        
        except Exception as e:
            print(f"[JUDGE] LLM evaluation error: {e}", file=sys.stderr)
            
            # Fall back to exact match
            return {
                "correct": expected_clean == predicted_clean,
                "score": 1.0 if expected_clean == predicted_clean else 0.0,
                "match_type": "exact_fallback",
                "reasoning": f"LLM judge failed, used exact match. Error: {str(e)[:50]}"
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
