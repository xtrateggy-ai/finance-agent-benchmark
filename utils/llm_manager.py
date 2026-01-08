# app/utils/llm_manager.py
"""
Rate-Limited LLM Manager with Free Fallbacks
Handles 429 errors and provides free alternatives via HuggingFace
"""
import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import litellm
from functools import wraps

from utils.env_setup import init_environment

init_environment()


class RateLimitTracker:
    """Track API calls and enforce rate limits"""
    
    def __init__(self):
        self.calls = {}  # model -> list of timestamps
        self.limits = {
            # Gemini free tier: ~15 RPM (requests per minute)
            "gemini": {"rpm": 15, "tpm": 1_000_000},  # tokens per minute
            # OpenAI free tier: ~3 RPM
            "openai": {"rpm": 3, "tpm": 40_000},
            # Anthropic free tier: ~5 RPM
            "anthropic": {"rpm": 5, "tpm": 100_000},
            # HuggingFace Inference API: ~1000 RPM (very generous)
            "huggingface": {"rpm": 1000, "tpm": 10_000_000}
        }
    
    def can_call(self, provider: str) -> bool:
        """Check if we can make a call without hitting rate limit"""
        if provider not in self.limits:
            return True
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old timestamps
        if provider in self.calls:
            self.calls[provider] = [
                ts for ts in self.calls[provider] 
                if ts > one_minute_ago
            ]
        else:
            self.calls[provider] = []
        
        # Check if under limit
        rpm_limit = self.limits[provider]["rpm"]
        return len(self.calls[provider]) < rpm_limit
    
    def record_call(self, provider: str):
        """Record a successful call"""
        if provider not in self.calls:
            self.calls[provider] = []
        self.calls[provider].append(datetime.now())
    
    def get_wait_time(self, provider: str) -> float:
        """Get seconds to wait before next call"""
        if provider not in self.limits:
            return 0
        
        if not self.calls.get(provider):
            return 0
        
        now = datetime.now()
        oldest_call = min(self.calls[provider])
        time_since_oldest = (now - oldest_call).total_seconds()
        
        if time_since_oldest < 60:
            return 60 - time_since_oldest
        return 0


class LLMManager:
    """
    Manages LLM calls with rate limiting and free fallbacks.
    
    Priority order:
    1. Gemini (if not rate-limited)
    2. HuggingFace Inference API (free, high limits)
    3. Local cache (if same prompt seen before)
    """
    
    def __init__(self):
        self.rate_tracker = RateLimitTracker()
        self.cache = {}  # Simple in-memory cache
        self.primary_llm_model   = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.primary_llm_api_key = os.getenv("LLM_API_KEY")
        self.secondary_llm_model = os.getenv("HF_LLM_MODEL","huggingface/Qwen/Qwen3-Next-80B-A3B-Instruct")  
        self.secondary_llm_api_key   = os.getenv("HF_LLM_API_KEY")
        
        # Free HuggingFace models (no API key required for inference)
        #self.free_models = [
        #    "huggingface/mistralai/Mistral-7B-Instruct-v0.2",
        #    "huggingface/microsoft/phi-2",
        #    "huggingface/google/flan-t5-large"
        #]
    
    def _get_provider(self, model: str) -> str:
        """Extract provider from model string"""
        if "gemini" in model.lower():
            return "gemini"
        elif "gpt" in model.lower():
            return "openai"
        elif "claude" in model.lower():
            return "anthropic"
        elif "huggingface" in model.lower() or "hf/" in model.lower():
            return "huggingface"
        return "unknown"
    
    def _get_cache_key(self, messages: List[Dict], model: str) -> str:
        """Generate cache key from messages"""
        content = json.dumps([m.get("content", "") for m in messages])
        return f"{model}:{hash(content)}"
    
    async def completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        response_format: Dict = {"type": "json_object"},
        temperature: float = 0.1,
        max_tokens: int = 1000,
        #json_mode: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
    #) -> Dict[str, Any]:
        """
        Make LLM completion with rate limiting and fallbacks.
        
        Returns:
            {
                "content": "response text",
                "model_used": "actual model that responded",
                "from_cache": bool,
                "provider": "gemini|huggingface|etc"
            }
        """
        model = model or self.primary_llm_model
        provider = self._get_provider(model)
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(messages, model)
            if cache_key in self.cache:
                print(f"[LLM] Cache hit for {model}")
                return {
                    "content": self.cache[cache_key],
                    "model_used": model,
                    "from_cache": True,
                    "provider": provider
                }
        
        # Try primary model with rate limiting
        if self.rate_tracker.can_call(provider):
            try:
                print(f"[LLM_MANAGER] completion(): model={model}")
                response = await self._call_llm(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    #json_mode=json_mode,
                    **kwargs
                )
                
                self.rate_tracker.record_call(provider)
                
                content = response.choices[0].message.content
                
                # Cache result
                if use_cache:
                    self.cache[cache_key] = content
                """
                return {
                    "content": content,
                    "model_used": model,
                    "from_cache": False,
                    "provider": provider
                }
                """
                return response 
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if rate limit error
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    print(f"[LLM_MANAGER] Rate limit hit for {model}, falling back...")
                else:
                    print(f"[LLM_MANAGER] Error with {model}: {e}")
        else:
            wait_time = self.rate_tracker.get_wait_time(provider)
            print(f"[LLM_MANAGER] Rate limit: wait {wait_time:.1f}s for {provider}")
        
        # Fallback to HuggingFace (free, high limits)
        print(f"[LLM_MANAGER] Using secondary model={self.secondary_llm_model} fallback", file=sys.stderr)
        response = await self._huggingface_completion(
            messages=messages,
            model=self.secondary_llm_model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens
            #json_mode=json_mode
        )
    
        """
        content = response.choices[0].message.content
        return {
            "content": content,
            "model_used": self.secondary_llm_model,
            "from_cache": False,
            "provider": provider
        } 
        """
        return response
    
    async def _call_llm(
        self,

        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        #json_mode: bool,
        response_format: Dict = {"type": "text"},
        **kwargs
    ):
        """Call LLM via LiteLLM"""
        params = {
            "model": model,
            "messages": messages,
            "api_key": self.primary_llm_api_key,
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        #if json_mode:
        #    params["response_format"] = {"type": "json_object"}
        print(f"[LLM_MANAGER] _call_llm(): model={model}")

        response = litellm.completion(**params)
        return response
    
    async def _huggingface_completion(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        response_format: Dict = {"type": "text"},
        #json_mode: bool
        **kwargs
    ):
        """
        Fallback to free HuggingFace Inference API.
        
        NO API KEY REQUIRED for public models!
        Rate limit: ~1000 RPM (very generous)
        """
       
        #litellm._turn_on_debug()
        
        # Format prompt for instruction-tuned model
        #prompt = self._format_for_instruct(messages, json_mode)
        
        print(f"[LLM_MANAGER] _huggingface_completion() model={model}", file=sys.stderr)
        
        try:
            """Call LLM via LiteLLM"""
            params = {
                "model": model,
                "messages": messages,
                "api_key": self.secondary_llm_api_key,
                "temperature": temperature,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "api_base":"https://router.huggingface.co",
                **kwargs
            }
            
            """
            # LiteLLM supports HuggingFace Inference API
            response = litellm.completion(
                model=model,
                #messages=[{"role": "user", "content": prompt}],
                messages=messages,
                api_key=self.secondary_llm_api_key,  # Can be None for public models
                temperature=temperature,
                max_tokens=max_tokens,
                #api_base="https://api-inference.huggingface.co/models"
                
            )
            
            content = response.choices[0].message.content
            
             If JSON mode, try to clean response
            if json_mode:
                content = self._extract_json(content)
            """
            response = litellm.completion(**params)
            
            return response
            """
            return {
                "content": content,
                "model_used": model,
                "from_cache": False,
                "provider": "huggingface"
            }
            """
        except Exception as e:
            print(f"[LLM_MANAGER] HuggingFace error: {e}")
            
            # Last resort: return error message
            return {
                "content": "ERROR: All LLM providers failed",
                "model_used": "none",
                "from_cache": False,
                "provider": "error",
                "error": str(e)
            }
    
    def _format_for_instruct(self, messages: List[Dict], json_mode: bool) -> str:
        """Format messages for instruction-tuned models"""
        # Combine all messages into single prompt
        parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(f"<SYSTEM>\n{content}\n</SYSTEM>")
            elif role == "user":
                parts.append(f"<USER>\n{content}\n</USER>")
            elif role == "assistant":
                parts.append(f"<ASSISTANT>\n{content}\n</ASSISTANT>")
        
        prompt = "\n\n".join(parts)
        
        if json_mode:
            prompt += "\n\n<ASSISTANT>\nRespond ONLY with valid JSON, no explanations:\n{"
        
        return prompt
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from model output"""
        import re
        
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return original
        return text

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extract JSON from Gemini's verbose responses.
        Handles cases where Gemini adds explanations before/after JSON.
        """
        import re
        import json
        
        # Try direct parse first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', response_text)
        text = re.sub(r'```\s*$', '', text)
        
        # Find JSON object (handles nested braces)
        match = re.search(
            r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}',
            text,
            re.DOTALL
        )
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"[WHITE] JSON parse error: {e}")
                print(f"[WHITE] Attempted to parse: {match.group(0)[:200]}...")
        
        # Last resort: return error dict
        return {
            "action": "answer",
            "answer": "ERROR: Failed to parse LLM response",
            "reasoning": f"Response was not valid JSON: {response_text[:100]}..."
        }   



# Singleton instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get or create LLM manager singleton"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

"""
async def safe_llm_call(
    messages: List[Dict],
    model: Optional[str] = None,
    response_format: Dict = {"type": "json_object"},
    **kwargs
    ):
    #) -> Dict[str, Any]:
    
    #Safe LLM call with automatic fallbacks.
    #
    #Usage:
    #    result = await safe_llm_call([
    #        {"role": "user", "content": "What is revenue?"}
    #    ])
    #    
    #    answer = result["content"]
    #    model_used = result["model_used"]
    #
    manager = get_llm_manager()
    response = await manager.completion(messages=messages, 
                                    model=model, 
                                    response_format=response_format, 
                                    **kwargs)
    return response
"""

#---------------------------------------
async def safe_llm_call(model, messages, api_key=None, **kwargs):
    """
    Safe wrapper for litellm.completion() that handles errors gracefully.
    
    Returns:
        - Normal response object on success
        - Error dict {"error": "...", "status": "failed"} on failure
    """
    import litellm
    
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            **kwargs
        )
        return response
    
    except litellm.APIError as e:
        # API-level errors (rate limits, auth, etc.)
        error_msg = str(e)
        print(f"[LLM_MANAGER] API error: {error_msg}", file=sys.stderr)
        
        # Check for specific error types
        if "rate limit" in error_msg.lower() or "usage limit" in error_msg.lower():
            return {
                "error": "Rate limit exceeded",
                "status": "failed",
                "details": error_msg
            }
        elif "auth" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "error": "Authentication failed",
                "status": "failed",
                "details": error_msg
            }
        else:
            return {
                "error": "API error",
                "status": "failed",
                "details": error_msg
            }
    
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[LLM_MANAGER] Unexpected error: {e}", file=sys.stderr)
        return {
            "error": "Unexpected error",
            "status": "failed",
            "details": str(e)
        }
    

# Example usage
async def test_rate_limiting():
    """Test rate limiting and fallbacks"""
    
    print("Test 1: Normal call (should use Gemini)")
    result1 = await safe_llm_call([
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ])
    print(f"Result: {result1['content']}")
    print(f"Model: {result1['model_used']}")
    print(f"Provider: {result1['provider']}")
    
    print("\n" + "="*60 + "\n")
    
    print("Test 2: Rapid-fire calls (should hit rate limit and fallback)")
    for i in range(20):
        result = await safe_llm_call([
            {"role": "user", "content": f"What is {i}+1? Answer in one word."}
        ])
        print(f"Call {i+1}: {result['content'][:20]} via {result['provider']}")
        
        if i % 5 == 0:
            await asyncio.sleep(0.1)  # Small delay
    
    print("\n" + "="*60 + "\n")
    
    print("Test 3: JSON mode with fallback")
    result3 = await safe_llm_call([
        {"role": "user", "content": "List 3 colors as JSON array"}
    ], json_mode=True)
    print(f"Result: {result3['content']}")
    print(f"Model: {result3['model_used']}")

async def test_hf():
    manager = LLMManager()
    #model="huggingface/sambanova/meta-llama/Llama-3.3-70B-Instruct"
    #model="huggingface/upstage/solar-10.7b-instruct-v1.0" 
    
    
    result = await manager._huggingface_completion(
            model="huggingface/Qwen/Qwen3-Next-80B-A3B-Instruct",
            #model = "huggingface/MiniMaxAI/MiniMax-M2",
            messages=[{"role": "user", "content": "What is 2+2? Return answer in JSON format."}],
            temperature=0.1,
            max_tokens=200,
            json_mode=True
        )
    print ("result=",result)
    
if __name__ == "__main__":
    #asyncio.run(test_rate_limiting())
    asyncio.run(test_hf())
