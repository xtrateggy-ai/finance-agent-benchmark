#!/usr/bin/env python3
"""
Local LLM Wrapper - Compatible with litellm interface
Provides drop-in replacement for Gemini/OpenAI API calls
"""

import os
import json
from typing import List, Dict, Any, Optional
from llama_cpp import Llama


class LocalLLMWrapper:
    """
    Wrapper for local LLM that mimics litellm.completion() interface.
    
    Usage:
        # Instead of:
        response = litellm.completion(model="gemini/...", messages=[...])
        
        # Use:
        wrapper = LocalLLMWrapper()
        response = wrapper.completion(messages=[...])
    """
    
    def __init__(
        self,
        model_path: str = None,
        use_gpu: bool = True,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        verbose: bool = False
    ):
        # Get model path from env or use default
        if model_path is None:
            model_path = os.getenv(
                "LOCAL_LLM_MODEL_PATH",
                "models/llama-3.2-1b-instruct-q4_k_m.gguf"
            )
        
        self.model_path = model_path
        self.verbose = verbose
        
        # Initialize llama-cpp model
        print(f"[LOCAL_LLM] Loading model: {model_path}")
        print(f"[LOCAL_LLM] GPU enabled: {use_gpu}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers if use_gpu else 0,
            verbose=verbose
        )
        
        print(f"[LOCAL_LLM] ✅ Model loaded successfully")
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1000,
        response_format: Dict = None,
        **kwargs
    ) -> Any:
        """
        Drop-in replacement for litellm.completion()
        
        Args:
            messages: List of {"role": "user/system/assistant", "content": "..."}
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Max tokens in response
            response_format: {"type": "json_object"} for JSON mode
            
        Returns:
            Object with .choices[0].message.content (mimics OpenAI format)
        """
        
        # Build prompt from messages
        prompt = self._format_messages(messages)
        
        # Check if JSON mode requested
        json_mode = response_format and response_format.get("type") == "json_object"
        
        if json_mode:
            # Add JSON instruction
            prompt += "\n\nRespond with ONLY valid JSON, no explanations:\n{"
        
        # Generate response
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "User:", "Human:", "\n\n\n"],  # Stop sequences
                echo=False
            )
            
            # Extract text
            text = response["choices"][0]["text"].strip()
            
            # Handle JSON mode
            if json_mode:
                # Add opening brace if model didn't include it
                if not text.startswith("{"):
                    text = "{" + text
                
                # Clean up JSON (remove markdown, explanations)
                text = self._extract_json(text)
            
            # Return in OpenAI-compatible format
            return self._create_response_object(text)
            
        except Exception as e:
            print(f"[LOCAL_LLM] Error during generation: {e}")
            # Return error in compatible format
            return self._create_response_object(
                json.dumps({"error": str(e), "action": "answer", "answer": "ERROR"})
            )
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into Llama/Qwen prompt format.
        
        Llama 3.2 format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "user":
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
        
        # Add assistant prompt to trigger response
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        
        return "\n".join(prompt_parts)
    
    def _extract_json(self, text: str) -> str:
        """Extract valid JSON from model output"""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Find JSON object (handles nested braces)
        match = re.search(
            r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}',
            text,
            re.DOTALL
        )
        
        if match:
            return match.group(0)
        
        # Fallback: return original
        return text
    
    def _create_response_object(self, text: str):
        """Create OpenAI-compatible response object"""
        class Message:
            def __init__(self, content):
                self.content = content
        
        class Choice:
            def __init__(self, message):
                self.message = message
        
        class Response:
            def __init__(self, text):
                self.choices = [Choice(Message(text))]
        
        return Response(text)


# Global singleton instance
_local_llm_instance = None


# Replace the singleton with a dictionary
_local_llm_instances = {}  # ← Store multiple instances by component


def get_local_llm(component: str = "white") -> LocalLLMWrapper:
    """
    Get or create local LLM instance for specific component.
    
    Args:
        component: "white", "judge", or "rag"
    """
    global _local_llm_instances
    
    # Return existing instance for this component
    if component in _local_llm_instances:
        print(f"[LOCAL_LLM] Reusing {component} instance")
        return _local_llm_instances[component]
    
    # Create new instance for this component
    use_gpu = bool(int(os.getenv("USE_LOCAL_LLM_GPU", "1")))
    
    if component == "judge":
        model_path = os.getenv(
            "LOCAL_LLM_JUDGE_MODEL", 
            "models/llama-3.2-1b-instruct-q4_k_m.gguf"
        )
        n_ctx = int(os.getenv("LOCAL_LLM_JUDGE_CONTEXT", "2048"))
        print(f"[LOCAL_LLM] Creating {component} instance (model={model_path}, context={n_ctx})")
    
    elif component == "rag":
        model_path = os.getenv(
            "LOCAL_LLM_MODEL_PATH", 
            "models/llama-3.2-1b-instruct-q4_k_m.gguf"
        )
        n_ctx = int(os.getenv("LOCAL_LLM_RAG_CONTEXT", "4096"))
        print(f"[LOCAL_LLM] Creating {component} instance (model={model_path}, context={n_ctx})")
    
    else:  # white agent (default)
        model_path = os.getenv(
            "LOCAL_LLM_MODEL_PATH", 
            "models/qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        n_ctx = int(os.getenv("LOCAL_LLM_WHITE_CONTEXT", "8192"))
        print(f"[LOCAL_LLM] Creating {component} instance (model={model_path}, context={n_ctx})")
    
    _local_llm_instances[component] = LocalLLMWrapper(
        model_path=model_path,
        use_gpu=use_gpu,
        n_ctx=n_ctx
    )
    
    return _local_llm_instances[component]


async def safe_local_llm_call(
    messages: List[Dict],
    temperature: float = 0.1,
    max_tokens: int = 1000,
    response_format: Dict = None,
    component: str = "rag",
    **kwargs
    ):
    """
    Async wrapper for local LLM.
    
    Args:
        component: Which component is calling ("white", "judge", "rag")
    """
    llm = get_local_llm(component=component)  # ← Pass component
    
    return llm.completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        **kwargs
    )

# Test function
async def test_local_llm():
    """Test local LLM wrapper"""
    
    print("=" * 60)
    print("Testing Local LLM Wrapper")
    print("=" * 60)
    
    # Test 1: Simple question
    print("\n[TEST 1] Simple question:")
    response = await safe_local_llm_call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ]
    )
    print(f"Answer: {response.choices[0].message.content}")
    
    # Test 2: JSON mode
    print("\n[TEST 2] JSON mode:")
    response = await safe_local_llm_call(
        messages=[
            {"role": "system", "content": "You are a JSON-only assistant."},
            {"role": "user", "content": 'List 3 colors as JSON array: {"colors": [...]}'}
        ],
        response_format={"type": "json_object"}
    )
    print(f"JSON: {response.choices[0].message.content}")
    
    # Test 3: Tool decision (white agent scenario)
    print("\n[TEST 3] Tool decision:")
    response = await safe_local_llm_call(
        messages=[
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": """
Question: What was Netflix revenue in 2023?

Available tools:
- sec_search_handler: Search SEC filings
- get_financial_metrics: Get metrics from Yahoo Finance

Respond with JSON:
{
    "action": "tool_call",
    "tool": "tool_name",
    "params": {},
    "reasoning": "why"
}
"""}
        ],
        response_format={"type": "json_object"}
    )
    print(f"Decision: {response.choices[0].message.content}")
    
    print("\n" + "=" * 60)
    print("✅ Tests complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_local_llm())
