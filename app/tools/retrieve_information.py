# app/tools/retrieve_information.py
"""
Retrieve Information Tool
Based on Vals.ai RetrieveInformation class
Extracts portions of stored documents and sends to LLM for analysis
"""
import re
import litellm
from utils.env_setup import init_environment

init_environment()


async def retrieve_information(
    prompt: str,
    input_character_ranges: dict,
    data_storage: dict,
    llm_model: str,
    llm_api_key: str
) -> dict:
    """
    Retrieve information from data_storage and allow character range extraction.
    
    This tool:
    1. Finds all {{key}} placeholders in the prompt
    2. Replaces them with content from data_storage
    3. Optionally extracts specific character ranges to avoid token limits
    4. Sends the complete prompt to LLM
    5. Returns the LLM's response
    
    Args:
        prompt: The prompt with {{key}} placeholders
        input_character_ranges: Dict mapping keys to [start, end] character ranges
        data_storage: Dict containing stored document content
        llm_model: The LLM model to use
        llm_api_key: API key for the LLM
        
    Returns:
        dict: {"retrieval": "LLM response", "usage": {...}} or {"error": "..."}
        
    Example:
        prompt = "Find revenue in: {{apple_10k}}"
        input_character_ranges = {"apple_10k": [1000, 5000]}
        data_storage = {"apple_10k": "...100,000 chars of 10-K..."}
        
        → Extracts chars 1000-5000 from apple_10k
        → Sends "Find revenue in: [5000 chars]" to LLM
        → Returns LLM's answer
    """
    
    # Verify that the prompt contains at least one placeholder
    if not re.search(r"{{[^{}]+}}", prompt):
        return {
            "error": "Prompt must include at least one key from data storage in format {{key_name}}"
        }
    
    # Find all keys in the prompt
    keys = re.findall(r"{{([^{}]+)}}", prompt)
    formatted_data = {}
    
    # Apply character range to each document before substitution
    for key in keys:
        if key not in data_storage:
            available_keys = ', '.join(data_storage.keys()) if data_storage else "none"
            return {
                "error": f"Key '{key}' not found in data storage. Available: {available_keys}"
            }
        
        # Extract the specified character range from the document if provided
        doc_content = data_storage[key]
        
        if key in input_character_ranges:
            char_range = input_character_ranges[key]
            
            if len(char_range) == 0:
                # Empty list = use full document
                formatted_data[key] = doc_content
            elif len(char_range) != 2:
                return {
                    "error": f"Character range for '{key}' must be [start, end] or [] for full doc"
                }
            else:
                # Extract specified range
                start_idx = int(char_range[0])
                end_idx = int(char_range[1])
                formatted_data[key] = doc_content[start_idx:end_idx]
        else:
            # Use the full document if no range is specified
            formatted_data[key] = doc_content
    
    # Convert {{key}} format to Python string formatting {key}
    formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
    
    try:
        # Replace placeholders with actual content
        final_prompt = formatted_prompt.format(**formatted_data)
    except KeyError as e:
        return {
            "error": f"Key {str(e)} not found in data storage"
        }
    
    # Send to LLM
    try:
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": final_prompt}],
            api_key=llm_api_key
        )
        
        # Extract response and usage
        llm_answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return {
            "retrieval": llm_answer,
            "usage": usage
        }
        
    except Exception as e:
        return {
            "error": f"LLM error: {str(e)}"
        }
