# app/tools/parse_html_page.py
"""
HTML Parser Tool
Based on Vals.ai ParseHtmlPage class with BeautifulSoup
"""
import aiohttp
import backoff
from bs4 import BeautifulSoup
from utils.env_setup import init_environment

init_environment()


def is_429(exception):
    """Check if exception is a 429 rate limit error"""
    return (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )


@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    max_tries=8,
    base=2,
    factor=3,
    jitter=backoff.full_jitter,
    giveup=lambda e: not is_429(e),
)
async def parse_html(url: str) -> dict:
    """
    Parse an HTML page and extract its text content.
    
    This tool:
    1. Fetches the HTML from the given URL
    2. Removes script and style tags
    3. Extracts clean text content
    4. Returns the parsed text
    
    Args:
        url: The URL of the HTML page to parse
        
    Returns:
        dict: {"text": "parsed content"} or {"error": "..."}
    """
    headers = {"User-Agent": "FinanceAgent/berkeley@agentbeats.org"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=60) as response:
                response.raise_for_status()
                html_content = await response.text()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return {"text": text}
        
    except aiohttp.ClientError as e:
        if len(str(e)) == 0:
            return {"error": "Timeout error after 60 seconds. URL might be blocked or server is slow."}
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}"}
