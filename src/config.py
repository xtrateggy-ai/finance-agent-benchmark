# src/config.py

from dataclasses import dataclass
import os


@dataclass
class GreenAgentConfig:
    """
    Central configuration for the Green Agent
    """

    # -----------------------------
    # RAG settings
    # -----------------------------

    max_filings: int = int(os.getenv("MAX_FILINGS", 100))

    use_disk_cache: bool = os.getenv(
        "USE_DISK_CACHE",
        "true"
    ).lower() == "true"

    use_local_llm_rag: bool = os.getenv(
        "USE_LOCAL_LLM_RAG",
        "false"
    ).lower() == "true"

    use_local_llm_gpu: bool = os.getenv(
        "USE_LOCAL_LLM_GPU",
        "false"
    ).lower() == "true"


# global singleton config
config = GreenAgentConfig()