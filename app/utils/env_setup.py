# app/utils/env_setup.py
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def is_docker():
    """Detect if running inside a Docker or containerized environment."""
    path_cgroup = "/proc/1/cgroup"
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.isfile(path_cgroup):
        with open(path_cgroup, "rt") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    return False


def init_environment(verbose: bool = False):
    """
    Ensures imports work from anywhere and loads secrets/secrets.env if present.
    Skips .env loading when running inside a Docker/AgentBeats container.
    """
    # --- Ensure /app is importable ---
    if Path(__file__).parent.name != "app":
        sys.path.insert(0, str(Path(__file__).parent.parent))

    # --- Skip dotenv in container runtime ---
    if is_docker():
        if verbose:
            print("[ENV_SETUP] Running inside container — skipping .env load.")
        return
        
    # --- Load secrets.env if exists ---
    app_dir = Path(__file__).resolve().parent.parent
    env_file = app_dir / "secrets" / "secrets.env"
    if env_file.exists():
        load_dotenv(env_file)
        if verbose:
            print(f"[ENV_SETUP] Loaded {env_file}")
    elif verbose:
        print(f"[ENV_SETUP] [Errno 2] {env_file} not found")
    
    return env_file

