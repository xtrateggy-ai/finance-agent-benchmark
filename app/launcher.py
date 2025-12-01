# app/launcher.py
"""
Launcher for Finance-Agent-Benchmark on AgentBeats
Works in both local (multiprocessing) and Docker (spawns) modes
"""
import multiprocessing
import os
import sys
import asyncio
import argparse
import json
import httpx
import time
#from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from utils.env_setup import init_environment, is_docker

# Initialize environment BEFORE importing agents
init_environment()


async def wait_agent_ready(url: str, timeout: int = 30) -> bool:
    """Poll agent's /card endpoint until ready"""
    async with httpx.AsyncClient() as client:
        for i in range(timeout):
            try:
                r = await client.get(f"{url}/card", timeout=5.0)
                if r.status_code == 200:
                    print(f"✓ Agent ready: {url}")
                    return True
            except Exception as e:
                if i == 0:
                    print(f"  Waiting for {url}... ({e.__class__.__name__})")
            await asyncio.sleep(1)
        print(f"✗ Timeout waiting for {url}")
        return False


def start_green_agent(host: str, port: int):
    """Start green agent in subprocess"""
    # Re-initialize environment in subprocess
    from utils.env_setup import init_environment
    init_environment()
    
    # Import and create agent
    from green_agent_mcp_a2a_judge_rag import GreenAgent
    print("[LAUNCHER] Using green_agent_mcp_a2a.py", file=sys.stderr)
    agent = GreenAgent()
    
    # Override host/port
    agent.agent_host = host
    agent.agent_port = port
    
    # Run
    agent.run()


def start_white_agent(host: str, port: int):
    """Start white agent in subprocess"""
    # Re-initialize environment in subprocess
    from utils.env_setup import init_environment
    init_environment()
    
    # Import and create agent
    from white_agent_mcp_memory import WhiteAgent
    agent = WhiteAgent()
    
    # Override host/port
    agent.agent_host = host
    agent.agent_port = port
    
    # Run
    agent.run()


async def async_main():
    secret_env="secrets/secrets.env"
    parser = argparse.ArgumentParser(description="Local sim for Finance-Agent-Benchmark on AgentBeats")
    parser.add_argument("--num_tasks", 
                        type=int, default=5, 
                        help="Number of tasks")
    parser.add_argument("--env", 
                        type=str, default=secret_env, 
                        help="Relative secrets.env path")
    args = parser.parse_args()

    # Reload env if custom path
    if args.env != secret_env:
        print(f"[LAUNCHER] Reading {args.env} file.", file=sys.stderr)
        ret = load_dotenv(args.env, override=True)
        
        if not ret and not find_dotenv(args.env):
            raise Exception(f"[LAUNCHER] Errno 2: File [{args.env}] not found.")

    # ---- Configuration ----
    GREEN_HOST = os.getenv("GREEN_AGENT_HOST", "127.0.0.1")
    GREEN_PORT = int(os.getenv("GREEN_AGENT_PORT", 9000))
    WHITE_HOST = os.getenv("WHITE_AGENT_HOST", "127.0.0.1")
    WHITE_PORT = int(os.getenv("WHITE_AGENT_PORT", 8000))
    NUM_TASKS  = args.num_tasks if args.num_tasks else int(os.getenv("NUM_TASKS", 5))
    
    green_url = f"http://{GREEN_HOST}:{GREEN_PORT}"
    white_url = f"http://{WHITE_HOST}:{WHITE_PORT}"
    
    print("=" * 60)
    print("Finance-Agent-Benchmark Launcher")
    print("=" * 60)
    print(f"Green Agent: {green_url}")
    print(f"White Agent: {white_url}")
    print(f"Tasks to evaluate: {NUM_TASKS}")
    print(f"Environment: {'Docker' if is_docker() else 'Local'}")
    print("=" * 60)

    # ---- Start agents in separate processes ----
    print("\n[1/4] Launching green agent...")
    p_green = multiprocessing.Process(
        target=start_green_agent,
        args=(GREEN_HOST, GREEN_PORT),
        daemon=True
    )
    p_green.start()
    
    # Give green agent time to start
    await asyncio.sleep(3)

    print("[2/4] Launching white agent...")
    p_white = multiprocessing.Process(
        target=start_white_agent,
        args=(WHITE_HOST, WHITE_PORT),
        daemon=True
    )
    p_white.start()
    
    await asyncio.sleep(2)

    # ---- Wait for both agents to be ready ----
    print("[3/4] Waiting for agents to be ready...")
    
    green_ready = await wait_agent_ready(green_url, timeout=30)
    if not green_ready:
        print("ERROR: Green agent failed to start")
        p_green.terminate()
        p_white.terminate()
        return
    
    white_ready = await wait_agent_ready(white_url, timeout=30)
    if not white_ready:
        print("ERROR: White agent failed to start")
        p_green.terminate()
        p_white.terminate()
        return

    print("\n✓ Both agents are ready!\n")

    # ---- Send assessment task to green agent (single call) ----
    print("[4/4] Sending assessment task to green agent...")
    
    task_config = {
        "method": "run_assessment",
        "args": {
            "num_tasks": NUM_TASKS,
            "white_address": white_url
        }
    }

    try:
        async with httpx.AsyncClient(timeout=900.0) as client:   # 300=5 minuts , 900=15 minutes 
            start_time = time.time()
            
            response = await client.post(
                f"{green_url}/a2a",
                json=task_config
            )
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            result = response.json()
            
            # ---- Display results ----
            print("\n" + "═" * 60)
            print("ASSESSMENT COMPLETE")
            print("═" * 60)
            print(f"Time elapsed: {elapsed:.2f}s")
            print(json.dumps(result, indent=2))
            print("═" * 60)
            
    except Exception as e:
        print(f"\nERROR during assessment: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ---- Cleanup ----
        print("\nTerminating agents...")
        p_green.terminate()
        p_white.terminate()
        p_green.join(timeout=5)
        p_white.join(timeout=5)
        print("Done.")


def main():
    """Entry point"""
    # Set multiprocessing start method
    # 'spawn' works on all platforms (Windows, Linux, Mac)
    # In Docker/Linux, this ensures clean process isolation
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Already set, ignore
        pass
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
