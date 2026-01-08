# Kill Agentbeats
pkill -f agentbeats || true
pkill -f uvicorn || true
pkill -f "python.*main.py" || true
pkill -f "python.*green_agent" || true
pkill -f "python.*white_agent" || true

ps aux | grep "$USER.*python.*main.py" | awk '{print $2}'  | xargs kill -9 2>/dev/null || true

# Delete .ab (app-local)
rm -rf .ab
rm -rf ~/.cache/agentbeats ~/.cache/earthshaker



# Earthshaker/AgentBeats SDK globals (no clear_cache() API, so manual rm)
rm -rf ~/.cache/earthshaker ~/.cache/agentbeats ~/.local/share/earthshaker

pip cache purge  # Purges wheels (includes agentbeats/earthshaker)

# Unset envs pinning stale state (e.g., from prior runs)
unset AGENTBEATS_AGENT_ID AGENTBEATS_DEBUG EARTHSHAKER_CACHE_DIR

unset $(env | grep -i agentbeats | cut -d= -f1)
unset $(env | grep -i earthshaker | cut -d= -f1)
