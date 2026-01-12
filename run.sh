#!/usr/bin/env bash
set -e

export HOST=0.0.0.0
export AGENT_PORT=${PORT:-9000}   # Cloud Run sets $PORT automatically

echo "Starting Finance-Agent-Benchmark launcher..."

exec python launcher.py
