#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"

source "$PROJECT_DIR/.venv/bin/activate"

export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
export ASR_ENDPOINT="${ASR_ENDPOINT:-http://localhost:18001}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:18002/v1}"
export TTS_ENDPOINT="${TTS_ENDPOINT:-http://localhost:18003}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/livekit_agent"

cd "$PROJECT_DIR/livekit_agent"
nohup python agent.py dev > "$PROJECT_DIR/logs/agent.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/agent.pid"
echo "Agent started (PID $(cat "$PROJECT_DIR/logs/agent.pid")). Log: logs/agent.log"
