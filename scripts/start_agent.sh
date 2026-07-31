#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"

source "$PROJECT_DIR/.venv/bin/activate"

export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/livekit_agent"

# NOTE: jangan export ASR_ENDPOINT/VLLM_ENDPOINT/TTS_ENDPOINT di sini.
# agent.py baca dari livekit_agent/.env via load_dotenv(); export di sini
# justru OVERRIDE .env (load_dotenv default override=False).

cd "$PROJECT_DIR/livekit_agent"
nohup python agent.py dev > "$PROJECT_DIR/logs/agent.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/agent.pid"
echo "Agent started (PID $(cat "$PROJECT_DIR/logs/agent.pid")). Log: logs/agent.log"
