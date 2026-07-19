#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"

source "$PROJECT_DIR/.venv/bin/activate"

export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"

setsid python "$PROJECT_DIR/clients/server.py" > "$PROJECT_DIR/logs/web.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/web.pid"
echo "Web client started (PID $(cat "$PROJECT_DIR/logs/web.pid")). Open http://localhost:8080"
