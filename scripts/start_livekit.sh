#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"
source "$PROJECT_DIR/.venv/bin/activate"
nohup livekit-server --dev --bind 0.0.0.0 > "$PROJECT_DIR/logs/livekit.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/livekit.pid"
echo "LiveKit server started (PID $(cat "$PROJECT_DIR/logs/livekit.pid")). Log: logs/livekit.log"
