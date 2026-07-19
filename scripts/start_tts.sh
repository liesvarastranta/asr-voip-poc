#!/bin/bash
set -e
mkdir -p logs
source .venv/bin/activate
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR/tts_service" && nohup env PYTHONPATH="$PROJECT_DIR/src" uvicorn main:app --host 0.0.0.0 --port 18003 > "$PROJECT_DIR/logs/tts.log" 2>&1 &
echo $! > "$PROJECT_DIR/logs/tts.pid"
echo "TTS service started (PID $(cat "$PROJECT_DIR/logs/tts.pid")). Log: logs/tts.log"
