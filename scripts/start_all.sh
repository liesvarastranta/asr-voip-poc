#!/bin/bash
set -e
mkdir -p logs

echo "=== Starting LiveKit server ==="
bash scripts/start_livekit.sh
sleep 2

echo "=== Starting ASR service ==="
bash scripts/start_asr.sh
sleep 3

echo "=== Starting LLM service ==="
bash scripts/start_llm.sh
sleep 5

echo "=== Starting TTS service ==="
bash scripts/start_tts.sh
sleep 3

echo "=== Starting Agent ==="
bash scripts/start_agent.sh

echo "=== All services started ==="
echo "Check logs/ for output. Run 'bash scripts/stop_all.sh' to stop."
