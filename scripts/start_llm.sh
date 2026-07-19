#!/bin/bash
set -e
mkdir -p logs
source .venv/bin/activate
nohup env PYTHONPATH=src python -m llm_service.server > logs/llm.log 2>&1 &
echo $! > logs/llm.pid
echo "LLM service started (PID $(cat logs/llm.pid)). Log: logs/llm.log"
