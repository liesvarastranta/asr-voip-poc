#!/bin/bash
set -e
mkdir -p logs
source .venv/bin/activate
nohup env PYTHONPATH=src uvicorn asr_service.main:app --host 0.0.0.0 --port 18001 > logs/asr.log 2>&1 &
echo $! > logs/asr.pid
echo "ASR service started (PID $(cat logs/asr.pid)). Log: logs/asr.log"
