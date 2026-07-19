#!/bin/bash
set -e
source .venv/bin/activate

echo "=== Downloading LLM GGUF model ==="
python -m llm_service.download_model

echo "=== ASR model (faster-whisper) auto-downloads on first load ==="
echo "=== TTS model (Chatterbox) auto-downloads on first load ==="
echo "=== Models download complete ==="
