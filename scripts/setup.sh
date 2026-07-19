#!/bin/bash
set -e

echo "=== Creating shared Python venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing ASR deps ==="
pip install --upgrade pip
pip install faster-whisper ctranslate2 fastapi uvicorn sse-starlette \
    pydantic-settings python-multipart soundfile numpy scipy httpx httpx-sse

echo "=== Installing LLM deps (CUDA build) ==="
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install "llama-cpp-python[server]" huggingface-hub pydantic-settings

echo "=== Installing TTS deps ==="
pip install chatterbox-tts==0.1.1 torch torchaudio soundfile \
    huggingface-hub safetensors ml_dtypes numpy

echo "=== Installing Agent deps ==="
pip install "livekit-agents[openai,silero]" livekit httpx python-dotenv soundfile numpy

echo "=== Installing dev deps ==="
pip install pytest pytest-asyncio pytest-httpx

echo "=== Installing LiveKit server binary ==="
curl -sSL https://get.livekit.io | bash

echo "=== Downloading models ==="
bash scripts/download_models.sh

echo "=== Setup complete ==="
echo "Run 'make all' to start all services."
