# ASR VoIP POC — Voice AI Bahasa Indonesia

Real-time voice pipeline: ASR → LLM → TTS, native WSL2 with CUDA acceleration.

## Demo

[![Demo video](https://raw.githubusercontent.com/liesvarastranta/asr-voip-poc/main/assets-visual/short-demo.mp4)](assets-visual/short-demo.mp4)

[Download demo video](assets-visual/short-demo.mp4)

## Stack

| Component | Model | VRAM | Port |
|-----------|-------|------|------|
| ASR | openai/whisper-small (faster-whisper, int8_float16) | ~1GB | 18001 |
| LLM | Llama-3.2-1B-Instruct (GGUF Q4_K_M, CUDA) | ~0.7GB | 18002 |
| TTS | grandhigh/Chatterbox-TTS-Indonesian | ~1.5GB | 18003 |
| Real-time | LiveKit Agents SDK + Silero VAD | — | 7880 |
| Web App | Vanilla HTML/JS + LiveKit client | — | 8080 |

**Total VRAM**: ~4.2GB (RTX 3070Ti 8GB)

## Quickstart

```bash
# 1. Install system deps (one-time)
sudo apt-get install -y build-essential cmake nvidia-cuda-toolkit

# 2. Setup: venv + deps + LiveKit binary + download models
make setup

# 3. Start full stack
make all && make web

# 4. Open browser
# → http://localhost:8080
# → Klik "Mulai Bicara" → Bicara dalam Bahasa Indonesia
```

## Web App

Buka **http://localhost:8080** setelah menjalankan `make all && make web`:

1. Klik **"Mulai Bicara"**
2. Izinkan akses mikrofon
3. Bicara dalam Bahasa Indonesia
4. Web app menampilkan: transkripsi realtime (user + agent), aktivitas sistem, audio agent

Fitur: mic auto-disable saat agent memproses, transkripsi via text streams.

## Services (`make` targets)

| Target | Service | Port |
|--------|---------|------|
| `make setup` | One-time venv + deps + model download | — |
| `make all` | Start all services | — |
| `make asr` | ASR (faster-whisper) | 18001 |
| `make llm` | LLM (llama-cpp-python) | 18002 |
| `make tts` | TTS (Chatterbox) | 18003 |
| `make livekit` | LiveKit WebRTC SFU | 7880 |
| `make agent` | Voice agent (LiveKit Agents) | — |
| `make web` | Web client | 8080 |
| `make stop` | Stop all services | — |
| `make test` | Run test suite | — |
| `make download` | Download models | — |

## Architecture

```
LiveKit Server :7880
      ↕ WebRTC
  Voice Agent ──→ ASR :18001 (whisper-small)
      │          ──→ LLM :18002 (Llama-1B CUDA)
      │          ──→ TTS :18003 (Chatterbox)
      ↕ WebRTC
  Web App :8080 (LiveKit client)
```

## Development

```bash
source .venv/bin/activate
export PYTHONPATH=src
pytest tests/ -v -k "not skip and not gpu"
```

## API (ASR Service)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/asr/stream` | Streaming real-time (SSE) |
| POST | `/v1/asr/transcribe` | Batch file upload |
| GET | `/health` | Health + model status |

Full contract: [docs/openapi.yaml](docs/openapi.yaml) | Architecture: [docs/architecture.md](docs/architecture.md)

## License

Apache 2.0 (model) — service code: see repository license.
