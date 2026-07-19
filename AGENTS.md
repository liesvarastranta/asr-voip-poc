# ASR VoIP POC — Project Context

## Overview

Automatic Speech Recognition (ASR) voice AI pipeline for **Bahasa Indonesia**.
Runs on **WSL2 Ubuntu-24.04** (RTX 3070Ti 8GB VRAM, Ryzen 6900HX, 24GB RAM).

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Web Framework | FastAPI, Uvicorn |
| ML/AI | faster-whisper (CTranslate2), llama-cpp-python (CUDA GGUF), Chatterbox-TTS |
| ASR Model | Systran/faster-whisper-large-v3 (int8_float16) |
| LLM Model | meta-llama/Llama-3.2-3B-Instruct (GGUF int4, llama-cpp-python) |
| TTS Model | grandhigh/Chatterbox-TTS-Indonesian |
| Real-time Voice | LiveKit Agents SDK, Silero VAD |
| Runtime | Native WSL2 Ubuntu-24.04, NVIDIA CUDA 12.x |
| Testing | pytest, pytest-asyncio |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LiveKit Server                        │
│                  (WebRTC SFU :7880)                      │
└────────────┬───────────────────────────────┬────────────┘
             │                               │
    ┌────────▼────────┐             ┌────────▼────────┐
    │  LiveKit Agent   │             │   Client Apps    │
    │  (agent.py)      │             │   (Web/Mobile)   │
    │                  │             └─────────────────┘
    │  Silero VAD      │
    │  ↓               │
    │  WhisperASRSTT ───► ASR Service (:18001)
    │  ↓               │        faster-whisper-large-v3
    │  llama-cpp LLM ───► LLM Service (:18002)
    │  ↓               │        Llama-3.2-3B-Instruct
    │  ChatterboxTTS ──► TTS Service (:18003)
    │                  │        Chatterbox-TTS-Indonesian
    └──────────────────┘
```

## Services (Native WSL2)

| Service | Port | Description |
|---|---|---|
| `asr` | 18001 | ASR service (FastAPI + faster-whisper) |
| `llm` | 18002 | LLM service (llama-cpp-python + Llama-3.2-3B) |
| `tts` | 18003 | TTS service (FastAPI + Chatterbox-TTS) |
| `livekit` | 7880 | WebRTC SFU server |
| `agent` | — | LiveKit voice agent (orchestrator) |

## Project Structure

```
├── src/asr_service/          # ASR FastAPI service
│   ├── api/                  # Route handlers (health, transcribe, stream)
│   ├── audio/                # Audio processing (resample)
│   ├── engines/              # ASR engine implementations (mock, faster_whisper)
│   ├── config.py             # Pydantic settings
│   └── main.py               # FastAPI app factory
├── livekit_agent/            # LiveKit voice agent
│   ├── agent.py              # Main agent orchestrator
│   ├── whisper_asr_stt.py    # Custom STT plugin (calls ASR service)
│   └── chatterbox_tts_plugin.py # Custom TTS plugin (calls TTS service)
├── tts_service/              # Chatterbox-TTS FastAPI service
├── llm_service/              # llama-cpp-python OpenAI-compatible service
├── livekit_server/           # LiveKit server config
├── tests/                    # pytest test suite
├── scripts/                  # Utility scripts (setup, start/stop, download)
├── docs/                     # Documentation (architecture, PRD, OpenAPI)
├── Makefile                  # Service management targets
└── opencode.json             # OpenCode agent configuration
```

## Key Conventions

### Python
- Python 3.11+ with type hints
- Async/await for all I/O operations
- Pydantic v2 for validation and settings
- FastAPI routers for API organization

### Audio
- ASR input: 16kHz mono PCM16
- TTS output: 24kHz mono PCM16
- WAV format for file exchange
- Use `soundfile` for read/write, `numpy` for processing

### API Design
- REST endpoints prefixed with `/v1/`
- SSE for streaming responses
- OpenAPI spec in `docs/openapi.yaml`
- Health checks at `/health`

### Native WSL2
- Shared Python venv at `.venv/`
- HuggingFace cache at `~/.cache/huggingface/hub`
- LiveKit server binary installed via `curl -sSL https://get.livekit.io | bash`
- Process management via shell scripts + Makefile

### Testing
- Test files: `test_*.py` in `tests/`
- Async tests with `pytest-asyncio`
- Mock engine for CPU-only testing
- Run: `pytest tests/ -v`

## Development Workflow

1. **Plan** (Tab → Plan agent, model mahal + Superpowers)
   - Brainstorming → PRD
   - Writing plans → `implementasi.md`

2. **Build** (Tab → Build agent, model murah + Ponytail skill)
   - Invoke `ponytail` skill before writing code (YAGNI, shortest diff)
   - Implementasi kode dengan prinsip YAGNI
   - TDD: write test → watch fail → write code → watch pass

3. **Review** (@code-reviewer, model menengah)
   - Security, performance, correctness check

4. **Document** (@docs-writer, model gratis + Caveman)
   - Update docs, hemat token

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_ID` | `large-v3` | ASR model |
| `ASR_DEVICE` | `cuda` | Compute device |
| `ASR_COMPUTE_TYPE` | `int8_float16` | CTranslate2 precision |
| `ASR_CHUNK_MS` | `500` | Streaming chunk size |
| `ASR_SAMPLE_RATE` | `16000` | Audio sample rate |
| `LLM_GGUF_REPO` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | GGUF repo |
| `LLM_GGUF_FILENAME` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | GGUF file |
| `LLM_N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `LLM_N_CTX` | `4096` | Context window |
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server |
| `LIVEKIT_API_KEY` | `devkey` | LiveKit auth (dev mode) |
| `LIVEKIT_API_SECRET` | `secret` | LiveKit auth (dev mode) |

## Quick Commands

```bash
# Setup (one-time)
make setup

# Start full stack
make all

# Stop all services
make stop

# Run tests
make test

# Download models
make download

# Health check
curl http://localhost:18001/health

# Test ASR
curl -F file=@test.wav http://localhost:18001/v1/asr/transcribe
```
