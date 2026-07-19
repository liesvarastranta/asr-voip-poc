# Architecture — ASR Service POC

**As-is date:** 2026-07-18

## 1. System context

```
                           ┌────────────────────┐
                           │  Downstream Service │ (future consumer)
                           │  (NLP, chatbot,    │
                           │   translation, ...) │
                           └────────┬───────────┘
                                    │ HTTP SSE (text)
           ┌────────────────────────┼────────────────────────┐
           │                        ▼                        │
           │  ┌─────────────────────────────────────────┐   │
           │  │       ASR Service (POC)                  │   │
           │  │  ┌─────────────────────────────────┐    │   │
           │  │  │  FastAPI (uvicorn, 1 worker)     │    │   │
           │  │  │  /v1/asr/stream (SSE)            │    │   │
           │  │  │  /v1/asr/transcribe (batch)      │    │   │
           │  │  │  /health   /docs                  │    │   │
           │  │  └────────────┬────────────────────┘    │   │
           │  │               │                          │   │
           │  │  ┌────────────▼────────────────────┐    │   │
           │  │  │  ASR Engine Layer                 │    │   │
           │  │  │  ┌──────────────────────────┐    │    │   │
           │  │  │  │  FasterWhisperEngine      │    │    │   │
           │  │  │  │  model: large-v3          │    │    │   │
           │  │  │  │  backend: CTranslate2      │    │    │   │
           │  │  │  │  device: CUDA              │    │    │   │
           │  │  │  │  compute_type: int8_float16│    │    │   │
           │  │  │  └──────────────────────────┘    │    │   │
           │  │  └─────────────────────────────────┘    │   │
           │  │               │ GPU (RTX 3070Ti)       │   │
           │  └───────────────┼────────────────────────┘   │
           │                  │                             │
           │  ┌───────────────▼────────────────────────┐   │
           │  │  WSL2 Ubuntu-24.04 (native)             │   │
           │  │  faster-whisper + CTranslate2 + CUDA    │   │
           │  │  RTX 3070Ti 8GB / Ryzen 6900HX / 24GB   │   │
           │  └────────────────────────────────────────┘   │
           └───────────────────────────────────────────────┘
```

## 2. Data flow — streaming

```
Client                          ASR Service
  │                                 │
  │  POST /v1/asr/stream            │
  │  Content-Type: octet-stream     │
  │  X-Audio-Sample-Rate: 16000     │
  │  X-Audio-Channels: 1            │
  │  X-Language: id                 │
  │ ──────────────────────────────► │
  │                                 │  1. Baca header, validasi
  │                                 │  2. Siapkan audio buffer (ringbuf)
  │                                 │  3. Mulai SSE response
  │                                 │
  │  SSE: event:ready               │
  │ ◄────────────────────────────── │
  │                                 │
  │  <PCM bytes chunk 1>            │
  │ ──────────────────────────────► │  4. Append ke ringbuf
  │                                 │  5. VAD check (energy-based)
  │                                 │  6. Jika akumulasi >= threshold:
  │                                 │     FasterWhisperEngine.infer_chunk()
  │                                 │     → text baru
  │                                 │
  │  SSE: event:partial             │
  │  data: {"text":"Halo",…}        │
  │ ◄────────────────────────────── │
  │                                 │
  │  <PCM bytes chunk 2>            │
  │ ──────────────────────────────► │  (loop 4-6)
  │                                 │
  │  SSE: event:partial             │
  │  data: {"text":"Halo selamat…"} │
  │ ◄────────────────────────────── │
  │                                 │
  │  <client closes stream>         │   7. Deteksi EOF / silence timeout
  │ ───────────✗                    │   8. Final inference pada buffer penuh
  │                                 │
  │  SSE: event:final               │
  │  data: {"text":"...",is_final:true}│
  │ ◄────────────────────────────── │
```

## 3. Component inventory

| Component | File | Description |
|---|---|---|
| App entry | `src/asr_service/main.py` | FastAPI app + uvicorn launcher |
| Config | `src/asr_service/config.py` | Env vars: MODEL_ID, DEVICE, DTYPE, CHUNK_MS, MAX_SESSION_S |
| SSE endpoint | `src/asr_service/api/stream.py` | POST /asr/stream handler |
| Batch endpoint | `src/asr_service/api/transcribe.py` | POST /asr/transcribe handler |
| SSE helpers | `src/asr_service/api/sse.py` | SSE event formatting, error events |
| Schemas | `src/asr_service/api/schemas.py` | Pydantic models (TranscribeResponse, HealthResponse, SSEEvent) |
| Audio resampler | `src/asr_service/audio/resample.py` | Normalize → 16kHz mono PCM16 |
| VAD | `src/asr_service/audio/vad.py` | Energy-based VAD (opsional, threshold configurable) |
| Engine abstraction | `src/asr_service/engines/base.py` | ABC: `transcribe_file()`, `infer_chunk()` |
| faster-whisper engine | `src/asr_service/engines/faster_whisper.py` | FasterWhisperEngine (CTranslate2) |
| Client example | `clients/python_client.py` | Downstream consumer demo |
| Setup script | `scripts/setup.sh` | One-time venv + deps + LiveKit binary |
| Start scripts | `scripts/start_*.sh` | Per-service native launchers |
| Makefile | `Makefile` | Convenience targets for service management |
| Model download | `scripts/download_models.sh` | Download GGUF + auto-download others |
| Benchmark | `scripts/benchmark.py` | RTF, latency, WER measurement |
| Verify model | `scripts/verify_model.py` | Fase 1: pastikan model load + offline inference jalan |

## 4. Key design decisions

| Decision | Rationale |
|---|---|
| SSE instead of WebSocket | User memilih REST HTTP; SSE adalah sub-protokol HTTP, didukung FastAPI via StreamingResponse, lebih sederhana |
| Chunked request body (not multipart per chunk) | Lebih natural untuk stream; TCP streaming via raw bytes; tidak perlu encode/decode multipart boundary |
| VAD optional, fallback: buffer time-based | Energy-based VAD simpel, tapi jika tidak berfungsi baik, chunking berbasis waktu (e.g., setiap 500ms) cukup untuk POC |
| Singleton model di startup | POC single-session: model dimuat sekali di event `startup`, tidak ada lifecycle per-request. faster-whisper WhisperModel lazy-loaded on first request |
| Re-inference full buffer setiap chunk | Sliding window sederhana; tidak cache-aware seperti Nemotron. Untuk POC 1 user, overhead tidak signifikan. Upgrade: implementasi cache-aware sendiri atau pindah ke Nemotron post-fine-tune |
| VRAM budget ~7.1GB | ASR ~2.1GB + LLM ~2.5GB + TTS ~1.5GB + overhead ~1GB; fits RTX 3070Ti 8GB |
| Native HF cache | Model cache di `~/.cache/huggingface/hub`; persistent across WSL2 restarts; tidak perlu download ulang |
