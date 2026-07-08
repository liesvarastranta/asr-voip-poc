# Architecture — ASR Service POC

**As-is date:** 2026-07-08

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
          │  │  │  /v1/models   /health   /docs     │    │   │
          │  │  └────────────┬────────────────────┘    │   │
          │  │               │                          │   │
          │  │  ┌────────────▼────────────────────┐    │   │
          │  │  │  ASR Engine Layer                 │    │   │
          │  │  │  ┌──────────────────────────┐    │    │   │
          │  │  │  │  Qwen3ASREngine           │    │    │   │
          │  │  │  │  model: Qwen3-ASR-1.7B    │    │    │   │
          │  │  │  │  processor: AutoProcessor  │    │    │   │
          │  │  │  │  device: CUDA (sm_120)     │    │    │   │
          │  │  │  │  dtype: bfloat16           │    │    │   │
          │  │  │  └──────────────────────────┘    │    │   │
          │  │  └─────────────────────────────────┘    │   │
          │  │               │ GPU (Blackwell)          │   │
          │  └───────────────┼────────────────────────┘   │
          │                  │                             │
          │  ┌───────────────▼────────────────────────┐   │
          │  │  Docker (linux/arm64)                   │   │
          │  │  nvidia/cuda + PyTorch + Transformers   │   │
          │  │  ASUS Ascent GX10 (GB10 Grace Blackwell) │   │
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
  │                                 │     Qwen3ASREngine.infer_chunk()
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
| Qwen3 engine | `src/asr_service/engines/qwen_asr.py` | Qwen3ASR implementation |
| Client example | `clients/python_client.py` | Downstream consumer demo |
| Dockerfile | `Dockerfile` | ARM64 + CUDA + Python deps |
| Compose | `docker-compose.yml` | Service + GPU passthrough + volumes |
| Model download | `scripts/download_model.py` | Pull HF model ke cache volume |
| Benchmark | `scripts/benchmark.py` | RTF, latency, WER measurement |
| Verify model | `scripts/verify_model.py` | Fase 1: pastikan model load + offline inference jalan |

## 4. Key design decisions

| Decision | Rationale |
|---|---|
| SSE instead of WebSocket | User memilih REST HTTP; SSE adalah sub-protokol HTTP, didukung FastAPI via StreamingResponse, lebih sederhana |
| Chunked request body (not multipart per chunk) | Lebih natural untuk stream; TCP streaming via raw bytes; tidak perlu encode/decode multipart boundary |
| VAD optional, fallback: buffer time-based | Energy-based VAD simpel, tapi jika tidak berfungsi baik, chunking berbasis waktu (e.g., setiap 500ms) cukup untuk POC |
| Singleton model di startup | POC single-session: model dimuat sekali di event `startup`, tidak ada lifecycle per-request |
| Re-inference full buffer setiap chunk | Sliding window sederhana; tidak cache-aware seperti Nemotron. Untuk POC 1 user, overhead tidak signifikan. Upgrade: implementasi cache-aware sendiri atau pindah ke Nemotron post-fine-tune |
| No VRAM management | 128 GB unified memory: model ~4 GB, buffer audio kilobyte-sized; no OOM risk |
| Docker volume mount untuk HF cache | Model ~4 GB; cache persistent across container restarts; tidak perlu download ulang |
