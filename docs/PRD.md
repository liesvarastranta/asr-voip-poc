# Product Requirements Document — ASR Service POC

**Status:** Approved  
**Date:** 2026-07-08  
**Engine:** Qwen3-ASR-1.7B  
**Hardware:** ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell)  
**Repository:** to-be-created (public GitHub)

---

## 1. Background

Service ASR (Automatic Speech Recognition) yang melayani transkripsi streaming real-time Bahasa Indonesia. Service ini dirancang sebagai **POC** — bukti kelayakan teknis — dengan API terbuka agar kelak dapat dihubungkan ke service lain (microservice / downstream consumer).

Target hardware: **ASUS Ascent GX10**, sebuah desktop AI supercomputer berbasis NVIDIA GB10 Grace Blackwell dengan CUDA compute.

## 2. Goals

| Goal | Keterangan |
|---|---|
| G1 | Service menerima audio streaming real-time (PCM 16-bit 16kHz mono) via HTTP dan mengembalikan transkrip partial Bahasa Indonesia |
| G2 | Akurasi WER yang layak (<15% pada uji internal) untuk Bahasa Indonesia umum |
| G3 | End-to-end latency kumulatif < 1 detik untuk 1 user |
| G4 | Real-Time Factor (RTF) < 1.0 — processing lebih cepat dari durasi audio |
| G5 | API kontrak terbuka (OpenAPI 3.0) — dapat dipanggil service lain tanpa dokumentasi tambahan |
| G6 | Berjalan dalam Docker pada GX10 (ARM64 + CUDA Blackwell) |
| G7 | Tersedia client contoh (Python) yang mengonsumsi SSE stream |

## 3. Non-goals

- Konkurensi > 1 user (single-session POC)
- Autentikasi / otorisasi / TLS
- Persistensi transkrip / database
- Diarization (speaker separation)
- Fine-tuning / transfer learning
- Multi-model serving / A/B routing
- High-availability / auto-scaling
- Production monitoring / observability
- Support bahasa selain Indonesia di POC ini

## 4. User stories

| ID | Story |
|---|---|
| US1 | Sebagai downstream service, aku kirim stream audio mentah dan terima transkrip partial via SSE, agar aku bisa menampilkan teks secara real-time |
| US2 | Sebagai developer, aku upload file audio (.wav/.mp3) dan dapat transkrip lengkap via endpoint batch, untuk testing & benchmark |
| US3 | Sebagai developer, aku buka `/docs` dan lihat Swagger UI dengan semua endpoint terdokumentasi |
| US4 | Sebagai operator, aku cek `/health` dan tahu apakah service & model GPU sudah siap |

## 5. API Contract

### 5.1 Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/asr/stream` | Streaming real-time: input chunked PCM → output SSE partial transcript |
| `POST` | `/v1/asr/transcribe` | Batch: upload file audio → JSON transkrip lengkap |
| `GET` | `/v1/models` | Info model yang dimuat (ID, device, dtype, status) |
| `GET` | `/health` | Liveness + GPU readiness check |
| `GET` | `/docs` | Swagger UI (auto FastAPI) |

### 5.2 Streaming endpoint detail

**Request:**
```
POST /v1/asr/stream
Content-Type: application/octet-stream
X-Audio-Sample-Rate: 16000
X-Audio-Channels: 1
X-Audio-Format: s16le
X-Language: id

<stream of raw PCM 16-bit LE bytes>
```

**Response:** `text/event-stream` (SSE)

```
event: partial
data: {"text": "Halo selamat pagi", "is_final": false, "t_ms": 1500}

event: partial
data: {"text": "Halo selamat pagi semuanya", "is_final": false, "t_ms": 2500}

event: final
data: {"text": "Halo selamat pagi semuanya.", "is_final": true, "t_ms": 3500, "language": "id"}
```

**Event fields:**

| Field | Type | Description |
|---|---|---|
| `text` | string | Akumulasi transkrip terbaru (replaces whole, bukan delta) |
| `is_final` | bool | `false` = masih ongoing, `true` = stream/utterance selesai |
| `t_ms` | int | Timestamp audio dalam milidetik saat token terakhir diproses |
| `language` | string | Kode bahasa (hanya di event `final`) |

**Error event:**
```
event: error
data: {"code": "AUDIO_FORMAT", "message": "expected PCM 16-bit LE, got float32"}
```

### 5.3 Batch endpoint detail

**Request:**
```
POST /v1/asr/transcribe
Content-Type: multipart/form-data

file: audio.wav (atau .mp3, .flac)
language: id (opsional, default "id")
```

**Response:**
```json
{
  "text": "Halo selamat pagi semuanya.",
  "language": "id",
  "duration_ms": 3500,
  "processing_ms": 420
}
```

### 5.4 Health endpoint

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-ASR-1.7B-hf",
  "device": "cuda:0",
  "model_loaded": true
}
```

## 6. Architecture

```
[Client / Downstream Service]
    │
    │  HTTP (chunked PCM)
    ▼
┌──────────────────────────────────────────────────┐
│  FastAPI ASR Service (uvicorn, single worker)    │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │  /v1/asr/stream (SSE endpoint)           │    │
│  │    │                                      │    │
│  │    ├─ Audio buffer (ringbuf)              │    │
│  │    ├─ Resampler → 16kHz mono PCM16       │    │
│  │    ├─ VAD (optional, energy-based)       │    │
│  │    ├─ Chunk builder (sized frames)       │    │
│  │    ├─ Qwen3ASREngine.infer_chunk()       │    │
│  │    └─ SSE writer (partial/final)         │    │
│  ├──────────────────────────────────────────┤    │
│  │  /v1/asr/transcribe (batch endpoint)     │    │
│  │    └─ Qwen3ASREngine.transcribe_file()   │    │
│  ├──────────────────────────────────────────┤    │
│  │  /v1/models  /health  /docs              │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  Engine layer                                     │
│  ┌──────────────────────────────────────────┐    │
│  │  class ASREngine (Abstract)              │    │
│  │    ├─ Qwen3ASREngine                     │    │
│  │    │   model: Qwen3ASRForCondGeneration  │    │
│  │    │   processor: AutoProcessor           │    │
│  │    │   device: cuda (Blackwell sm_120)    │    │
│  │    │   dtype: bfloat16                    │    │
│  │    └─ (future) FasterWhisperEngine       │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
    │
    │  SSE (text/event-stream)
    ▼
[Downstream service menerima transkrip]
```

**Docker layer:**
```
Base:   nvidia/cuda:12.8-cudnn-runtime-ubuntu24.04 (linux/arm64)
        atau nvcr.io/nvidia/pytorch:YY.MM-py3 (arm64)
        └─ PyTorch ≥ 2.7 (CUDA 12.8, sm_120 Blackwell)
        └─ transformers ≥ 5.13.0
        └─ torchaudio, ffmpeg
        └─ FastAPI, uvicorn, sse-starlette
```

## 7. Model specification

**Primary engine:** `Qwen/Qwen3-ASR-1.7B-hf`

| Property | Value |
|---|---|
| Model class | `Qwen3ASRForConditionalGeneration` |
| Processor | `AutoProcessor` |
| Parameters | ~2B (1.7B + multimodal encoder) |
| Precision | BF16 |
| Memory | ~4 GB VRAM (unified memory, fits easily in 128 GB) |
| Languages | 30 bahasa + 22 dialek (termasuk `id`) |
| Inference mode | Offline + Streaming (unified, single model) |
| Forced alignment | Via `Qwen3-ForcedAligner-0.6B-hf` (opsional POC) |
| Pipeline | `apply_transcription_request()` untuk batch, TBD streaming API |
| Transformers version | Install from source (`git+https://github.com/huggingface/transformers`) atau ≥ rilis yang memuat model |
| License | Apache 2.0 |

**Fallback engine** (jika Qwen3-ASR kendala streaming): `deepdml/faster-whisper-large-v3-turbo-ct2` via chunked approach + VAD.
**Future candidate** (jika fine-tuning → Bahasa Indonesia): `nvidia/nemotron-3.5-asr-streaming-0.6b`.

## 8. Hardware constraints

| Component | Spec | Impact |
|---|---|---|
| SoC | NVIDIA GB10 Grace Blackwell | CUDA tersedia, sm_120 |
| CPU | ARM v9.2-A (aarch64) | Base image Docker harus `linux/arm64` |
| GPU | NVIDIA Blackwell (integrated) | PyTorch butuh ≥2.7 untuk sm_120 |
| Memory | 128 GB LPDDR5x unified | VRAM tidak terbatas; model 4 GB ringan |
| Storage | 1TB NVMe | Cukup untuk model + cache |
| OS | Ubuntu Linux (shipped) | Standar |
| PSU | 240W | Mencukupi (GPU integrated) |

## 9. Technology stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | FastAPI + uvicorn | Ringan, async, auto OpenAPI, SSE native |
| ASR engine | Qwen3-ASR-1.7B (HF Transformers) | Dukung id, streaming, BF16, Apache-2.0 |
| ML runtime | PyTorch ≥ 2.7 + CUDA 12.8 | Support Blackwell (sm_120) |
| Audio processing | torchaudio + ffmpeg | Resample, decode, format conversion |
| Container | Docker (`linux/arm64`) | Portabel, isolasi dependensi CUDA |
| Container orchestration | docker-compose | Sederhana, single-node |
| SSE | sse-starlette | Server-Sent Events via FastAPI |
| OpenAPI | FastAPI auto (Swagger UI) | No extra work |
| Client | Python + httpx-sse | Contoh downstream consumer |

## 10. Implementation phases

| Phase | Deliverable | Verification |
|---|---|---|
| 0 | `docs/PRD.md`, `docs/architecture.md`, `docs/openapi.yaml` | Docs reviewed |
| 1 | Docker base image build (arm64+CUDA+PyTorch) + model download + offline transcribe | `python scripts/verify_model.py` → transkrip valid |
| 2 | FastAPI scaffold + batch endpoint `/transcribe` + OpenAPI | `curl -F file=@test.wav /v1/asr/transcribe` → 200 JSON |
| 3 | Streaming endpoint `/asr/stream` (SSE + chunked) | `python clients/python_client.py` → partial text emitted real-time |
| 4 | Dockerize + compose + healthcheck | `docker compose up` → `/health` returns ok |
| 5 | Example client + integration demo | Client terhubung SSE, terima transkrip |
| 6 | Benchmark: RTF, first-token latency, WER sampel | Benchmark report di `/scripts/benchmark.py` |
| 7 | Hardening: error handling, graceful shutdown, logging | Stress test 1 session, graceful SIGTERM |

## 11. Acceptance criteria

- [ ] Service berjalan dalam Docker pada GX10, GPU terdeteksi (`nvidia-smi` via CUDA passthrough)
- [ ] `POST /v1/asr/transcribe` dengan file `.wav` → JSON `{"text":"...","duration_ms":N}`
- [ ] `POST /v1/asr/stream` dengan chunked PCM → SSE `event:partial` setiap ~500-1000 ms dengan akumulasi teks
- [ ] End-to-end latency < 1 detik untuk final utterance (1 user)
- [ ] RTF < 1.0 pada sample audio 10 detik
- [ ] `/docs` menampilkan Swagger UI dengan semua endpoint (termasuk request/response schema SSE)
- [ ] `/health` menampilkan `{model_loaded:true,device:"cuda:0"}`
- [ ] Client contoh (`clients/python_client.py`) berhasil menjalankan siklus: send stream → terima partial → final transcript akurat
- [ ] WER ≤ ~12% pada uji internal 10 sample Bahasa Indonesia (optional, nice-to-have)

## 12. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| R1 — PyTorch aarch64 Blackwell wheel tidak tersedia | **High** | Gunakan container `nvcr.io/nvidia/pytorch` arm64 jika ada; fallback: build PyTorch dari source; validasi di Fase 1 |
| R2 — Qwen3-ASR tidak punya API streaming native | **Medium** | Implementasi sliding-window: akumulasi buffer audio, re-infer pada buffer penuh, emit partial transcript yang berubah. Cukup untuk demo 1 user. |
| R3 — Qwen3-ASR terlalu lambat untuk RTF < 1 | **Medium** | Fallback ke Qwen3-ASR-0.6B (lebih kecil, lebih cepat, WER 6.31). Atau gunakan `torch.compile` (klaim 2.4x speedup). |
| R4 — Transformers versi tidak kompatibel | **Medium** | Pin versi spesifik di `requirements.txt`; install from source dengan hash commit yang terverifikasi |
| R5 — CTranslate2 (faster-whisper fallback) tidak kompatibel aarch64+Blackwell | **Low** | Tidak dijadikan primary; jika Qwen3-ASR gagal, validasi dulu kompatibilitas CTranslate2 aarch64. |
| R6 — Format audio klien bervariasi | **Medium** | Normalisasi server-side: `torchaudio.sox_effects` resample ke 16kHz mono PCM16; batasi format yang didukung (PCM16, WAV, MP3). |
| R7 — Model download lambat / cache tidak persisten | **Low** | Volume mount `hf_cache:/root/.cache/huggingface` di compose |
| R8 — Memory exhaustion pada sesi audio panjang | **Low** | Batasi durasi maksimal streaming (e.g., 5 menit per sesi untuk POC); flush buffer jika > threshold |

## 13. Open questions / future

- Kapan BERT? — setelah POC tervalidasi. Streaming real-time + akurasi OK → tim bisa memutuskan full production.
- Bagaimana integrasi ke service downstream? — POC menyediakan OpenAPI + contoh client. Service downstream tinggal implementasi HTTP + SSE consumer.
- Bagaimana Nemotron untuk Indonesia? — fine-tuning Nemotron 3.5 ASR dengan dataset id adalah kandidat optimasi di fase post-POC, terutama untuk streaming latency yang lebih rendah (cache-aware native streaming).

## 14. Glossary

| Term | Definition |
|---|---|
| ASR | Automatic Speech Recognition |
| BF16 | Brain floating point 16-bit — format numerik hemat memori, akurasi mendekati FP32 |
| CUDA | Compute Unified Device Architecture — platform komputasi paralel NVIDIA |
| GB10 | Grace Blackwell 10 — superchip SoC di GX10 (CPU ARM + GPU Blackwell) |
| PCM | Pulse-Code Modulation — format audio mentah tanpa kompresi |
| RTF | Real-Time Factor — waktu proses / durasi audio; RTF < 1 berarti lebih cepat dari real-time |
| sm_120 | Streaming Multiprocessor architecture 120 — compute capability Blackwell |
| SSE | Server-Sent Events — HTTP-based push protocol untuk streaming data server → client |
| WER | Word Error Rate — metrik akurasi ASR (% kata yang dikenali salah) |
| VAD | Voice Activity Detection — deteksi segmen suara vs. diam |
