# Whisper ASR Service — Admin Request

## Need

Jalankan faster-whisper sebagai HTTP service untuk voice agent POC. Service ini dipakai oleh LiveKit agent (bahasa Indonesia).

## Spec

| Item | Value |
|------|-------|
| Host | `10.9.23.200` (GX10) atau VM admin |
| Port | `18001` |
| Model | `small` (~460 MB, balance akurasi/speed) atau `medium` (~1.5 GB, lebih akurat untuk `id`) |
| Compute | CPU ok; GPU lebih cepat |
| Auth | none untuk internal POC (atau header key kalau admin mau) |

## API contract (WAJIB match)

Service harus expose 2 endpoint (sama dengan `docs/openapi.yaml` project ini):

**1. `POST /v1/asr/transcribe`** — batch, multipart upload file audio
```
Request:  multipart/form-data, field `file` (wav/mp3/flac), optional `language` (default "id")
Response: {"text": "...", "language": "id", "duration_ms": N, "processing_ms": M}
```

**2. `POST /v1/asr/stream`** — streaming (optional untuk POC, batch cukup)
```
Request:  chunked octet-stream (PCM 16-bit 16kHz mono)
Response: text/event-stream (SSE): event=partial|final, data={"text":"...","is_final":bool}
```

**3. `GET /health`** — liveness
```
Response: {"status":"ok","model_loaded":true,"model":"...","device":"cpu|cuda"}
```

## Cara cepat (Docker, satu command)

Ada project `faster-whisper-server` (open source, OpenAI-compatible). Admin bisa:

```bash
docker run -d --name whisper-asr -p 18001:8000 \
  -e WHISPER_MODEL=small \
  -e WHISPER_LANGUAGE=id \
  fedirz/faster-whisper-server:latest
```

Atau build sendiri pake `src/asr_service/` project ini (engine faster-whisper, bukan Qwen3).

## Setelah admin jawab

User set di `livekit_agent/.env`:
```
ASR_ENDPOINT=http://10.9.23.200:18001
```

Agent otomatis pakai. Tidak ada code change.
