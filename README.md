# ASR Service POC

Automatic Speech Recognition service for **Bahasa Indonesia**.
Engine: **Qwen3-ASR-1.7B** on **ASUS Ascent GX10** (NVIDIA GB10 Grace Blackwell).

## Quickstart (GX10)

```bash
# 1. Download model to cache volume
docker compose run --rm asr python scripts/download_model.py

# 2. Start service
docker compose up --build

# 3. Test
curl http://localhost:8000/health
curl -F file=@test.wav http://localhost:8000/v1/asr/transcribe
python clients/python_client.py test.wav
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/asr/stream` | Streaming real-time (SSE) |
| POST | `/v1/asr/transcribe` | Batch file upload |
| GET | `/v1/models` | Loaded model info |
| GET | `/health` | Health + GPU status |
| GET | `/docs` | Swagger UI |

Full contract: [docs/openapi.yaml](docs/openapi.yaml)

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
uvicorn asr_service.main:app --reload
```

## Architecture

See [docs/architecture.md](docs/architecture.md) and [docs/PRD.md](docs/PRD.md).

## License

Apache 2.0 (model) — service code: see repository license.
