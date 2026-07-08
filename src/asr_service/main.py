from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import settings
from .engines.base import ASREngine
from .engines.mock import MockASREngine
from .engines.qwen_asr import Qwen3ASREngine
from .api import health, transcribe, stream


def create_app(engine: ASREngine | None = None) -> FastAPI:
    if engine is None:
        if settings.device.startswith("cuda"):
            engine = Qwen3ASREngine(
                model_id=settings.model_id,
                device=settings.device,
                dtype=settings.dtype,
            )
        else:
            engine = MockASREngine()

    app = FastAPI(
        title="ASR Service POC",
        version="0.1.0",
        description="Automatic Speech Recognition — Qwen3-ASR-1.7B on ASUS Ascent GX10",
    )

    # ponytail: set engine eagerly — httpx ASGITransport doesn't trigger lifespan
    app.state.engine = engine

    app.include_router(health.router)
    app.include_router(transcribe.router, prefix="/v1/asr")
    app.include_router(stream.router, prefix="/v1/asr")

    return app


app = create_app()
