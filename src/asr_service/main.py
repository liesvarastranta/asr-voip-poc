from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import settings
from .engines.base import ASREngine
from .engines.mock import MockASREngine
from .engines.faster_whisper import FasterWhisperEngine
from .api import health, transcribe, stream


def create_app(engine: ASREngine | None = None) -> FastAPI:
    if engine is None:
        if settings.device.startswith("cuda"):
            engine = FasterWhisperEngine(
                model_id=settings.model_id,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        else:
            engine = MockASREngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not engine.is_loaded:
            await engine.load()
        yield

    app = FastAPI(
        title="ASR Service POC",
        version="0.1.0",
        description="Automatic Speech Recognition — faster-whisper-large-v3 on WSL2",
        lifespan=lifespan,
    )

    # ponytail: set engine eagerly for test compat — ASGITransport doesn't trigger lifespan
    app.state.engine = engine

    app.include_router(health.router)
    app.include_router(transcribe.router, prefix="/v1/asr")
    app.include_router(stream.router, prefix="/v1/asr")

    return app


app = create_app()
