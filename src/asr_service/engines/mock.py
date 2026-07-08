import asyncio
from .base import ASREngine

class MockASREngine(ASREngine):
    def __init__(self):
        self._loaded = False
        self.model_id = "mock/model"
        self.device = "cpu"
        self.dtype_str = "float32"

    async def load(self) -> None:
        await asyncio.sleep(0.001)
        self._loaded = True

    async def transcribe_file(self, audio_path: str, language: str = "id") -> dict:
        return {
            "text": "mock transcription",
            "language": language,
            "duration_ms": 1000,
            "processing_ms": 10,
        }

    async def infer_chunk(self, audio_bytes: bytes, is_final: bool = False, language: str = "id", sample_rate: int = 16000) -> str:
        return "mock final" if is_final else "mock partial"

    @property
    def is_loaded(self) -> bool:
        return self._loaded
