from abc import ABC, abstractmethod

class ASREngine(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def transcribe_file(self, audio_path: str, language: str = "id") -> dict: ...

    @abstractmethod
    async def infer_chunk(self, audio_bytes: bytes, is_final: bool = False, language: str = "id", sample_rate: int = 16000) -> str: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
