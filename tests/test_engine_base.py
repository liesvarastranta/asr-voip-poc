import pytest
from asr_service.engines.base import ASREngine

def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        ASREngine()

def test_subclass_must_implement_all():
    with pytest.raises(TypeError):
        class Incomplete(ASREngine):
            async def load(self) -> None: pass
        Incomplete()

def test_valid_subclass():
    class Complete(ASREngine):
        _loaded = False
        async def load(self) -> None: self._loaded = True
        async def transcribe_file(self, audio_path: str, language: str = "id") -> dict: return {}
        async def infer_chunk(self, audio_bytes: bytes, is_final: bool = False) -> str: return ""
        @property
        def is_loaded(self) -> bool: return self._loaded

    instance = Complete()
    assert instance.is_loaded is False
