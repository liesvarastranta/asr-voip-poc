import os
import tempfile
import time
from .base import ASREngine
from ..audio.resample import pcm_bytes_to_float32


class FasterWhisperEngine(ASREngine):
    def __init__(
        self,
        model_id: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ):
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = None

    async def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_id, device=self.device, compute_type=self.compute_type
        )

    async def transcribe_file(self, audio_path: str, language: str = "id") -> dict:
        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            audio_path, language=language, beam_size=5, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        duration_ms = int(info.duration * 1000)
        return {
            "text": text,
            "language": language,
            "duration_ms": duration_ms,
            "processing_ms": elapsed_ms,
        }

    async def infer_chunk(
        self,
        audio_bytes: bytes,
        is_final: bool = False,
        language: str = "id",
        sample_rate: int = 16000,
    ) -> str:
        import soundfile as sf

        audio_np = pcm_bytes_to_float32(audio_bytes)
        if sample_rate != 16000:
            from ..audio.resample import resample
            audio_np = resample(audio_np, sample_rate, 16000)

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, audio_np, 16000, format="WAV", subtype="PCM_16")
            segments, _ = self._model.transcribe(
                tmp_path, language=language, beam_size=5, vad_filter=True
            )
            return " ".join(segment.text.strip() for segment in segments)
        finally:
            os.unlink(tmp_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
