import time
import tempfile
import os
from .base import ASREngine
from ..audio.resample import pcm_bytes_to_float32


class Qwen3ASREngine(ASREngine):
    def __init__(self, model_id: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype
        self._model = None
        self._processor = None

    async def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        dtype = getattr(torch, self.dtype_str)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device
        )
        self._model.eval()

    async def transcribe_file(self, audio_path: str, language: str = "id") -> dict:
        import torch
        import soundfile as sf

        t0 = time.monotonic()

        inputs = self._processor.apply_transcription_request(
            audio=audio_path, language=language
        ).to(self.device, self._model.dtype)

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=512)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(
            generated_ids, return_format="transcription_only"
        )[0]

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        info = sf.info(audio_path)
        duration_ms = int(info.frames / info.samplerate * 1000)

        return {
            "text": text,
            "language": language,
            "duration_ms": duration_ms,
            "processing_ms": elapsed_ms,
        }

    async def infer_chunk(self, audio_bytes: bytes, is_final: bool = False) -> str:
        import torch
        import soundfile as sf

        audio_np = pcm_bytes_to_float32(audio_bytes)

        # ponytail: temp WAV file — simplest reliable approach
        # upgrade: in-memory BytesIO if processor supports file-like objects
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, 16000, format="WAV", subtype="PCM_16")
            tmp_path = tmp.name

        try:
            inputs = self._processor.apply_transcription_request(
                audio=tmp_path, language="id"
            ).to(self.device, self._model.dtype)

            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, max_new_tokens=512)

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return self._processor.decode(
                generated_ids, return_format="transcription_only"
            )[0]
        finally:
            os.unlink(tmp_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
