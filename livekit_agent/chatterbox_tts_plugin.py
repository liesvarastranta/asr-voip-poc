"""TTS plugin — Edge TTS (cloud, free, no key) or local Chatterbox.

Priority:
  1. Edge TTS (default) — free Microsoft neural TTS, no auth, Indonesian voices.
     Set TTS_VOICE=id-ID-ArdiNeural (or Gadis, etc.). Override via TTS_VOICE env.
  2. Local Chatterbox — if TTS_ENDPOINT points to a local service (e.g. http://localhost:18003/tts).

Groq cloud path removed: Groq decommissioned playai-tts; Orpheus is English-only,
not usable for Indonesian POC. Kept as a comment in case of future models.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave

import numpy as np
from livekit.agents.tts import (
    TTS,
    TTSCapabilities,
    ChunkedStream,
    AudioEmitter,
)
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore


# Default Indonesian neural voices available in Microsoft Edge TTS.
EDGE_TTS_INDONESIAN_VOICES = [
    "id-ID-ArdiNeural",   # male
    "id-ID-GadisNeural",  # female
]


class ChatterboxTTSPlugin(TTS):
    def __init__(
        self,
        *,
        endpoint: str = "",
        sample_rate: int = 24000,
        num_channels: int = 1,
        voice: str = "id-ID-ArdiNeural",
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._endpoint = endpoint.rstrip("/") if endpoint else ""
        self._voice = voice or "id-ID-ArdiNeural"
        # ponytail: only local mode if endpoint is explicitly set to a non-cloud URL.
        # Otherwise default to Edge TTS (cloud, free, Indonesian).
        host = self._endpoint.split("//", 1)[-1].split("/", 1)[0].lower() if self._endpoint else ""
        self._is_local = bool(self._endpoint) and not any(
            tag in host for tag in ("groq.com", "openai.com", "api.openai")
        )

    @property
    def model(self) -> str:
        return "edge-tts" if not self._is_local else "grandhigh/Chatterbox-TTS-Indonesian"

    @property
    def provider(self) -> str:
        return "edge" if not self._is_local else "local-wsl2"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return _TTSChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            endpoint=self._endpoint,
            voice=self._voice,
            is_local=self._is_local,
        )


class _TTSChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        endpoint: str,
        voice: str,
        is_local: bool,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._endpoint = endpoint
        self._voice = voice
        self._is_local = is_local

    async def _run(self, output_emitter: AudioEmitter) -> None:
        if self._is_local:
            await self._run_local(output_emitter)
        else:
            await self._run_edge(output_emitter)

    async def _run_edge(self, output_emitter: AudioEmitter) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(self.input_text, voice=self._voice)
        # edge-tts streams chunks; collect into bytes
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            # chunk is dict with "type" and "data"; we only need audio chunks
            if chunk.get("type") == "audio" and chunk.get("data"):
                mp3_chunks.append(chunk["data"])

        mp3_bytes = b"".join(mp3_chunks)
        if not mp3_bytes:
            raise RuntimeError("edge-tts returned no audio data")

        pcm_int16, sample_rate = _decode_to_pcm16(mp3_bytes, target_sr=self._tts.sample_rate)  # type: ignore[attr-defined]
        _emit_pcm(output_emitter, pcm_int16, sample_rate)

    async def _run_local(self, output_emitter: AudioEmitter) -> None:
        if httpx is None:
            raise ImportError("httpx required for local TTS mode")
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{self._endpoint}/tts",
                params={"text": self.input_text},
            )
            resp.raise_for_status()
            wav_bytes = await resp.aread()
        pcm_int16, sample_rate = _decode_wav_to_pcm16(wav_bytes)
        _emit_pcm(output_emitter, pcm_int16, sample_rate)


def _emit_pcm(output_emitter: AudioEmitter, pcm_int16: np.ndarray, sample_rate: int) -> None:
    output_emitter.initialize(
        request_id=shortuuid(),
        sample_rate=sample_rate,
        num_channels=1,
        mime_type="audio/pcm",
    )
    output_emitter.push(pcm_int16.tobytes())
    output_emitter.flush()


def _decode_to_pcm16(mp3_bytes: bytes, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Decode MP3 bytes → int16 PCM at target sample rate."""
    if sf is None:
        raise ImportError("soundfile required to decode MP3 audio")
    data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        try:
            import torchaudio
            import torch
            wav_t = torch.from_numpy(data).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
            data = wav_t.squeeze(0).numpy()
            sr = target_sr
        except ImportError:
            ratio = target_sr / sr
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data), new_len, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
            sr = target_sr
    pcm_int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_int16, sr


def _decode_wav_to_pcm16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    if sf is None:
        raise ImportError("soundfile required to decode WAV")
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    pcm_int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_int16, sr
