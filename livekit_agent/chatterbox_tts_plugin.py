"""Chatterbox TTS plugin — calls TTS service at /tts."""
from __future__ import annotations

import io

import httpx
import numpy as np
from livekit import rtc
from livekit.agents.tts import (
    TTS,
    TTSCapabilities,
    ChunkedStream,
    AudioEmitter,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import shortuuid

try:
    import soundfile as sf
except ImportError:
    sf = None


class ChatterboxTTSPlugin(TTS):
    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:18003",
        sample_rate: int = 24000,
        num_channels: int = 1,
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._endpoint = endpoint.rstrip("/")

    @property
    def model(self) -> str:
        return "grandhigh/Chatterbox-TTS-Indonesian"

    @property
    def provider(self) -> str:
        return "local-wsl2"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        if conn_options is None:
            from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
            conn_options = DEFAULT_API_CONNECT_OPTIONS

        return _ChatterboxChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            endpoint=self._endpoint,
        )


class _ChatterboxChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        endpoint: str,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._endpoint = endpoint

    async def _run(self, output_emitter: AudioEmitter) -> None:
        if sf is None:
            raise ImportError("soundfile is required for Chatterbox TTS plugin")

        # ponytail: ignore conn_options timeout, force 300s — LiveKit default too short
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{self._endpoint}/tts",
                params={"text": self.input_text},
            )
            resp.raise_for_status()
            wav_bytes = await resp.aread()

        data_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data_np.ndim > 1:
            data_np = data_np[:, 0]

        data_int16 = (np.clip(data_np, -1.0, 1.0) * 32767).astype(np.int16)
        pcm = data_int16.tobytes()

        # ponytail: single-segment non-streaming — push all PCM at once
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=sr,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()
