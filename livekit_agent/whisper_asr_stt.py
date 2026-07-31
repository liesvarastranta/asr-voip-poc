"""Whisper ASR STT plugin.

Two modes:
  1. Local ASR service — POST {endpoint}/v1/asr/transcribe (custom OpenAPI contract).
  2. OpenAI-compatible cloud (Groq, OpenAI, etc.) — uses openai SDK
     `client.audio.transcriptions.create()`. Requires ASR_API_KEY.

Auto-detect: if ASR_API_KEY is set AND endpoint points to an OpenAI-compatible
host (groq/openai/api.openai.com), use mode 2. Otherwise mode 1.
"""
from __future__ import annotations

import io
import struct

import httpx
from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt import (
    STT,
    STTCapabilities,
    SpeechEvent,
    SpeechEventType,
    SpeechData,
)
from livekit.agents.utils import AudioBuffer
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr


def _wav_from_pcm(pcm: bytes, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    buf.write(
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(pcm),
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            len(pcm),
        )
    )
    buf.write(pcm)
    buf.seek(0)
    return buf


class WhisperASRSTT(STT):
    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:18001",
        language: str = "id",
        api_key: str = "",
        model: str = "",
    ):
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._endpoint = endpoint.rstrip("/")
        self._language = language
        self._api_key = api_key
        self._model = model or "whisper-large-v3"

        # Auto-detect: OpenAI-compatible cloud if api_key set AND host is groq/openai
        host = self._endpoint.split("//", 1)[-1].split("/", 1)[0].lower()
        self._is_cloud = bool(api_key) and any(
            tag in host for tag in ("groq.com", "openai.com", "api.openai")
        )

        if self._is_cloud:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key, base_url=self._endpoint)
        else:
            self._client = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq" if self._is_cloud else "local-wsl2"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        if isinstance(buffer, rtc.AudioFrame):
            frames = [buffer]
        else:
            frames = buffer

        if not frames:
            return self._empty_event()

        sample_rate = frames[0].sample_rate
        all_pcm = bytearray()
        for frame in frames:
            all_pcm.extend(frame.data)

        num_samples = len(all_pcm) // 2
        if num_samples == 0:
            return self._empty_event()

        lang = self._language
        if language is not NOT_GIVEN and language:
            lang = language

        if self._is_cloud:
            return await self._recognize_cloud(bytes(all_pcm), sample_rate, lang, num_samples)
        return await self._recognize_local(bytes(all_pcm), sample_rate, lang, num_samples, conn_options)

    async def _recognize_cloud(
        self, pcm: bytes, sample_rate: int, lang: str, num_samples: int
    ) -> SpeechEvent:
        wav_buf = _wav_from_pcm(pcm, sample_rate)
        wav_buf.name = "audio.wav"  # type: ignore[attr-defined]

        # Groq: model "whisper-large-v3", language "id", response_format "json"
        resp = await self._client.audio.transcriptions.create(  # type: ignore[union-attr]
            model=self._model,
            file=wav_buf,
            language=lang,
            response_format="json",
        )
        text = resp.text or ""

        duration_s = num_samples / sample_rate
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language=lang,
                    text=text,
                    start_time=0.0,
                    end_time=duration_s,
                    confidence=0.9,
                )
            ],
        )

    async def _recognize_local(
        self,
        pcm: bytes,
        sample_rate: int,
        lang: str,
        num_samples: int,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        wav_buf = _wav_from_pcm(pcm, sample_rate)
        timeout = conn_options.timeout if conn_options.timeout else 120.0
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(
                f"{self._endpoint}/v1/asr/transcribe",
                files={"file": ("audio.wav", wav_buf, "audio/wav")},
                data={"language": lang},
            )
            resp.raise_for_status()
            data = resp.json()

        duration_s = (
            data.get("duration_ms", num_samples / sample_rate * 1000) / 1000
        )
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language=lang,
                    text=data["text"],
                    start_time=0.0,
                    end_time=duration_s,
                    confidence=0.9,
                )
            ],
        )

    def _empty_event(self) -> SpeechEvent:
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language=self._language, text="", confidence=1.0)],
        )
