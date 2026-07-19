"""Whisper ASR STT plugin — calls ASR service at /v1/asr/transcribe."""
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


class WhisperASRSTT(STT):
    def __init__(
        self, *, endpoint: str = "http://localhost:18001", language: str = "id"
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

    @property
    def model(self) -> str:
        return "openai/whisper-small"

    @property
    def provider(self) -> str:
        return "local-wsl2"

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

        wav_buf = io.BytesIO()
        wav_buf.write(
            struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + len(all_pcm),
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
                len(all_pcm),
            )
        )
        wav_buf.write(all_pcm)
        wav_buf.seek(0)

        lang = self._language
        if language is not NOT_GIVEN and language:
            lang = language

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
