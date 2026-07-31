"""LiveKit Voice Agent — Indonesia ASR + LLM + TTS (WSL2 native).

Usage:
  python agent.py dev          # dev mode, connects to LiveKit server
"""
import os
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    AgentServer,
    cli,
    room_io,
)
from livekit.agents import stt as stt_mod
from livekit.plugins import silero, openai

from whisper_asr_stt import WhisperASRSTT
from chatterbox_tts_plugin import ChatterboxTTSPlugin

server = AgentServer()


class VoiceAgent(Agent):
    pass


@server.rtc_session(agent_name="voice-agent-id")
async def entrypoint(ctx: JobContext):
    asr_endpoint = os.getenv("ASR_ENDPOINT", "http://localhost:18001")
    asr_api_key = os.getenv("ASR_API_KEY", "")
    asr_model = os.getenv("ASR_MODEL", "")
    tts_endpoint = os.getenv("TTS_ENDPOINT", "http://localhost:18003")

    # LLM: Bifrost gateway (Gemma 4 via llama.cpp, remote, OpenAI-compatible)
    bifrost_url = os.getenv("BIFROST_URL", "https://bifrost.hcm-lab.id/v1")
    bifrost_key = os.getenv("BIFROST_API_KEY", "")
    bifrost_model = os.getenv("BIFROST_MODEL", "llama.cpp/gemma-4-12b-it-q8")

    # ASR: custom STT wrapped with StreamAdapter (VAD -> buffer -> batch -> final)
    # Auto-detect: if ASR_API_KEY set + endpoint is OpenAI-compatible (groq/openai),
    # uses openai SDK. Otherwise hits local custom /v1/asr/transcribe.
    asr_stt = WhisperASRSTT(
        endpoint=asr_endpoint, language="id", api_key=asr_api_key, model=asr_model
    )
    print(f"[AGENT] ASR provider: {asr_stt.provider} ({asr_stt.model})")
    streaming_stt = stt_mod.StreamAdapter(
        stt=asr_stt,
        vad=silero.VAD.load(),
    )

    # LLM: Bifrost gateway (Gemma 4 via llama.cpp, remote, OpenAI-compatible)
    llm = openai.LLM(
        model=bifrost_model,
        base_url=bifrost_url,
        api_key=bifrost_key,
    )

    # TTS: Chatterbox
    tts_inst = ChatterboxTTSPlugin(endpoint=tts_endpoint)

    # VAD
    vad = silero.VAD.load()

    session = AgentSession(
        stt=streaming_stt,
        llm=llm,
        tts=tts_inst,
        vad=vad,
    )

    @session.on("user_speech_committed")
    def on_user_speech(msg):
        print(f"[USER] {msg.content}")

    @session.on("agent_speech_committed")
    def on_agent_speech(msg):
        print(f"[AGENT] {msg.content}")

    await session.start(
        agent=VoiceAgent(
            instructions=(
                "Anda adalah asisten suara AI berbahasa Indonesia yang natural dan ramah. "
                "Jawablah dengan singkat, jelas, dan menggunakan bahasa Indonesia yang baik. "
                "Gunakan gaya bicara percakapan sehari-hari, bukan formal atau kaku. "
                "Jika pengguna menyapa, balas dengan sapaan ramah. "
                "Bantu pengguna dengan pertanyaan atau permintaan mereka sebaik mungkin. "
                "Jika tidak tahu jawabannya, akui dengan jujur."
            ),
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(
                sync_transcription=False,
            ),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Sapa pengguna dalam Bahasa Indonesia dengan ramah dan singkat."
    )


if __name__ == "__main__":
    cli.run_app(server)
