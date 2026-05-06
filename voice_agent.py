#!/usr/bin/env python3
"""
LiveKit Voice Agent (API v1.5) — the worker process that handles voice conversations.
Runs as a separate process, joins rooms when dispatched, and processes
audio through STT → LLM → TTS pipeline.
"""

import os
import sys
import json
import logging
import asyncio
import struct
from typing import Optional

import numpy as np

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm as llm_module,
    stt as stt_module,
    tts as tts_module,
    vad as vad_module,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import silero

logger = logging.getLogger("voice-agent")

# ── Config ──────────────────────────────────────────────────────────────────
HERMES_API_URL = os.environ.get("HERMES_API_URL", "http://hermes-api:8080/v1")
HERMES_API_KEY = os.environ.get("HERMES_API_KEY", "")
HERMES_MODEL = os.environ.get("HERMES_MODEL", "deepseek-v4-flash")
STT_MODEL = os.environ.get("STT_MODEL", "base")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-EricNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+0%")
AGENT_NAME = os.environ.get("AGENT_NAME", "Hermes")
LIVEKIT_AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "hermes-voice")


def prewarm(proc: JobProcess):
    """Called once per process to pre-warm models."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt_model_size"] = STT_MODEL


async def entrypoint(job: JobContext):
    """Main entrypoint when a voice agent is dispatched to a room."""
    logger.info(f"Joining room: {job.room.name}")

    # ── LLM — OpenAI-compatible, pointed at Hermes API ─────
    from openai import AsyncOpenAI

    # The OpenAI client requires an API key even for OpenAI-compatible backends
    # that do not enforce authentication. Use a placeholder when Hermes auth is off.
    openai_client = AsyncOpenAI(
        base_url=HERMES_API_URL,
        api_key=HERMES_API_KEY or "dummy",
    )

    class HermesLLM(llm_module.LLM):
        """Custom LLM that talks to Hermes' API server."""

        def __init__(self):
            super().__init__()
            self._client = openai_client

        async def chat(
            self,
            messages: list[llm_module.ChatMessage],
            temperature: Optional[float] = None,
            n: Optional[int] = 1,
            parallel_tool_calls: Optional[bool] = False,
        ) -> "llm_module.ChatResponse":
            openai_messages = []
            for msg in messages:
                role = msg.role if msg.role in ("user", "assistant", "system") else "user"
                openai_messages.append({"role": role, "content": msg.text or ""})

            response = await self._client.chat.completions.create(
                model=HERMES_MODEL,
                messages=openai_messages,
                temperature=temperature or 0.7,
                stream=False,
            )

            choice = response.choices[0]
            return llm_module.ChatResponse(
                message=llm_module.ChatMessage(
                    role="assistant",
                    text=choice.message.content or "",
                ),
                usage=(
                    llm_module.LLMUsage(
                        total_tokens=response.usage.total_tokens if response.usage else 0,
                    )
                    if response.usage
                    else None
                ),
            )

        async def chat_stream(
            self,
            messages: list[llm_module.ChatMessage],
            temperature: Optional[float] = None,
            n: Optional[int] = 1,
            parallel_tool_calls: Optional[bool] = False,
        ):
            """Streaming version for real-time responses."""
            openai_messages = []
            for msg in messages:
                role = msg.role if msg.role in ("user", "assistant", "system") else "user"
                openai_messages.append({"role": role, "content": msg.text or ""})

            stream = await self._client.chat.completions.create(
                model=HERMES_MODEL,
                messages=openai_messages,
                temperature=temperature or 0.7,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield llm_module.ChatChunk(
                            choices=[
                                llm_module.Choice(
                                    delta=llm_module.ChatMessage(
                                        role="assistant",
                                        text=delta.content,
                                    )
                                )
                            ]
                        )

            yield llm_module.ChatChunk(
                choices=[
                    llm_module.Choice(
                        delta=llm_module.ChatMessage(role="assistant", text=""),
                        finish_reason="stop",
                    )
                ]
            )

    # ── STT — Local Whisper via faster-whisper ──────────────
    from livekit.agents.stt import (
        STT,
        SpeechEvent,
        SpeechData,
        SpeechEventType,
        STTCapabilities,
    )
    from livekit.agents.types import (
        DEFAULT_API_CONNECT_OPTIONS,
        NOT_GIVEN,
        APIConnectOptions,
    )
    from livekit.agents.utils.audio import AudioBuffer, combine_frames

    class WhisperSTT(STT):
        def __init__(self, model_size: str = "base"):
            super().__init__(
                capabilities=STTCapabilities(
                    streaming=False,
                    interim_results=False,
                )
            )
            self._model_size = model_size
            self._model = None

        def _ensure_model(self):
            if self._model is None:
                from faster_whisper import WhisperModel
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
                logger.info(f"Loading faster-whisper {self._model_size} on {device}...")
                self._model = WhisperModel(self._model_size, device=device, compute_type="float32")

        async def _recognize_impl(
            self,
            buffer: AudioBuffer,
            *,
            language=NOT_GIVEN,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        ) -> SpeechEvent:
            self._ensure_model()
            frame = combine_frames(buffer)
            audio_int16 = np.frombuffer(frame.data, dtype=np.int16)
            language_value = None if language is NOT_GIVEN else str(language)
            loop = asyncio.get_event_loop()

            segments, info = await loop.run_in_executor(
                None,
                lambda: list(
                    self._model.transcribe(
                        audio_int16,
                        language=language_value,
                        beam_size=5,
                    )
                ),
            )

            text = " ".join(seg.text for seg in segments)
            logger.info(f"STT: {text}")
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=text, language=info.language or "en")],
            )

    # ── TTS — Edge TTS ──────────────────────────────────────
    from livekit.agents.tts import ChunkedStream, TTS, TTSCapabilities

    class EdgeTTS(TTS):
        def __init__(self, voice: str = "en-US-EricNeural", rate: str = "+0%"):
            super().__init__(
                capabilities=TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )
            self._voice = voice
            self._rate = rate

        def synthesize(
            self,
            text: str,
            *,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        ) -> ChunkedStream:
            parent = self

            class _EdgeChunkedStream(ChunkedStream):
                async def _run(self, output_emitter) -> None:
                    import io
                    import edge_tts
                    from pydub import AudioSegment
                    from livekit.agents.utils import shortuuid

                    communicate = edge_tts.Communicate(
                        self.input_text, parent._voice, rate=parent._rate
                    )
                    audio_chunks = []
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_chunks.append(chunk["data"])

                    output_emitter.initialize(
                        request_id=shortuuid(),
                        sample_rate=parent.sample_rate,
                        num_channels=parent.num_channels,
                        mime_type="audio/pcm",
                        stream=False,
                    )

                    if not audio_chunks:
                        return

                    combined = AudioSegment.empty()
                    for audio_data in audio_chunks:
                        seg = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                        combined += seg

                    raw = combined.set_frame_rate(parent.sample_rate).set_channels(
                        parent.num_channels
                    ).raw_data
                    logger.info(f"TTS: {len(raw)} bytes, {len(self.input_text)} chars")
                    output_emitter.push(raw)
                    output_emitter.flush()

            return _EdgeChunkedStream(
                tts=self,
                input_text=text,
                conn_options=conn_options,
            )

    # ── Build the voice pipeline using new API ──────────────
    vad = job.proc.userdata["vad"]

    # Configure turn handling for steer mode
    from livekit.agents.voice.turn import (
        TurnHandlingOptions,
        EndpointingOptions,
        InterruptionOptions,
        PreemptiveGenerationOptions,
    )

    turn_handling = TurnHandlingOptions(
        interruption=InterruptionOptions(
            enabled=True,
            mode="adaptive",
            min_duration=0.3,
            min_words=0,
            discard_audio_if_uninterruptible=False,
            resume_false_interruption=True,
            false_interruption_timeout=1.0,
        ),
        endpointing=EndpointingOptions(
            mode="dynamic",
            min_delay=0.6,
            max_delay=2.0,
        ),
        preemptive_generation=PreemptiveGenerationOptions(
            enabled=True,
            preemptive_tts=True,
        ),
    )

    # Create the session with STT, VAD, LLM, TTS and steer
    session = AgentSession(
        vad=vad,
        stt=WhisperSTT(model_size=STT_MODEL),
        llm=HermesLLM(),
        tts=EdgeTTS(voice=TTS_VOICE, rate=TTS_RATE),
        turn_handling=turn_handling,
    )

    # Create the voice agent with instructions
    agent = Agent(
        instructions=(
            f"You are {AGENT_NAME}, an AI voice assistant running on the Hermes Agent platform. "
            "You are having a real-time voice conversation with the user. "
            "Keep responses concise and conversational since this is a voice call. "
            "Be helpful, natural, and engaging.\n\n"
            "IMPORTANT — Response style rules:\n"
            "- Only speak the final answer. Never describe your internal thinking, reasoning steps, "
            "tool calls, function calls, or any intermediate processing.\n"
            "- Never mention that you are using tools, searching the web, accessing memory, or "
            "running any internal processes.\n"
            "- Just respond directly as if the answer came naturally to you.\n"
            "- Keep responses short and conversational — this is a real-time voice call.\n"
            "- If you don't know something, say so simply without over-explaining."
        ),
        allow_interruptions=True,
    )

    # Connect to the room and start the session
    await job.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info(f"Starting session in room: {job.room.name}")
    result = await session.start(agent, room=job.room)
    logger.info(f"Session ended: {job.room.name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=LIVEKIT_AGENT_NAME,
        )
    )
