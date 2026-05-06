#!/usr/bin/env python3
"""
LiveKit ↔ Hermes Voice Bridge
Web calling interface for real-time voice conversation with Hermes Agent.
"""

import os
import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from livekit import api
from livekit.api import LiveKitAPI, AccessToken, RoomConfiguration, VideoGrants

logger = logging.getLogger("livekit-hermes-bridge")

# ── Config ──────────────────────────────────────────────────────────────────
LIVEKIT_HOST = os.environ.get("LIVEKIT_HOST", "http://livekit-server:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
LIVEKIT_WS_URL = os.environ.get("LIVEKIT_WS_URL", "ws://livekit-server:7880")

HERMES_API_URL = os.environ.get("HERMES_API_URL", "http://hermes-api:8080/v1")
HERMES_API_KEY = os.environ.get("HERMES_API_KEY", "")
HERMES_MODEL = os.environ.get("HERMES_MODEL", "deepseek-v4-flash")
HERMES_PROVIDER = os.environ.get("HERMES_PROVIDER", "opencode")

STT_MODEL = os.environ.get("STT_MODEL", "base")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-EricNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+0%")

AGENT_NAME = os.environ.get("AGENT_NAME", "Hermes")
LIVEKIT_AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "hermes-voice")

# ── FastAPI app ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LiveKit-Hermes Bridge")
    yield
    logger.info("Shutting down LiveKit-Hermes Bridge")

app = FastAPI(title="LiveKit ↔ Hermes Voice Bridge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class CreateRoomRequest(BaseModel):
    room_name: str = ""  # auto-generated if empty


class CreateRoomResponse(BaseModel):
    room_name: str
    token: str
    ws_url: str


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web call interface."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>LiveKit-Hermes Bridge</h1><p>Static files not found.</p>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/room", response_model=CreateRoomResponse)
async def create_room(req: CreateRoomRequest):
    """Create a LiveKit room and return an access token."""
    import uuid
    room_name = req.room_name or f"hermes-{uuid.uuid4().hex[:8]}"

    async with LiveKitAPI(
        url=LIVEKIT_HOST,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    ) as lk:
        # Create the room
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=300,
                max_participants=2,
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=LIVEKIT_AGENT_NAME,
                    )
                ],
            )
        )
        logger.info(f"Created room: {room_name}")

        # Generate token for the user
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(f"user-{uuid.uuid4().hex[:8]}") \
            .with_name("You") \
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )).to_jwt()

    return CreateRoomResponse(
        room_name=room_name,
        token=token,
        ws_url=LIVEKIT_WS_URL,
    )


@app.get("/api/config")
async def get_config():
    """Expose bridge configuration to the web client."""
    return {
        "ws_url": LIVEKIT_WS_URL,
        "agent_name": AGENT_NAME,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
