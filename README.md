# LiveKit ↔ Hermes Voice Bridge

Real-time voice conversation with Hermes Agent over LiveKit WebRTC.

## Architecture

```
User's Browser (WebRTC)
       │
       ▼  (audio stream)
   LiveKit Server ─── Voice Agent (STT → LLM → TTS)
       │                        │
       │                        └──→ Hermes API (DeepSeek v4 Flash)
       │
       ▼  (audio stream)
User's Browser
```

- **STT:** faster-whisper (local, same as Telegram)
- **LLM:** DeepSeek v4 Flash (via Hermes API)
- **TTS:** Edge TTS (local, same as Telegram)

## Deploy on Coolify

### Option 1: Full stack (LiveKit + Bridge + Agent)

1. Create a new Docker Compose resource in Coolify
2. Paste the contents of `docker-compose.yml`
3. Set environment variables:
   - `LIVEKIT_API_SECRET` — a random secret string
   - `PUBLIC_IP` — your server's public IP (for TURN)
   - `HERMES_API_URL` — your Hermes API server URL
   - `HERMES_API_KEY` — your Hermes API key (if enabled)
   - `HERMES_MODEL` — model name (default: `deepseek-v4-flash`)
4. Deploy
5. Access the web interface at `http://your-server:8000`

### Option 2: Bridge only (if you already have LiveKit)

1. Create a new Docker Compose resource in Coolify
2. Only include the `bridge` and `voice-agent` services
3. Point `LIVEKIT_HOST` and `LIVEKIT_WS_URL` at your existing LiveKit server

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Web server port |
| `LIVEKIT_HOST` | `http://livekit-server:7880` | LiveKit server REST API |
| `LIVEKIT_WS_URL` | `ws://livekit-server:7880` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | `devkey` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `secret` | LiveKit API secret |
| `HERMES_API_URL` | `http://hermes-api:8080/v1` | Hermes OpenAI-compatible API endpoint |
| `HERMES_API_KEY` | `` | Hermes API key (if enabled) |
| `HERMES_MODEL` | `deepseek-v4-flash` | LLM model name |
| `STT_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `TTS_VOICE` | `en-US-EricNeural` | Edge TTS voice |
| `TTS_RATE` | `+0%` | TTS speech rate |
| `AGENT_NAME` | `Hermes` | Agent display name |

## Hermes API Server

To use the Hermes API server (recommended for full agent capabilities):

1. Enable the API server in your Hermes config:
   ```yaml
   # In ~/.hermes/config.yaml
   # Set API_SERVER_HOST=0.0.0.0 and API_SERVER_KEY=your-key
   ```
2. Or run Hermes with: `API_SERVER_KEY=your-key hermes gateway run`
3. Set `HERMES_API_URL` to your Hermes API server URL

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run web server
python main.py

# Run voice agent (separate terminal)
python voice_agent.py

# Need a LiveKit server? Run one locally:
docker run -d -p 7880:7880 livekit/livekit-server:latest \
  --keys devkey:secret \
  --address 0.0.0.0
```
