FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for faster-whisper (CTranslate2) and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY voice_agent.py .
COPY static/ static/

# Pre-download Whisper model (base) during build
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='float32')" 2>&1 || echo "Whisper model will be downloaded on first use"

# Default: run web server
CMD ["python", "main.py"]
