# ======================================================
#   FunASR SenseVoiceSmall Inference Server
# ======================================================
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime@sha256:79c5599719e0b1afdb56ac2d14588b530283752d7ae6ec3c36e18ec9deb8b229

LABEL org.opencontainers.image.source="https://github.com/QwenAudio/SenseVoice" \
      org.opencontainers.image.description="SenseVoiceSmall FastAPI inference server" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libsndfile1 git python3-venv && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv --system-site-packages /opt/sensevoice-venv
ENV PATH=/opt/sensevoice-venv/bin:$PATH

# Copy only requirements first
COPY requirements.txt /app/

# Install dependencies (cached if requirements.txt didn't change)
RUN python -m pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your code
COPY . /app


# Optional: preload model weights during build (saves runtime download)
# RUN python -c "from funasr import AutoModel; AutoModel(model='iic/SenseVoiceSmall')"

# Expose FastAPI port
EXPOSE 50000

# Environment variables
ENV SENSEVOICE_DEVICE=auto
ENV PYTHONUNBUFFERED=1
ENV MODELSCOPE_CACHE=/models

# Create model cache directory (helps reuse between restarts)
RUN mkdir -p /models

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:50000/', timeout=3)" || exit 1

# Start FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "50000"]
