# Qwen3-ASR Realtime Server
#
# This image provides a ready-to-use WebSocket server for Qwen3-ASR streaming transcription.
# Based on vLLM 0.14.0 with all dependencies pre-installed.
#
# Build Args:
#   VLLM_VERSION - vLLM version (default: 0.14.0, MUST be 0.14.0 for qwen-asr compatibility)
#
# Environment Variables:
#   QWEN3_ASR_MODEL_PATH     - HuggingFace model ID or local path (default: Qwen/Qwen3-ASR-0.6B)
#   SERVER_HOST              - Server bind address (default: 0.0.0.0)
#   SERVER_PORT              - Server port (default: 8080)
#   GPU_MEMORY_UTILIZATION   - GPU memory utilization 0.0-1.0 (default: 0.8)
#   MAX_NEW_TOKENS           - Max tokens per inference (default: 64)
#   MODEL_DTYPE              - Model dtype: auto/half/float16/bfloat16 (default: auto)
#   VAD_ENABLED              - Enable VAD (default: true)
#   VAD_THRESHOLD            - VAD threshold (default: 0.5)
#   VAD_SILENCE_DURATION_MS  - Silence duration in ms (default: 400)
#   STREAMING_CHUNK_SIZE_SEC - Audio chunk size in seconds (default: 2.0)
#   AUTO_COMMIT_INTERVAL_SEC - Auto commit interval in seconds (default: 60.0)
#   LOG_LEVEL                - Log level (default: info)

ARG VLLM_VERSION=0.14.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

LABEL maintainer="Rookie_Zoe <i@rookiezoe.com>"
LABEL org.opencontainers.image.source="https://github.com/RookieZoe/qwen3-asr-realtime-api"
LABEL org.opencontainers.image.description="Qwen3-ASR Realtime WebSocket Server with vLLM backend"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (skip torch/vllm as they're in base image)
# Use --no-build-isolation to use existing packages
RUN set -ex \
    && uv pip install --system --no-cache-dir \
        fastapi uvicorn websockets numpy pydantic python-dotenv \
        silero-vad accelerate librosa soundfile \
        qwen-asr qwen-omni-utils transformers

# Copy application code
COPY main.py ./
COPY src/ ./src/

# Setup entrypoint
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN set -ex \
    && mkdir -p /docker-entrypoint.d \
    && chmod +x /docker-entrypoint.sh

# Create models directory for local model mounting
RUN mkdir -p /app/models

# Environment defaults
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

EXPOSE 8080

ENTRYPOINT ["/docker-entrypoint.sh"]
