#!/bin/bash
# vim:sw=4:ts=4:et

set -e

if [ -z "${QWEN3_ASR_ENTRYPOINT_QUIET_LOGS:-}" ]; then
    exec 3>&1
else
    exec 3>/dev/null
fi

# Run initialization scripts if present
if /usr/bin/find "/docker-entrypoint.d/" -mindepth 1 -maxdepth 1 -type f -print -quit 2>/dev/null | read -r; then
    echo >&3 "$0: /docker-entrypoint.d/ is not empty, will attempt to perform configuration"

    echo >&3 "$0: Looking for shell scripts in /docker-entrypoint.d/"
    find "/docker-entrypoint.d/" -follow -type f -print | sort -V | while read -r f; do
        case "$f" in
        *.sh)
            if [ -x "$f" ]; then
                echo >&3 "$0: Launching $f"
                "$f"
            else
                echo >&3 "$0: Ignoring $f, not executable"
            fi
            ;;
        *) echo >&3 "$0: Ignoring $f" ;;
        esac
    done

    echo >&3 "$0: Configuration complete; ready for start up"
else
    echo >&3 "$0: No files found in /docker-entrypoint.d/, skipping configuration"
fi

# Default values from environment
QWEN3_ASR_MODEL_PATH="${QWEN3_ASR_MODEL_PATH:-Qwen/Qwen3-ASR-0.6B}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-28787}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MODEL_DTYPE="${MODEL_DTYPE:-auto}"
VAD_ENABLED="${VAD_ENABLED:-true}"
VAD_THRESHOLD="${VAD_THRESHOLD:-0.5}"
VAD_SILENCE_DURATION_MS="${VAD_SILENCE_DURATION_MS:-400}"
STREAMING_CHUNK_SIZE_SEC="${STREAMING_CHUNK_SIZE_SEC:-2.0}"
AUTO_COMMIT_INTERVAL_SEC="${AUTO_COMMIT_INTERVAL_SEC:-60.0}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo >&3 "============================================================"
echo >&3 "  Qwen3-ASR Realtime WebSocket Server"
echo >&3 "============================================================"
echo >&3 "  Model: ${QWEN3_ASR_MODEL_PATH}"
echo >&3 "  Server: ${SERVER_HOST}:${SERVER_PORT}"
echo >&3 "  GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo >&3 "  Model Dtype: ${MODEL_DTYPE}"
echo >&3 "  VAD: ${VAD_ENABLED} (threshold=${VAD_THRESHOLD}, silence=${VAD_SILENCE_DURATION_MS}ms)"
echo >&3 "============================================================"

# Resolve model path
# If relative path (starts with ./), convert to absolute
if [[ "${QWEN3_ASR_MODEL_PATH}" == ./* ]]; then
    QWEN3_ASR_MODEL_PATH="$(cd "$(dirname "${QWEN3_ASR_MODEL_PATH}")" && pwd)/$(basename "${QWEN3_ASR_MODEL_PATH}")"
    echo >&3 "$0: Resolved relative path to: ${QWEN3_ASR_MODEL_PATH}"
fi

# Export environment variables for the Python application
export QWEN3_ASR_MODEL_PATH
export SERVER_HOST
export SERVER_PORT
export GPU_MEMORY_UTILIZATION
export MAX_NEW_TOKENS
export MODEL_DTYPE
export VAD_ENABLED
export VAD_THRESHOLD
export VAD_SILENCE_DURATION_MS
export STREAMING_CHUNK_SIZE_SEC
export AUTO_COMMIT_INTERVAL_SEC
export LOG_LEVEL

echo >&3 "$0: Starting Qwen3-ASR server..."

# Start the server
exec python3 main.py
