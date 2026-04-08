#!/bin/sh
# Entrypoint.sh
# HF Spaces always uses port 7860. PORT env var can override for local dev.
PORT="${PORT:-7860}"

# Set RUN_INFERENCE=1 to run inference.py (for openenv validator).
# Default: start the uvicorn web server.
if [ "${RUN_INFERENCE:-0}" = "1" ]; then
    exec uv run python inference.py "$@"
else
    exec uv run uvicorn server.app:app --host 0.0.0.0 --port "$PORT"
fi