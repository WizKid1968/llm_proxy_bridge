#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Kimi Proxy (Standalone)..."
echo "Use http://127.0.0.1:8080/v1 in your app."
# Use uv to run with dependencies on the fly
/opt/homebrew/bin/uv run \
  --with fastapi \
  --with uvicorn \
  --with httpx \
  --with sentence-transformers \
  --with python-dotenv \
  python proxy_vision_v4.py
