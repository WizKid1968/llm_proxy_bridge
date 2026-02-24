@echo off
cd /d "%~dp0"
echo Starting Kimi Proxy (Standalone)...
echo Use http://127.0.0.1:8080/v1 in your app.

uv run ^
  --with fastapi ^
  --with uvicorn ^
  --with httpx ^
  --with sentence-transformers ^
  --with python-dotenv ^
  python main.py

pause
