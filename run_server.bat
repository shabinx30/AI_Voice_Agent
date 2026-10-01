@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_server.py %*
) else (
    uv run python run_server.py %*
)
