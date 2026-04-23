@echo off
if not exist .venv\Scripts\python.exe (
  echo [ERROR] 未找到 .venv，请先执行 scripts\setup_windows.ps1
  exit /b 1
)

.venv\Scripts\python.exe -m pytest -q
