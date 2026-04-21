@echo off
setlocal

if "%~1"=="" (
  set PYTHON_CMD=py -3
) else (
  set PYTHON_CMD=%*
)

echo [1/4] Create virtual environment (.venv)
%PYTHON_CMD% -m venv .venv
if errorlevel 1 exit /b 1

echo [2/4] Install/upgrade pip
.venv\Scripts\python.exe -m pip install -U pip
if errorlevel 1 exit /b 1

echo [3/4] Install requirements
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/4] Setup complete

echo Run demo: .venv\Scripts\python.exe -m src.main --scenario s1_close_threat --steps 20
endlocal
