@echo off
title SwarmChat - 1-Click Launch & Setup
echo ====================================================
echo               SwarmChat Launch Script
echo ====================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please download and install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking and installing Python dependencies...
python -m pip install --quiet -r backend\requirements.txt

echo [2/4] Running hardware & runtime diagnostic checks...
python setup.py

echo [3/4] Checking frontend build...
if not exist "frontend\dist" (
    echo Building frontend web interface...
    cd frontend
    call npm install
    call npm run build
    cd ..
)

echo [4/4] Opening SwarmChat in your default web browser...
start http://localhost:8000

echo.
echo ====================================================
echo   SwarmChat Server is running on http://localhost:8000
echo   Press Ctrl+C to stop the server when done.
echo ====================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
