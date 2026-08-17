#!/usr/bin/env bash
set -e

echo "===================================================="
echo "               SwarmChat Launch Script"
echo "===================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[1/4] Installing/verifying Python dependencies..."
python3 -m pip install --quiet -r backend/requirements.txt

echo "[2/4] Running hardware & runtime diagnostics..."
python3 setup.py

echo "[3/4] Checking frontend build..."
if [ ! -d "frontend/dist" ]; then
    echo "Building frontend web interface..."
    (cd frontend && npm install && npm run build)
fi

echo "[4/4] Opening SwarmChat web interface..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000 &
elif command -v open &> /dev/null; then
    open http://localhost:8000 &
fi

echo ""
echo "===================================================="
echo "   SwarmChat Server is running on http://localhost:8000"
echo "   Press Ctrl+C to stop the server when done."
echo "===================================================="
echo ""

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
