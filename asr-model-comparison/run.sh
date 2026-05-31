#!/bin/bash
#
# ASR Model Comparison Platform - Single App Startup Script
#
# このスクリプトは常に「シングルアプリモード」で起動します。
# 1. フロントエンドを本番ビルド
# 2. ビルド成果物を backend/static/ に配置
# 3. バックエンド1プロセスだけで API + フロントエンド全体を配信
#
# セキュリティのため、デフォルトのホストは 127.0.0.1（localhost のみ）です。
# 外部からアクセスしたい場合は明示的に --host 0.0.0.0 を指定してください。
#
# Usage:
#   ./run.sh
#   ./run.sh --host 0.0.0.0 --port 8000
#   ./run.sh --port 9000
#

set -e

# Default values（安全のためデフォルトは localhost のみ）
HOST="127.0.0.1"
PORT="8000"

# 引数パース（getopts 風の堅牢版）
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --host requires a value"
                exit 1
            fi
            HOST="$2"
            shift 2
            ;;
        --port)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --port requires a value"
                exit 1
            fi
            PORT="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--build-only]"
            echo "  --host        Bind address (default: 127.0.0.1)"
            echo "  --port        Port number   (default: 8000)"
            echo "  --build-only  Build frontend only (do not start server)"
            echo ""
            echo "Logs are output to standard output (stdout) by default."
            echo "To save logs to a file, redirect output (e.g. ./run.sh > logs/app.log 2>&1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--port PORT] [--build-only]"
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=============================================================="
echo "  ASR Model Comparison Platform (AMCP) - Single App Mode"
echo "=============================================================="
echo ""

# Frontend build logic (inline)
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BACKEND_STATIC_DIR="$PROJECT_ROOT/backend/static"

echo ">>> Building frontend (Qwik City static build)..."

pushd "$FRONTEND_DIR" > /dev/null

if [ ! -d "node_modules" ]; then
    echo "    → Installing dependencies first..."
    npm install
fi

npm run build

popd > /dev/null

echo ">>> Copying build output to backend/static/ ..."

rm -rf "$BACKEND_STATIC_DIR"
mkdir -p "$BACKEND_STATIC_DIR"
cp -r "$FRONTEND_DIR/dist/"* "$BACKEND_STATIC_DIR/"

echo "Build complete!"

if [ "$BUILD_ONLY" = true ]; then
    echo "Build-only mode. Exiting without starting server."
    exit 0
fi

echo ""
echo ">>> Starting backend (serving both API and Frontend)..."
cd "$BACKEND_DIR"

# 仮想環境の検出と有効化
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "    → Using virtual environment (.venv)"
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "    → Using virtual environment (venv)"
else
    echo "    → No virtual environment detected (using system Python)"
fi

echo ""
echo "Starting server on http://${HOST}:${PORT}"
echo "Press Ctrl+C to stop."
echo ""

# 本番向け起動（--reload は付けない）
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
