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
# --reload / -r を付けると開発モードになります (uvicorn --reload + フロントエンドソース変更時の自動再ビルド)。
#
# Usage:
#   ./run.sh
#   ./run.sh --host 0.0.0.0 --port 8000
#   ./run.sh --port 9000
#   ./run.sh --reload
#   ./run.sh --reload --host 0.0.0.0
#

set -e

# Default values（安全のためデフォルトは localhost のみ）
HOST="127.0.0.1"
PORT="8000"

# 引数パース（getopts 風の堅牢版）
BUILD_ONLY=false
RELOAD=false

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
        --reload|-r)
            RELOAD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--build-only] [--reload]"
            echo "  --host        Bind address (default: 127.0.0.1)"
            echo "  --port        Port number   (default: 8000)"
            echo "  --build-only  Build frontend only (do not start server)"
            echo "  --reload, -r  Development mode: uvicorn --reload + auto-rebuild frontend on source changes"
            echo ""
            echo "Logs are output to standard output (stdout) by default."
            echo "To save logs to a file, redirect output (e.g. ./run.sh > logs/app.log 2>&1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--port PORT] [--build-only] [--reload]"
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

echo ">>> Building frontend (Pure Qwik + Vite static build)..."

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

# In reload mode, start a background subshell watcher that auto-rebuilds frontend on source changes.
# We use find -newer against a marker file (portable, no inotify required).
if [ "$RELOAD" = true ]; then
    echo "RELOAD MODE: will use uvicorn --reload and background frontend watcher."
    touch /tmp/.amcp_last_frontend_build 2>/dev/null || true

    (
        # watcher subshell
        set +e
        echo "[sh-watcher] Monitoring $FRONTEND_DIR/src and public for changes..."
        cd "$FRONTEND_DIR" || exit 1
        while true; do
            sleep 2
            CHANGED=$(find src public -type f -newer /tmp/.amcp_last_frontend_build 2>/dev/null | head -1 || true)
            if [ -n "$CHANGED" ]; then
                echo "[sh-watcher] Change detected ($CHANGED). Rebuilding..."
                npm run build 2>&1 | sed 's/^/[build] /'
                rm -rf "$BACKEND_STATIC_DIR"
                mkdir -p "$BACKEND_STATIC_DIR"
                cp -r "$FRONTEND_DIR/dist/"* "$BACKEND_STATIC_DIR/"
                touch /tmp/.amcp_last_frontend_build 2>/dev/null || true
                echo "[sh-watcher] Rebuild + copy done. Static files are live."
            fi
        done
    ) &
    WATCHER_PID=$!

    # Ensure we kill the watcher when this script exits (Ctrl+C etc)
    trap 'echo "Stopping watcher (pid $WATCHER_PID)"; kill $WATCHER_PID 2>/dev/null || true; exit' INT TERM EXIT
fi

# pushd here (matching the PowerShell version) so directory intent is clear.
# Note: exec below replaces the process, so no matching popd is performed.
# When run as a script (not sourced), the caller's shell pwd is unaffected anyway.
pushd "$BACKEND_DIR" > /dev/null || exit 1

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
if [ "$RELOAD" = true ]; then
    echo "(reload mode: uvicorn will auto-restart on .py changes)"
fi
echo "Press Ctrl+C to stop."
echo ""

RELOAD_FLAG=""
if [ "$RELOAD" = true ]; then
    RELOAD_FLAG="--reload"
fi

# 起動（--reload フラグが付いている場合は開発モードで uvicorn も --reload）
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
