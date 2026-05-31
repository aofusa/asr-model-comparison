"""
AMCP Backend - FastAPI entry point.

Design principles (per CLAUDE.md & specs):
- Only ONE ASR model loaded in memory at any time.
- All heavy model loading is delegated to the model service.
- Processing time is measured for every transcription request.
- No Gemma3/Bonsai post-processing is performed.
"""
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.api.routers import models_router, status_router, transcribe_router

app = FastAPI(
    title="ASR Model Comparison Platform (AMCP) API",
    description="Compare Whisper, Qwen3-ASR, and Voxtral models on identical audio samples.",
    version="0.1.0",
)

# Allow local frontend development (Qwik dev server)
# In single-app production mode these can be removed or restricted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (these must come before static/SPA handlers)
app.include_router(models_router)
app.include_router(transcribe_router)
app.include_router(status_router)

# Debug print
print("=== Registered routes (debug) ===", flush=True)
for route in app.routes:
    if hasattr(route, "path"):
        methods = getattr(route, "methods", None) or ["WS" if "websocket" in str(type(route)).lower() else "GET"]
        print(f"  {methods} {route.path}", flush=True)


@app.get("/health")
async def health_check():
    """Simple health check for load balancers / monitoring."""
    return {"status": "ok", "service": "amcp-backend"}


# ============================================================
# Production Static File Serving (Single App Mode)
# ============================================================
# When the frontend is built and copied into ../static/, the backend
# serves the entire SPA from a single process.
#
# Build flow:
#   frontend/  →  npm run build  →  copied into backend/static/
#
# Then run:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# The frontend will be available at http://localhost:8000/
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.isdir(STATIC_DIR):
    # Mount common asset folders produced by Qwik/Vite build
    for asset_folder in ["assets", "build", "chunks"]:
        folder_path = os.path.join(STATIC_DIR, asset_folder)
        if os.path.isdir(folder_path):
            app.mount(f"/{asset_folder}", StaticFiles(directory=folder_path), name=asset_folder)

    # Serve other root-level files (manifest.json, favicon, etc.)
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static_root")

# Use middleware for SPA fallback instead of a catch-all route.
# This is more reliable with WebSocket and other non-GET protocols.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SPAFallbackMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip API, WebSocket upgrade, health, and static paths
        path = request.url.path
        if (
            path.startswith("/api/")
            or path.startswith("/ws/")
            or path == "/health"
            or path.startswith("/static/")
            or path.startswith("/assets/")
            or path.startswith("/build/")
            or path.startswith("/chunks/")
        ):
            return await call_next(request)

        # For all other GET requests, serve the SPA index.html if it exists
        if request.method == "GET" and os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
            return FileResponse(os.path.join(STATIC_DIR, "index.html"))

        return await call_next(request)

if os.path.isdir(STATIC_DIR):
    app.add_middleware(SPAFallbackMiddleware)
