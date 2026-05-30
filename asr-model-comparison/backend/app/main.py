"""
AMCP Backend - FastAPI entry point.

Design principles (per CLAUDE.md & specs):
- Only ONE ASR model loaded in memory at any time.
- All heavy model loading is delegated to the model service.
- Processing time is measured for every transcription request.
- No Gemma3/Bonsai post-processing is performed.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import models_router, transcribe_router

app = FastAPI(
    title="ASR Model Comparison Platform (AMCP) API",
    description="Compare Whisper, Qwen3-ASR, and Voxtral models on identical audio samples.",
    version="0.1.0",
)

# Allow local frontend development (Qwik dev server)
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

# Include routers
app.include_router(models_router)
app.include_router(transcribe_router)


@app.get("/health")
async def health_check():
    """Simple health check for load balancers / monitoring."""
    return {"status": "ok", "service": "amcp-backend"}


# Note: The actual model management service and /api/transcribe will be added
# in subsequent TDD cycles (after model service tests are written).
