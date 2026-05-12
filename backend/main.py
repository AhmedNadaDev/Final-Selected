"""
NovaCine Backend — FastAPI Application
Text-to-Video Generation API with async queue and WebSocket progress streaming
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger
import os

from generator import VideoGenerator
from core.queue_manager import GenerationQueue
from api.routes import router

# ──────────────────────────────────────────────
# Application lifecycle
# ──────────────────────────────────────────────

generator: Optional[VideoGenerator] = None
queue: Optional[GenerationQueue] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator, queue
    logger.info("Initializing NovaCine backend...")
    generator = VideoGenerator()
    queue = GenerationQueue(generator)
    await queue.start()
    app.state.generator = generator
    app.state.queue = queue
    logger.info("Backend ready")
    yield
    logger.info("Shutting down...")
    await queue.stop()


# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="NovaCine API",
    description="Text-to-Video Generation via Latent Diffusion Models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static output directory for video serving
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(router, prefix="/api/v1")


# ──────────────────────────────────────────────
# WebSocket for real-time progress
# ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id] = ws

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def send_progress(self, job_id: str, data: dict):
        ws = self.active.get(job_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(job_id)


ws_manager = ConnectionManager()
app.state.ws_manager = ws_manager


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            # Keep connection alive; progress pushed from generator callbacks
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
