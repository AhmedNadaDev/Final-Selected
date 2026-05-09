"""
NovaCine API Routes
"""
from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List

from core.generator import GenerationConfig

router = APIRouter()


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    negative_prompt: str = "blurry, distorted, low quality, worst quality"
    num_frames: int = Field(32, ge=8, le=64)
    fps: int = Field(8, ge=4, le=30)
    width: int = Field(256, ge=128, le=512)
    height: int = Field(256, ge=128, le=512)
    num_inference_steps: int = Field(25, ge=5, le=100)
    guidance_scale: float = Field(9.0, ge=1.0, le=20.0)
    seed: Optional[int] = None
    use_ddim: bool = True
    use_enhancement_a: bool = True
    use_enhancement_b: bool = True
    num_candidates: int = Field(2, ge=1, le=4)

    # ── Bonus: Text-to-Audio ───────────────────────────────
    enable_audio: bool = False
    audio_script: Optional[str] = Field(None, max_length=2000)
    audio_voice: str = "en-US-AriaNeural"
    audio_rate: str = "+0%"
    audio_pitch: str = "+0Hz"
    audio_emotion: str = "neutral"


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    total_steps: int
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class VoiceInfo(BaseModel):
    id: str
    lang: str
    gender: str
    label: str


class TTSCatalogResponse(BaseModel):
    voices:   List[VoiceInfo]
    emotions: List[str]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate_video(req: GenerateRequest, request: Request):
    """Submit a new video generation job."""
    queue = request.app.state.queue
    ws_manager = request.app.state.ws_manager
    queue.set_ws_manager(ws_manager)

    job_id = str(uuid.uuid4())
    config = GenerationConfig(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_frames=req.num_frames,
        fps=req.fps,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        use_ddim=req.use_ddim,
        use_enhancement_a=req.use_enhancement_a,
        use_enhancement_b=req.use_enhancement_b,
        num_candidates=req.num_candidates,
        enable_audio=req.enable_audio,
        audio_script=req.audio_script,
        audio_voice=req.audio_voice,
        audio_rate=req.audio_rate,
        audio_pitch=req.audio_pitch,
        audio_emotion=req.audio_emotion,
        job_id=job_id,
    )

    job = await queue.enqueue(config)
    return GenerateResponse(job_id=job.job_id, status="queued", message="Job queued successfully")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    """Get status and result of a generation job."""
    queue = request.app.state.queue
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        total_steps=job.total_steps,
        message=job.message,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs")
async def list_jobs(request: Request):
    """List all jobs."""
    queue = request.app.state.queue
    return {"jobs": queue.get_all_jobs()}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, request: Request):
    """Remove a job from history."""
    queue = request.app.state.queue
    if job_id not in queue.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del queue.jobs[job_id]
    return {"message": "Job deleted"}


# ──────────────────────────────────────────────
# Bonus TTS: voices & emotions catalog
# ──────────────────────────────────────────────

@router.get("/tts/catalog", response_model=TTSCatalogResponse)
async def tts_catalog():
    """Return available neural voices + supported emotions."""
    try:
        from utils.tts import NeuralTTS
        return TTSCatalogResponse(
            voices=NeuralTTS.list_voices(),
            emotions=NeuralTTS.list_emotions(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"TTS engine unavailable: {e}",
        )
