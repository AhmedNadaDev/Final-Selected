"""
NovaCine — generation configuration (ZeroScope V2 576w + optional enhancements).

Enhancements are gated individually; set any ENABLE_* to False to disable that path.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

MODEL_ID = "cerspense/zeroscope_v2_576w"

WIDTH = 576
HEIGHT = 320

DEFAULT_WIDTH = WIDTH
DEFAULT_HEIGHT = HEIGHT

# ── Optional enhancements (each wrapped with `if ENABLE_*` in code paths) ──
ENABLE_COSINE_SCHEDULE = True
ENABLE_TEMPORAL_SMOOTHING = True
ENABLE_CLIP_RERANK = True

# Temporal blend strength (clamped to [0.05, 0.15] in enhancements.py)
TEMPORAL_SMOOTHING_LAMBDA = 0.08

# Used only when ENABLE_CLIP_RERANK; capped at 3 in code
CLIP_NUM_CANDIDATES = 2


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = (
        "blurry, distorted, low quality, watermark, text, worst quality"
    )
    num_frames: int = 24
    fps: int = 8
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None

    enable_audio: bool = False
    audio_script: Optional[str] = None
    audio_voice: str = "en-US-AriaNeural"
    audio_rate: str = "+0%"
    audio_pitch: str = "+0Hz"
    audio_emotion: str = "neutral"

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
