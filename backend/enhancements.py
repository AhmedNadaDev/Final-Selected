"""
Optional enhancements for ZeroScope — each path gated by ENABLE_* in config.py.

- Cosine schedule: applied once when building the scheduler (not per denoising step).
- Temporal smoothing: small blend along time (lambda clamped 0.05–0.15), non-destructive mix.
- CLIP reranking: full forward passes per candidate, select best after all complete.

Safety: only torch.nan_to_num on latent tensors after optional ops (no heavy repair).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler
from loguru import logger
from PIL import Image

import config as cfg


def _clamp_lambda(v: float) -> float:
    return float(min(max(v, 0.05), 0.15))


def make_cosine_alphas_cumprod(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps + s) / (1 + s)) * (np.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5, max=1.0 - 1e-5)
    return alphas_cumprod[1:].float()


def _apply_cosine_to_scheduler(scheduler) -> None:
    """Single init-time patch to alphas_cumprod (not called each step)."""
    n = len(scheduler.alphas_cumprod)
    cosine_ac = make_cosine_alphas_cumprod(n).to(
        device=scheduler.alphas_cumprod.device,
        dtype=scheduler.alphas_cumprod.dtype,
    )
    scheduler.alphas_cumprod = cosine_ac
    logger.info("Enhancement: cosine alphas_cumprod applied (scheduler init only)")


def build_scheduler(pipe):
    """
    Build DPMSolverMultistepScheduler from pipeline config.
    Cosine schedule is applied at most once here when ENABLE_COSINE_SCHEDULE.
    """
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if cfg.ENABLE_COSINE_SCHEDULE:
        _apply_cosine_to_scheduler(scheduler)
    return scheduler


def blend_temporal_latents(lat: torch.Tensor) -> torch.Tensor:
    """
    Gentle temporal blend with previous frame (non-destructive mix), lambda clamped.
    """
    if not cfg.ENABLE_TEMPORAL_SMOOTHING:
        return lat
    if lat.dim() < 5 or lat.shape[2] <= 1:
        return lat
    w = _clamp_lambda(cfg.TEMPORAL_SMOOTHING_LAMBDA)
    out = lat.clone()
    prev = out[:, :, :-1]
    curr = out[:, :, 1:]
    out[:, :, 1:] = (1.0 - w) * curr + w * prev
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)


def clip_num_candidates() -> int:
    """Clamp to [1, 3]; reranking only meaningful when >= 2."""
    n = int(cfg.CLIP_NUM_CANDIDATES)
    return max(1, min(n, 3))


class CLIPReranker:
    def __init__(self, device: str):
        self.device = device
        self._model = None
        self._preprocess = None
        self._tried = False

    def _lazy_load(self):
        if self._model is not None or self._tried:
            return
        self._tried = True
        try:
            import clip

            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self._model.eval()
            logger.info("CLIP reranker loaded")
        except Exception as e:
            logger.warning(f"CLIP unavailable — reranking skipped ({e})")

    def score_frames(self, pil_frames: list[Image.Image], prompt: str) -> float:
        self._lazy_load()
        if self._model is None or not pil_frames:
            return 0.0
        try:
            import clip

            rgb = [f.convert("RGB") for f in pil_frames]
            idx = [
                max(0, len(rgb) // 4),
                len(rgb) // 2,
                min(len(rgb) - 1, 3 * len(rgb) // 4),
            ]
            text_tokens = clip.tokenize([prompt], truncate=True).to(self.device)
            with torch.no_grad():
                txt_feat = F.normalize(self._model.encode_text(text_tokens), dim=-1)
                sims = []
                for i in idx:
                    t = self._preprocess(rgb[i]).unsqueeze(0).to(self.device)
                    img_feat = F.normalize(self._model.encode_image(t), dim=-1)
                    sims.append((img_feat * txt_feat).sum().item())
            return float(np.mean(sims))
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.0

    def select_best(
        self,
        candidates: list[list[Image.Image]],
        prompt: str,
    ) -> tuple[list[Image.Image], list[float]]:
        scores = [self.score_frames(c, prompt) for c in candidates]
        best_idx = int(np.argmax(scores)) if scores else 0
        logger.info(
            f"CLIP rerank scores: {[f'{s:.4f}' for s in scores]} → pick [{best_idx}]"
        )
        return candidates[best_idx], scores
