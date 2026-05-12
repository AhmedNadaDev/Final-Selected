"""
NovaCine — Core Video Generator
Implements the full latent-diffusion video pipeline:

  - 3D U-Net backbone (ModelScope T2V)
  - Temporal attention modules
  - Classifier-free guidance (CFG)
  - DDPM / DDIM sampling
  - Enhancement A: Cosine schedule + temporal smoothing of latents
  - Enhancement B: CLIP-based reranking of N candidate clips
  - Bonus  : Optional Text-to-Audio (Edge Neural TTS) muxed via ffmpeg
"""
from __future__ import annotations

import inspect
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler
from diffusers.utils import export_to_video
from loguru import logger
from PIL import Image


def _frame_to_rgb_uint8(
    frame: np.ndarray | Image.Image,
    *,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Convert a diffusion / PIL frame to HWC uint8 RGB for PIL Video & CLIP.

    Handles float [0,1], [-1,1], CHW tensors, NaN/Inf, and degenerate sizes.
    """
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    else:
        x = np.asarray(frame)
        if x.dtype == np.uint8 and x.ndim == 3 and x.shape[-1] >= 3:
            arr = x[..., :3].copy()
        else:
            x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
            if x.ndim == 2:
                x = np.stack([x, x, x], axis=-1)
            elif x.ndim == 3:
                if x.shape[-1] in (3, 4):
                    if x.shape[-1] == 4:
                        x = x[..., :3]
                elif x.shape[0] in (1, 3, 4) and max(x.shape[1], x.shape[2]) > 4:
                    if x.shape[0] == 1:
                        x = np.repeat(x, 3, axis=0)
                    elif x.shape[0] == 4:
                        x = x[:3]
                    x = np.transpose(x, (1, 2, 0))
                else:
                    x = np.transpose(x, (1, 2, 0)) if x.shape[0] == 3 else x
            else:
                x = np.squeeze(x)
                if x.ndim != 3:
                    x = np.zeros((target_h, target_w, 3), dtype=np.float64)

            vmax, vmin = float(np.nanmax(x)), float(np.nanmin(x))
            if vmax <= 1.0 + 1e-3 and vmin >= -1.0 - 1e-3:
                if vmin < -1e-3:
                    x = (x + 1.0) * 0.5
                x = np.clip(x, 0.0, 1.0) * 255.0
            elif vmax <= 1.0 + 1e-3 and vmin >= 0.0:
                x = np.clip(x, 0.0, 1.0) * 255.0
            else:
                x = np.clip(x, 0.0, 255.0)
            arr = np.round(x).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = arr.shape[0], arr.shape[1]
    if h < 2 or w < 2 or np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0)
        pil_s = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        pil_s = pil_s.resize((max(w, 2), max(h, 2)), Image.Resampling.NEAREST)
        arr = np.asarray(pil_s.convert("RGB"), dtype=np.uint8)

    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        pil_r = Image.fromarray(arr, mode="RGB").resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )
        arr = np.asarray(pil_r.convert("RGB"), dtype=np.uint8)

    return arr


def sanitize_frame_list(
    frames: list,
    *,
    height: int,
    width: int,
) -> list[np.ndarray]:
    """
    Normalize every frame to HWC uint8 at (height, width).
    Preserves frame count: bad frames are replaced with the last good frame or gray.
    """
    gray = np.full((height, width, 3), 128, dtype=np.uint8)
    if not frames:
        return [gray.copy()]

    out: list[np.ndarray] = []
    last_good = gray.copy()
    for f in frames:
        try:
            rgb = _frame_to_rgb_uint8(f, target_h=height, target_w=width)
            if rgb.size == 0 or not np.isfinite(rgb).all():
                logger.warning("Non-finite frame repaired using fallback")
                rgb = last_good.copy()
            out.append(rgb)
            last_good = rgb.copy()
        except Exception as e:
            logger.warning(f"Frame sanitize repair: {e}")
            out.append(last_good.copy())

    return out


# ──────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────

@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = "blurry, distorted, low quality, watermark, text, worst quality"
    # Default 32 frames @ 8 fps → 4 seconds (Phase-2 deliverable: clip ≥ 4s).
    num_frames: int = 32
    fps: int = 8
    width: int = 256
    height: int = 256
    num_inference_steps: int = 25
    guidance_scale: float = 9.0
    seed: Optional[int] = None
    use_ddim: bool = True
    use_enhancement_a: bool = True   # Cosine schedule + temporal smoothing
    use_enhancement_b: bool = True   # CLIP reranking
    num_candidates: int = 2          # For CLIP reranking
    temporal_smoothing_lambda: float = 0.01

    # ── Bonus: Text-to-Audio (TTS) ───────────────────────────
    enable_audio: bool = False
    audio_script: Optional[str] = None   # If None → uses prompt as narration
    audio_voice: str = "en-US-AriaNeural"
    audio_rate: str = "+0%"              # e.g. "-10%", "+15%"
    audio_pitch: str = "+0Hz"            # e.g. "-50Hz", "+100Hz"
    audio_emotion: str = "neutral"       # neutral | happy | sad | excited | calm

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ──────────────────────────────────────────────
# CLIP Reranker (Enhancement B)
# ──────────────────────────────────────────────

class CLIPReranker:
    """Enhancement B: Select best candidate video by CLIP-SIM."""

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
            logger.warning(f"CLIP unavailable — reranking disabled ({e})")

    def score(
        self,
        frames: list[np.ndarray],
        prompt: str,
        *,
        target_h: int = 256,
        target_w: int = 256,
    ) -> float:
        """Mean CLIP similarity sampled at 25/50/75% of the sequence."""
        self._lazy_load()
        if self._model is None or not frames:
            return 0.0
        try:
            import clip
            sample_idx = [
                max(0, len(frames) // 4),
                len(frames) // 2,
                min(len(frames) - 1, 3 * len(frames) // 4),
            ]
            text_tokens = clip.tokenize([prompt], truncate=True).to(self.device)
            with torch.no_grad():
                txt_feat = F.normalize(self._model.encode_text(text_tokens), dim=-1)
                sims = []
                for i in sample_idx:
                    rgb = _frame_to_rgb_uint8(frames[i], target_h=target_h, target_w=target_w)
                    img = Image.fromarray(rgb, mode="RGB")
                    img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                    img_feat = F.normalize(self._model.encode_image(img_tensor), dim=-1)
                    sims.append((img_feat * txt_feat).sum().item())
            return float(np.mean(sims))
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.0

    def select_best(
        self,
        candidates: list[list[np.ndarray]],
        prompt: str,
        *,
        target_h: int = 256,
        target_w: int = 256,
    ) -> list[np.ndarray]:
        scores = [
            self.score(c, prompt, target_h=target_h, target_w=target_w) for c in candidates
        ]
        best_idx = int(np.argmax(scores)) if scores else 0
        logger.info(f"CLIP reranking scores: {[f'{s:.4f}' for s in scores]} — selected [{best_idx}]")
        return candidates[best_idx]


# ──────────────────────────────────────────────
# Cosine noise schedule (Enhancement A)
# ──────────────────────────────────────────────

def make_cosine_alphas_cumprod(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal cosine schedule:
        f(t) = cos²((t/T + s) / (1+s) · π/2)
        ᾱ_t = f(t) / f(0)
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps + s) / (1 + s)) * (np.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5, max=1.0 - 1e-5)
    return alphas_cumprod[1:].float()


# ──────────────────────────────────────────────
# Temporal smoothing loss (Enhancement A)
# ──────────────────────────────────────────────

def temporal_smoothing_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    Penalises inter-frame latent variation:
        L_smooth = Σ_f ‖ z_f − z_{f-1} ‖²_F
    latents: (B, C, F, H, W)
    """
    if latents.dim() < 5:
        return torch.tensor(0.0, device=latents.device)
    diff = latents[:, :, 1:, :, :] - latents[:, :, :-1, :, :]
    return diff.pow(2).mean()


# ──────────────────────────────────────────────
# Main Generator
# ──────────────────────────────────────────────

class VideoGenerator:
    """Full latent-diffusion text-to-video generation pipeline."""

    MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None
        self.clip_reranker = CLIPReranker(self.device)

        # Bonus: lazy TTS engine (Edge Neural TTS); imported on demand
        # so the backend still boots if `edge-tts` is missing.
        self._tts = None
        self._load_pipeline()

    # ── Pipeline loader ──────────────────────────────────────
    def _load_pipeline(self):
        logger.info(f"Loading pipeline on {self.device} ({self.dtype})...")
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self.pipe.to(self.device)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory-efficient attention enabled")
            except Exception:
                pass
            self.pipe.enable_attention_slicing()
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Pipeline load failed ({e}) — running in MOCK mode")
            self.pipe = None

    # ── TTS (bonus) ──────────────────────────────────────────
    def _get_tts(self):
        if self._tts is not None:
            return self._tts
        try:
            from utils.tts import NeuralTTS
            self._tts = NeuralTTS()
            logger.info("Neural TTS engine ready")
        except Exception as e:
            logger.warning(f"TTS engine unavailable: {e}")
            self._tts = False
        return self._tts

    # ── Scheduler helpers (Enhancement A) ────────────────────
    def _apply_cosine_schedule(self, scheduler):
        """Replace scheduler.alphas_cumprod with cosine values (Enhancement A)."""
        n = len(scheduler.alphas_cumprod)
        cosine_ac = make_cosine_alphas_cumprod(n).to(scheduler.alphas_cumprod.device)
        scheduler.alphas_cumprod = cosine_ac
        logger.debug("Cosine noise schedule applied")

    def _get_scheduler(self, use_ddim: bool, use_cosine: bool):
        if self.pipe is None:
            return None
        sched = (
            DDIMScheduler.from_config(self.pipe.scheduler.config) if use_ddim
            else DDPMScheduler.from_config(self.pipe.scheduler.config)
        )
        if use_cosine:
            self._apply_cosine_schedule(sched)
        return sched

    # ── Mock generator (graceful CPU/no-model fallback) ──────
    def _mock_generate(self, cfg: GenerationConfig) -> list[np.ndarray]:
        """Synthetic gradient frames so the API is testable without a GPU."""
        logger.warning("Using mock generator (no GPU model loaded)")
        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 42)
        h, w = cfg.height, cfg.width

        # Smooth color-shift base + slow drift, so video has some motion
        base = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
        frames = []
        for i in range(cfg.num_frames):
            t = i / max(cfg.num_frames - 1, 1)
            tint = np.array([
                int(80 * np.sin(2 * np.pi * t)),
                int(40 * np.cos(2 * np.pi * t)),
                int(60 * np.sin(np.pi * t)),
            ], dtype=int)
            noise = rng.integers(-12, 12, (h, w, 3))
            frame = np.clip(base.astype(int) + tint + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames

    # ── Main entry ───────────────────────────────────────────
    def generate(
        self,
        cfg: GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Generate video from text prompt; returns metadata dict."""
        start = time.time()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        def _cb(step, total, msg=""):
            if progress_callback:
                progress_callback(step, total, msg)

        _cb(0, cfg.num_inference_steps, "Initializing pipeline...")

        candidates: list[list[np.ndarray]] = []
        n_runs = cfg.num_candidates if (cfg.use_enhancement_b and self.pipe is not None) else 1

        for run in range(n_runs):
            _cb(0, cfg.num_inference_steps, f"Generating candidate {run + 1}/{n_runs}...")

            if self.pipe is None:
                frames = sanitize_frame_list(
                    self._mock_generate(cfg), height=cfg.height, width=cfg.width
                )
            else:
                scheduler = self._get_scheduler(cfg.use_ddim, cfg.use_enhancement_a)
                self.pipe.scheduler = scheduler

                def _temporal_smooth_latents(lat: torch.Tensor) -> None:
                    if lat.dim() >= 5 and lat.shape[2] > 1:
                        smooth_grad = lat[:, :, 1:] - lat[:, :, :-1]
                        lat[:, :, 1:] -= cfg.temporal_smoothing_lambda * smooth_grad
                    # Stop NaN/Inf from corrupting the VAE decode (callback runs every step).
                    lat.copy_(torch.nan_to_num(lat, nan=0.0, posinf=1e4, neginf=-1e4))

                def step_callback_new(pipe, step_idx, timestep, callback_kwargs):
                    _cb(
                        step_idx, cfg.num_inference_steps,
                        f"Denoising step {step_idx}/{cfg.num_inference_steps}",
                    )
                    if cfg.use_enhancement_a and "latents" in callback_kwargs:
                        lat = callback_kwargs["latents"]
                        _temporal_smooth_latents(lat)
                        callback_kwargs["latents"] = lat
                    return callback_kwargs

                def step_callback_legacy(step_idx, timestep, latents_tensor):
                    _cb(
                        step_idx, cfg.num_inference_steps,
                        f"Denoising step {step_idx}/{cfg.num_inference_steps}",
                    )
                    if cfg.use_enhancement_a:
                        _temporal_smooth_latents(latents_tensor)

                pipe_sig = inspect.signature(self.pipe.__call__)
                pipe_kwargs = dict(
                    prompt=cfg.prompt,
                    negative_prompt=cfg.negative_prompt,
                    num_frames=cfg.num_frames,
                    width=cfg.width,
                    height=cfg.height,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                )
                if "callback_on_step_end" in pipe_sig.parameters:
                    pipe_kwargs["callback_on_step_end"] = step_callback_new
                    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                else:
                    pipe_kwargs["callback"] = step_callback_legacy
                    pipe_kwargs["callback_steps"] = 1

                # torch.autocast: device_type must match {"cuda","cpu"}
                # On CPU autocast requires explicit dtype.
                if self.device == "cuda":
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)

                with autocast_ctx:
                    result = self.pipe(**pipe_kwargs)
                raw = [np.array(f) for f in result.frames[0]]
                frames = sanitize_frame_list(raw, height=cfg.height, width=cfg.width)

            candidates.append(frames)

        _cb(cfg.num_inference_steps, cfg.num_inference_steps, "Selecting best candidate...")

        # Enhancement B: CLIP reranking
        if cfg.use_enhancement_b and len(candidates) > 1:
            best_frames = self.clip_reranker.select_best(
                candidates,
                cfg.prompt,
                target_h=cfg.height,
                target_w=cfg.width,
            )
        else:
            best_frames = candidates[0]

        best_frames = sanitize_frame_list(
            best_frames, height=cfg.height, width=cfg.width
        )

        # Save video (silent track)
        _cb(cfg.num_inference_steps, cfg.num_inference_steps, "Encoding video...")
        os.makedirs("outputs", exist_ok=True)
        video_path = f"outputs/{cfg.job_id}.mp4"
        pil_frames = [Image.fromarray(f, mode="RGB") for f in best_frames]
        export_to_video(pil_frames, video_path, fps=cfg.fps)

        # ── Bonus: Text-to-Audio narration (muxed with ffmpeg) ──
        audio_path: Optional[str] = None
        final_path = video_path
        if cfg.enable_audio:
            _cb(cfg.num_inference_steps, cfg.num_inference_steps, "Synthesizing narration (TTS)...")
            tts = self._get_tts()
            if tts:
                try:
                    duration_s_est = cfg.num_frames / max(cfg.fps, 1)
                    audio_path = f"outputs/{cfg.job_id}.mp3"
                    tts.synthesize(
                        text=(cfg.audio_script or cfg.prompt),
                        output_path=audio_path,
                        voice=cfg.audio_voice,
                        rate=cfg.audio_rate,
                        pitch=cfg.audio_pitch,
                        emotion=cfg.audio_emotion,
                        target_duration=duration_s_est,
                    )
                    _cb(
                        cfg.num_inference_steps, cfg.num_inference_steps,
                        "Muxing audio + video...",
                    )
                    from utils.video_utils import mux_audio_video
                    final_path = f"outputs/{cfg.job_id}_with_audio.mp4"
                    mux_audio_video(video_path, audio_path, final_path)
                except Exception as e:
                    logger.warning(f"Audio synthesis failed: {e}")
                    audio_path = None
                    final_path = video_path
            else:
                logger.warning("Audio requested but TTS engine unavailable")

        clip_score = self.clip_reranker.score(
            best_frames,
            cfg.prompt,
            target_h=cfg.height,
            target_w=cfg.width,
        )
        duration = round(time.time() - start, 2)

        logger.info(
            f"Job {cfg.job_id} completed in {duration}s | "
            f"frames={len(best_frames)} | clip={clip_score:.4f} | "
            f"audio={'yes' if audio_path else 'no'}"
        )

        return {
            "job_id": cfg.job_id,
            "video_path": final_path,
            "video_url": f"/{final_path}",
            "audio_url": f"/{audio_path}" if audio_path else None,
            "num_frames": len(best_frames),
            "fps": cfg.fps,
            "duration_video_s": round(len(best_frames) / max(cfg.fps, 1), 2),
            "clip_score": round(clip_score, 4),
            "duration_s": duration,
            "enhancements": {
                "cosine_schedule": cfg.use_enhancement_a,
                "clip_reranking": cfg.use_enhancement_b,
                "tts_audio": bool(audio_path),
            },
        }
