"""
NovaCine — ZeroScope V2 text-to-video generator (Diffusers).

Baseline: DiffusionPipeline + DPMSolverMultistepScheduler, fp16 on CUDA.
Optional enhancements are gated in config.py and implemented in enhancements.py.
"""
from __future__ import annotations

import inspect
import os
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from loguru import logger
from PIL import Image

import config as cfg_mod
import enhancements


def extract_frame_list(result) -> list:
    frames = result.frames
    if frames is None:
        raise RuntimeError("Pipeline returned no frames")
    first = frames[0]
    if isinstance(first, (list, tuple)):
        return list(first)
    return list(frames)


def frames_to_pil_rgb(frames: list) -> list[Image.Image]:
    out: list[Image.Image] = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(f.convert("RGB"))
            continue
        arr = np.asarray(f)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] >= 3:
            arr = arr[..., :3]
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= 1.01 and lo >= -1.01 and lo < 0:
                arr = (arr + 1.0) * 0.5
            if hi <= 1.01:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = np.round(arr).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def temporal_mean_abs_diff(pil_frames: list[Image.Image]) -> float:
    if len(pil_frames) < 2:
        return 0.0
    diffs = []
    for a, b in zip(pil_frames[:-1], pil_frames[1:]):
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        diffs.append(np.mean(np.abs(aa - bb)))
    return float(np.mean(diffs))


class VideoGenerator:
    MODEL_ID = cfg_mod.MODEL_ID

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[DiffusionPipeline] = None
        self._load_time_s: float = 0.0
        self._tts = None
        self._clip_reranker: Optional[enhancements.CLIPReranker] = None
        self._load_pipeline()

    def _log_cuda_mem(self, tag: str):
        if self.device != "cuda":
            return
        alloc = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        logger.info(f"[GPU] {tag} — allocated: {alloc:.1f} MB | peak: {peak:.1f} MB")

    def _load_pipeline(self):
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading {self.MODEL_ID} on {self.device} ({dtype})...")
        t0 = time.perf_counter()
        try:
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            self.pipe = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
            )
            self.pipe.scheduler = enhancements.build_scheduler(self.pipe)
            self.pipe.to(self.device)

            try:
                self.pipe.enable_attention_slicing()
                logger.info("attention_slicing enabled")
            except Exception as e:
                logger.warning(f"attention_slicing skipped: {e}")

            self._load_time_s = time.perf_counter() - t0
            if self.device == "cuda":
                self._log_cuda_mem("after load")

            logger.info(
                f"Pipeline ready — {self.pipe.__class__.__name__} | "
                f"scheduler: {self.pipe.scheduler.__class__.__name__} | "
                f"load_time: {self._load_time_s:.2f}s"
            )
        except Exception as e:
            logger.warning(f"Pipeline load failed ({e}) — MOCK mode")
            self.pipe = None
            self._load_time_s = time.perf_counter() - t0

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

    def _get_clip_reranker(self) -> Optional[enhancements.CLIPReranker]:
        if not cfg_mod.ENABLE_CLIP_RERANK:
            return None
        if self._clip_reranker is None:
            self._clip_reranker = enhancements.CLIPReranker(self.device)
        return self._clip_reranker

    def _mock_generate(self, cfg: cfg_mod.GenerationConfig) -> list[Image.Image]:
        logger.warning("Using mock generator (no GPU / model not loaded)")
        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 42)
        h, w = cfg.height, cfg.width
        base = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
        frames: list[Image.Image] = []
        for i in range(cfg.num_frames):
            t = i / max(cfg.num_frames - 1, 1)
            tint = np.array(
                [
                    int(80 * np.sin(2 * np.pi * t)),
                    int(40 * np.cos(2 * np.pi * t)),
                    int(60 * np.sin(np.pi * t)),
                ],
                dtype=int,
            )
            noise = rng.integers(-12, 12, (h, w, 3))
            raw = np.clip(base.astype(int) + tint + noise, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(raw, mode="RGB"))
        return frames

    def _run_inference(
        self,
        cfg: cfg_mod.GenerationConfig,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]],
        seed_override: Optional[int] = None,
    ) -> list[Image.Image]:
        assert self.pipe is not None

        # Scheduler (re)init once per forward — cosine applied only inside build_scheduler, not each step
        self.pipe.scheduler = enhancements.build_scheduler(self.pipe)

        def _cb(step: int, total: int, msg: str = ""):
            if progress_callback:
                progress_callback(step, total, msg)

        seed_val = seed_override if seed_override is not None else cfg.seed
        if seed_val is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed_val))
        else:
            gen = torch.Generator(device=self.device).manual_seed(
                random.randint(0, 2**31 - 2)
            )

        call_kw: dict = dict(
            prompt=cfg.prompt,
            num_frames=cfg.num_frames,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=gen,
        )

        pipe_sig = inspect.signature(self.pipe.__call__)
        use_temporal = cfg_mod.ENABLE_TEMPORAL_SMOOTHING

        def on_step_new(pipe, step_idx, timestep, callback_kwargs):  # noqa: ARG001
            _cb(
                step_idx,
                cfg.num_inference_steps,
                f"Denoising step {step_idx}/{cfg.num_inference_steps}",
            )
            if use_temporal and "latents" in callback_kwargs:
                lat = callback_kwargs["latents"]
                callback_kwargs["latents"] = enhancements.blend_temporal_latents(lat)
            return callback_kwargs

        def on_step_legacy(step_idx, timestep, latents_tensor):  # noqa: ARG001
            _cb(
                step_idx,
                cfg.num_inference_steps,
                f"Denoising step {step_idx}/{cfg.num_inference_steps}",
            )
            if use_temporal:
                blended = enhancements.blend_temporal_latents(latents_tensor)
                latents_tensor.copy_(blended)

        if "callback_on_step_end" in pipe_sig.parameters:
            call_kw["callback_on_step_end"] = on_step_new
            call_kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        else:
            call_kw["callback"] = on_step_legacy
            call_kw["callback_steps"] = 1

        if self.device == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = torch.autocast(
                device_type="cpu", dtype=torch.bfloat16, enabled=False
            )

        with autocast_ctx:
            try:
                out = self.pipe(negative_prompt=cfg.negative_prompt, **call_kw)
            except TypeError:
                out = self.pipe(**call_kw)

        raw_frames = extract_frame_list(out)
        return frames_to_pil_rgb(raw_frames)

    def generate(
        self,
        cfg: cfg_mod.GenerationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        job_start = time.perf_counter()

        def _cb(step: int, total: int, msg: str = ""):
            if progress_callback:
                progress_callback(step, total, msg)

        _cb(0, cfg.num_inference_steps, "Initializing...")

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed % (2**32))

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        clip_score: Optional[float] = None

        if self.pipe is None:
            pil_frames = self._mock_generate(cfg)
        elif cfg_mod.ENABLE_CLIP_RERANK:
            n = enhancements.clip_num_candidates()
            reranker = self._get_clip_reranker()
            base_seed = (
                cfg.seed
                if cfg.seed is not None
                else random.randint(0, 2**31 - 2)
            )
            candidates: list[list[Image.Image]] = []
            for run in range(n):
                _cb(
                    0,
                    cfg.num_inference_steps,
                    f"Generating candidate {run + 1}/{n}...",
                )
                seed_run = base_seed + run
                candidates.append(
                    self._run_inference(
                        cfg,
                        progress_callback=progress_callback,
                        seed_override=seed_run,
                    )
                )
            if reranker is not None and len(candidates) > 1:
                pil_frames, scores = reranker.select_best(candidates, cfg.prompt)
                clip_score = float(max(scores)) if scores else None
            else:
                pil_frames = candidates[0]
        else:
            pil_frames = self._run_inference(
                cfg,
                progress_callback=progress_callback,
                seed_override=cfg.seed,
            )

        motion_score = temporal_mean_abs_diff(pil_frames)
        logger.info(f"[GEN] motion diagnostic (mean |Δframe|): {motion_score:.4f}")
        if motion_score < 0.5 and len(pil_frames) > 1:
            logger.warning(
                "[GEN] Low inter-frame delta — verify output if video looks static"
            )

        _cb(cfg.num_inference_steps, cfg.num_inference_steps, "Encoding video...")
        os.makedirs("outputs", exist_ok=True)
        video_path = f"outputs/{cfg.job_id}.mp4"
        export_to_video(pil_frames, video_path, fps=cfg.fps)

        audio_path: Optional[str] = None
        final_path = video_path
        if cfg.enable_audio:
            _cb(
                cfg.num_inference_steps,
                cfg.num_inference_steps,
                "Synthesizing narration (TTS)...",
            )
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
                        cfg.num_inference_steps,
                        cfg.num_inference_steps,
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

        gen_elapsed = time.perf_counter() - job_start
        peak_mb = None
        if self.device == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            self._log_cuda_mem("after generation")

        logger.info(
            f"Job {cfg.job_id} — generation: {gen_elapsed:.2f}s | "
            f"frames={len(pil_frames)} | motion={motion_score:.4f} | "
            f"audio={'yes' if audio_path else 'no'}"
        )

        return {
            "job_id": cfg.job_id,
            "model_id": cfg_mod.MODEL_ID,
            "video_path": final_path,
            "video_url": f"/{final_path}",
            "audio_url": f"/{audio_path}" if audio_path else None,
            "num_frames": len(pil_frames),
            "fps": cfg.fps,
            "duration_video_s": round(len(pil_frames) / max(cfg.fps, 1), 2),
            "motion_score": round(motion_score, 4),
            "clip_score": round(clip_score, 4) if clip_score is not None else None,
            "duration_s": round(gen_elapsed, 2),
            "timing": {
                "pipeline_load_s": round(self._load_time_s, 2),
                "generation_s": round(gen_elapsed, 2),
            },
            "memory_peak_mb": round(peak_mb, 2) if peak_mb is not None else None,
            "tts_audio": bool(audio_path),
            "enhancements": {
                "cosine_schedule": cfg_mod.ENABLE_COSINE_SCHEDULE,
                "temporal_smoothing": cfg_mod.ENABLE_TEMPORAL_SMOOTHING,
                "clip_reranking": cfg_mod.ENABLE_CLIP_RERANK,
            },
        }


GenerationConfig = cfg_mod.GenerationConfig
