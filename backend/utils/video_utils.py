"""
NovaCine — Video Utilities
Frame analysis, optical flow, video stats, audio muxing
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


def load_frames(video_path: str) -> list[np.ndarray]:
    """Load video as list of RGB uint8 frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def save_frames_as_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 8,
) -> str:
    """Save list of RGB frames as MP4 using OpenCV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def compute_frame_difference_variance(frames: list[np.ndarray]) -> float:
    """
    Measure temporal flickering via frame-difference variance.
    Higher value = more flickering.
    Var(|I_t+1 - I_t|) averaged over all adjacent pairs.
    """
    diffs = []
    for i in range(len(frames) - 1):
        d = np.abs(frames[i + 1].astype(float) - frames[i].astype(float))
        diffs.append(d.mean())
    if not diffs:
        return 0.0
    return float(np.var(diffs))


def compute_optical_flow_magnitude(frames: list[np.ndarray]) -> list[float]:
    """
    Compute average optical flow magnitude between consecutive frames.
    Returns per-transition flow magnitudes.
    """
    magnitudes = []
    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(float(mag.mean()))
    return magnitudes


def get_video_stats(video_path: str) -> dict:
    """Return metadata dict for a video file."""
    cap = cv2.VideoCapture(video_path)
    stats = {
        "path": video_path,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "size_mb": round(os.path.getsize(video_path) / 1e6, 2) if os.path.exists(video_path) else 0,
    }
    stats["duration_s"] = round(stats["frames"] / max(stats["fps"], 1), 2)
    cap.release()
    return stats


def _ffmpeg_path() -> str:
    """Locate an ffmpeg binary (system path or imageio-ffmpeg fallback)."""
    local = shutil.which("ffmpeg")
    if local:
        return local
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Mux a silent video and an external audio track into one MP4.
    If ffmpeg is unavailable the original silent video is returned unchanged.
    """
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-loglevel", "error",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Muxed audio+video → {output_path}")
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"ffmpeg muxing failed ({e}) — returning silent video")
        if os.path.abspath(video_path) != os.path.abspath(output_path):
            try:
                shutil.copy2(video_path, output_path)
                return output_path
            except OSError:
                pass
        return video_path


def apply_temporal_median_filter(frames: list[np.ndarray], window: int = 3) -> list[np.ndarray]:
    """
    Post-processing: median filter across temporal window to reduce flicker.
    For each frame t, replace with median of [t-k, ..., t, ..., t+k].
    """
    n = len(frames)
    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_frames = np.stack(frames[lo:hi], axis=0).astype(float)
        result.append(np.median(window_frames, axis=0).astype(np.uint8))
    return result
