"""
NovaCine AI Model — Inference entry point
Wraps the full pipeline for standalone usage outside FastAPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Allow running directly: python ai-model/pipeline/infer.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import argparse
from config import GenerationConfig
from generator import VideoGenerator


def main():
    parser = argparse.ArgumentParser(description="NovaCine — Standalone Video Generation")
    parser.add_argument("prompt", help="Text prompt for video generation")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = GenerationConfig(
        prompt=args.prompt,
        num_frames=args.frames,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    def progress(step, total, msg):
        pct = int(step / max(total, 1) * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}%  {msg[:50]:<50}", end="", flush=True)

    print(f"\n🎬 NovaCine — Generating video for: \"{args.prompt}\"\n")
    gen = VideoGenerator()
    result = gen.generate(cfg, progress_callback=progress)
    print(f"\n\n✅ Done!")
    print(f"   Video:        {result['video_path']}")
    print(f"   Frames:       {result['num_frames']}")
    print(f"   Motion score: {result.get('motion_score')}")
    print(f"   Time:         {result['duration_s']}s")


if __name__ == "__main__":
    main()
