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
from core.generator import VideoGenerator, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description="NovaCine — Standalone Video Generation")
    parser.add_argument("prompt", help="Text prompt for video generation")
    parser.add_argument("--output",     default="output.mp4")
    parser.add_argument("--frames",     type=int,   default=16)
    parser.add_argument("--fps",        type=int,   default=8)
    parser.add_argument("--steps",      type=int,   default=25)
    parser.add_argument("--cfg",        type=float, default=9.0)
    parser.add_argument("--width",      type=int,   default=256)
    parser.add_argument("--height",     type=int,   default=256)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--no-ddim",    action="store_true")
    parser.add_argument("--no-enh-a",   action="store_true")
    parser.add_argument("--no-enh-b",   action="store_true")
    parser.add_argument("--candidates", type=int, default=2)
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
        use_ddim=not args.no_ddim,
        use_enhancement_a=not args.no_enh_a,
        use_enhancement_b=not args.no_enh_b,
        num_candidates=args.candidates,
    )

    def progress(step, total, msg):
        pct = int(step / max(total, 1) * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}%  {msg[:50]:<50}", end="", flush=True)

    print(f"\n🎬 NovaCine — Generating video for: \"{args.prompt}\"\n")
    gen = VideoGenerator()
    result = gen.generate(cfg, progress_callback=progress)
    print(f"\n\n✅ Done!")
    print(f"   Video:     {result['video_path']}")
    print(f"   Frames:    {result['num_frames']}")
    print(f"   CLIP-SIM:  {result['clip_score']}")
    print(f"   Time:      {result['duration_s']}s")


if __name__ == "__main__":
    main()
