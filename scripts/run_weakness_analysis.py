"""
NovaCine — Phase-1 Weakness Quantification Runner
=================================================

Runs the formal weakness experiments required by the project rubric:

  Weakness 1 — Temporal Flickering
    Metric: Flow Warping Error (FWE) + frame-difference variance.
  Weakness 2 — Prompt Misalignment Under High Motion
    Metric: Pearson correlation between CLIP-SIM and optical-flow magnitude.

Operates on a directory of .mp4 clips. If `--prompts` is given,
each clip is paired with its prompt for CLIP-SIM scoring.

Usage:
    python scripts/run_weakness_analysis.py \
        --videos backend/outputs \
        --prompts evaluation/prompts.json \
        --output evaluation/results/weakness_report.json

If no real videos are available, runs in `--demo` mode using
synthetic data so the pipeline is verifiable end-to-end without a
GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make the project's evaluation package importable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────
# Demo (no videos required)
# ──────────────────────────────────────────────────────────────

def demo_report() -> dict:
    rng = np.random.default_rng(42)
    flow_mag  = rng.uniform(0.3, 4.0, 50)
    clip_sim  = 0.32 - 0.025 * flow_mag + rng.normal(0, 0.02, 50)
    fwes      = rng.uniform(0.08, 0.16, 50)
    fdv       = float(np.var(rng.uniform(0.05, 0.25, 50)))

    pearson_r = float(np.corrcoef(clip_sim, flow_mag)[0, 1])
    return {
        "mode": "demo",
        "n_clips": 50,
        "weakness_1_temporal_flickering": {
            "FWE_mean":            float(np.mean(fwes)),
            "FWE_std":             float(np.std(fwes)),
            "frame_diff_variance": fdv,
            "interpretation":
                "FWE > 0.08 indicates noticeable inter-frame inconsistency.",
        },
        "weakness_2_prompt_misalignment": {
            "CLIP_SIM_static_mean":  float(np.mean(clip_sim[flow_mag < 1.0])),
            "CLIP_SIM_dynamic_mean": float(np.mean(clip_sim[flow_mag > 2.0])),
            "pearson_r_clipsim_vs_motion": pearson_r,
            "interpretation":
                "Negative correlation confirms prompt fidelity drops as motion magnitude increases.",
        },
    }


# ──────────────────────────────────────────────────────────────
# Real-videos path
# ──────────────────────────────────────────────────────────────

def real_report(videos_dir: str, prompts_file: str | None) -> dict:
    from evaluation.run_metrics import (
        load_video_frames,
        compute_flow_warping_error,
        compute_clip_sim,
    )
    from backend.utils.video_utils import (
        compute_frame_difference_variance,
        compute_optical_flow_magnitude,
    )

    prompts: dict[str, str] = {}
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file) as f:
            prompts = json.load(f)

    videos = sorted(Path(videos_dir).glob("*.mp4"))
    if not videos:
        print(f"No videos in {videos_dir}; falling back to --demo mode.")
        return demo_report()

    fwes, fdvs, sims, flows = [], [], [], []
    for vp in videos[:50]:
        frames = load_video_frames(str(vp))
        if not frames or len(frames) < 2:
            continue
        fwes.append(compute_flow_warping_error(frames))
        fdvs.append(compute_frame_difference_variance(frames))
        flow = compute_optical_flow_magnitude(frames)
        flows.append(float(np.mean(flow)) if flow else 0.0)
        prompt = prompts.get(vp.name, "a generated video clip")
        sims.append(compute_clip_sim(frames, prompt))

    flows = np.array(flows)
    sims  = np.array(sims)
    pearson_r = float(np.corrcoef(sims, flows)[0, 1]) if len(sims) > 1 else 0.0

    static_mask  = flows < np.median(flows)
    dynamic_mask = ~static_mask

    return {
        "mode": "real",
        "n_clips": len(fwes),
        "weakness_1_temporal_flickering": {
            "FWE_mean":            float(np.mean(fwes)),
            "FWE_std":             float(np.std(fwes)),
            "frame_diff_variance": float(np.mean(fdvs)),
        },
        "weakness_2_prompt_misalignment": {
            "CLIP_SIM_static_mean":  float(np.mean(sims[static_mask]))  if static_mask.any()  else None,
            "CLIP_SIM_dynamic_mean": float(np.mean(sims[dynamic_mask])) if dynamic_mask.any() else None,
            "pearson_r_clipsim_vs_motion": pearson_r,
        },
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos",  default="backend/outputs")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--output",  default="evaluation/results/weakness_report.json")
    ap.add_argument("--demo",    action="store_true",
                    help="Skip real videos and emit demo data.")
    args = ap.parse_args()

    report = demo_report() if args.demo else real_report(args.videos, args.prompts)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 64)
    print("WEAKNESS QUANTIFICATION REPORT")
    print("=" * 64)
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
