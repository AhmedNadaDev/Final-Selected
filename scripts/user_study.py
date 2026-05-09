"""
NovaCine — User Study Logger
Collects and aggregates human evaluation scores for generated videos.
Stores results in evaluation/results/user_study.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


USER_STUDY_PATH = "evaluation/results/user_study.json"
CRITERIA = {
    "temporal_coherence": "How smooth and consistent is the motion? (1=very flickery, 5=perfectly smooth)",
    "prompt_alignment":   "How well does the video match the text prompt? (1=no match, 5=perfect match)",
    "visual_quality":     "How high is the visual/image quality? (1=very poor, 5=very high)",
    "motion_naturalness": "How natural does the motion look? (1=robotic/unnatural, 5=very natural)",
    "overall":            "Overall quality score? (1=poor, 5=excellent)",
}


def load_study() -> dict:
    os.makedirs(os.path.dirname(USER_STUDY_PATH), exist_ok=True)
    if os.path.exists(USER_STUDY_PATH):
        with open(USER_STUDY_PATH) as f:
            return json.load(f)
    return {"entries": [], "summary": {}}


def save_study(data: dict):
    with open(USER_STUDY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_score(criterion: str, description: str) -> int:
    while True:
        try:
            val = input(f"  {description}\n  → Score [1-5]: ").strip()
            score = int(val)
            if 1 <= score <= 5:
                return score
            print("  Please enter a number between 1 and 5.")
        except (ValueError, KeyboardInterrupt):
            print("\n  Skipping.")
            return 3


def log_entry():
    print("\n" + "=" * 60)
    print("NovaCine — User Study Entry")
    print("=" * 60)

    video_id = input("Video job ID (or filename): ").strip()
    prompt   = input("Prompt used: ").strip()
    condition = input("Condition [baseline/enh_a/enh_b/combined]: ").strip()

    print("\nPlease rate the following aspects:")
    scores = {}
    for key, desc in CRITERIA.items():
        print()
        scores[key] = get_score(key, desc)

    comments = input("\nAny comments? (optional): ").strip()

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "video_id": video_id,
        "prompt": prompt,
        "condition": condition,
        "scores": scores,
        "comments": comments,
        "mean_score": round(sum(scores.values()) / len(scores), 2),
    }

    data = load_study()
    data["entries"].append(entry)

    # Recompute summary
    cond_scores: dict[str, list] = {}
    for e in data["entries"]:
        c = e["condition"]
        if c not in cond_scores:
            cond_scores[c] = []
        cond_scores[c].append(e["mean_score"])

    data["summary"] = {
        c: {"n": len(v), "mean": round(sum(v) / len(v), 3), "std": round(float(__import__("numpy").std(v)), 3)}
        for c, v in cond_scores.items()
    }

    save_study(data)
    print(f"\n✓ Entry logged. Mean score: {entry['mean_score']}/5.0")
    return entry


def print_summary():
    data = load_study()
    if not data["entries"]:
        print("No entries yet.")
        return

    print("\n" + "=" * 60)
    print("USER STUDY SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(data['entries'])}\n")

    for cond, stats in data["summary"].items():
        print(f"  {cond:<15}  n={stats['n']:>3}  mean={stats['mean']:.3f}/5  std=±{stats['std']:.3f}")

    print("\nPer-criterion breakdown:")
    for criterion in CRITERIA:
        vals = [e["scores"].get(criterion, 3) for e in data["entries"]]
        mean = sum(vals) / len(vals)
        print(f"  {criterion:<22}  mean={mean:.2f}/5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    else:
        while True:
            log_entry()
            again = input("\nLog another entry? [y/N]: ").strip().lower()
            if again != "y":
                break
        print_summary()
