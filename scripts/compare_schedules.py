"""
NovaCine — Noise Schedule Comparison
Plots linear vs cosine schedule: beta, alpha_cumprod, SNR curves.
Run: python scripts/compare_schedules.py
"""
from __future__ import annotations
import math
import sys
import numpy as np

# Make stdout UTF-8 friendly on Windows consoles (cp1252 chokes on β/ᾱ/✓).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T = 1000

# ── Linear schedule ──────────────────────────────────────────────────
beta_start, beta_end = 1e-4, 0.02
betas_linear = np.linspace(beta_start, beta_end, T)
alphas_linear = 1 - betas_linear
ac_linear = np.cumprod(alphas_linear)

# ── Cosine schedule ──────────────────────────────────────────────────
s = 0.008
steps = np.arange(T + 1)
f = np.cos(((steps / T + s) / (1 + s)) * (math.pi / 2)) ** 2
ac_cosine = f / f[0]
ac_cosine = np.clip(ac_cosine[1:], 1e-5, 1 - 1e-5)
betas_cosine = 1 - (ac_cosine / np.concatenate([[1.0], ac_cosine[:-1]]))
betas_cosine = np.clip(betas_cosine, 1e-5, 0.999)

# ── SNR (signal-to-noise ratio) ──────────────────────────────────────
snr_linear = ac_linear / (1 - ac_linear)
snr_cosine = ac_cosine / (1 - ac_cosine)

t = np.arange(T)

print("=" * 60)
print("NOISE SCHEDULE COMPARISON")
print("=" * 60)
print(f"\n{'Timestep':>10} {'β_lin':>10} {'β_cos':>10} {'ᾱ_lin':>10} {'ᾱ_cos':>10}")
print("-" * 55)
for ti in [0, 100, 200, 400, 500, 700, 800, 900, 999]:
    print(f"{ti:>10d} {betas_linear[ti]:>10.5f} {betas_cosine[ti]:>10.5f} "
          f"{ac_linear[ti]:>10.5f} {ac_cosine[ti]:>10.5f}")

print("\nKey observations:")
print(f"  Linear: ᾱ_0={ac_linear[0]:.4f}, ᾱ_500={ac_linear[500]:.4f}, ᾱ_999={ac_linear[999]:.6f}")
print(f"  Cosine: ᾱ_0={ac_cosine[0]:.4f}, ᾱ_500={ac_cosine[500]:.4f}, ᾱ_999={ac_cosine[999]:.6f}")
print(f"\n  Linear SNR at t=0: {snr_linear[0]:.2f}  (very high → nearly clean)")
print(f"  Cosine SNR at t=0: {snr_cosine[0]:.2f}  (more moderate)")
print(f"\n  → Cosine avoids extreme SNR at boundaries,")
print(f"    providing more uniform destruction across timesteps.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#05050a")

    plots = [
        (axes[0], t, betas_linear,   betas_cosine,  "β_t (Noise Level)",        "β"),
        (axes[1], t, ac_linear,      ac_cosine,     "ᾱ_t (Signal Preservation)", "ᾱ"),
        (axes[2], t, snr_linear,     snr_cosine,    "SNR(t) = ᾱ_t / (1−ᾱ_t)",   "SNR"),
    ]

    for ax, xs, y_lin, y_cos, title, ylabel in plots:
        ax.plot(xs, y_lin, color="#ef4444", linewidth=1.5, label="Linear", alpha=0.85)
        ax.plot(xs, y_cos, color="#a78bfa", linewidth=1.5, label="Cosine", alpha=0.85)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Timestep t", color="#6b6b8a", fontsize=9)
        ax.set_ylabel(ylabel, color="#6b6b8a", fontsize=9)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#6b6b8a", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#252538")
        ax.legend(fontsize=8, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#252538")
        if ylabel == "SNR":
            ax.set_ylim(0, 20)

    plt.tight_layout()
    out = "evaluation/results/noise_schedules.png"
    import os; os.makedirs("evaluation/results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#05050a")
    plt.close()
    print(f"\n✓ Schedule comparison chart saved to {out}")
except ImportError:
    print("\n(matplotlib not available — skipping chart)")
