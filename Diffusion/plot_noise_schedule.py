"""
Diagnostic plot of the cosine noise schedule actually used by the live diffusion model
(threeDSparseTransDiffusion.cosine_beta_schedule): per-step noise variance, cumulative signal
retention, and the resulting signal/noise mixing coefficients in x_k = sqrt(alpha_bar_k) x0 +
sqrt(1 - alpha_bar_k) eps.

Notation: k is the diffusion timestep, matching this codebase's convention (see
plot_forward_noise_trajectory.py) that k indexes the diffusion process and is kept distinct from t,
which is reserved for the AIPP replanning/mission timestep.

The schedule formula below is copied verbatim from threeDSparseTransDiffusion.cosine_beta_schedule
(not re-derived) so this plot exactly matches what the live model trains/samples with, without
importing that module's heavier dependencies (torch model classes, mmap'd training dataset).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "plots"

T = 20
S = 0.008
BETA_MIN, BETA_MAX = 1e-4, 0.999


def cosine_beta_schedule(timesteps, s=S):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, BETA_MIN, BETA_MAX)


def main():
    betas = cosine_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    k = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(
        rf"Cosine noise schedule ($T={T}$, $s={S}$, "
        rf"$\beta_k \in [10^{{-4}}, {BETA_MAX}]$)",
        fontsize=13, y=1.03,
    )

    color = "#c0392b"
    marker_kwargs = dict(marker="o", color=color, markersize=5, linewidth=1.8)

    ax = axes[0]
    ax.plot(k, betas, **marker_kwargs)
    ax.set_title(r"(a) Per-step noise variance $\beta_k$")
    ax.set_xlabel("diffusion timestep $k$")
    ax.set_ylabel(r"$\beta_k$")
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1]
    ax.plot(k, alphas_cumprod, **marker_kwargs)
    ax.set_yscale("log")
    ax.set_title(r"(b) Cumulative signal retention $\bar\alpha_k$")
    ax.set_xlabel("diffusion timestep $k$")
    ax.set_ylabel(r"$\bar\alpha_k$ (log scale)")
    ax.grid(alpha=0.3, linestyle="--", which="both")
    final_val = alphas_cumprod[-1]
    ax.annotate(
        rf"$\bar\alpha_T \approx${final_val:.1e}",
        xy=(k[-1], final_val), xytext=(k[-1] - 9, final_val * 25),
        color=color, fontsize=10,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )

    ax = axes[2]
    sqrt_ab = np.sqrt(alphas_cumprod)
    sqrt_1mab = np.sqrt(1.0 - alphas_cumprod)
    ax.fill_between(k, 0, 1, where=sqrt_ab >= sqrt_1mab, color="#3465a4", alpha=0.08, zorder=0)
    ax.fill_between(k, 0, 1, where=sqrt_ab < sqrt_1mab, color="#c0392b", alpha=0.08, zorder=0)
    ax.plot(k, sqrt_ab, marker="o", color="#3465a4", markersize=5, linewidth=1.8, label=r"signal $\sqrt{\bar\alpha_k}$")
    ax.plot(k, sqrt_1mab, marker="o", color=color, markersize=5, linewidth=1.8, label=r"noise $\sqrt{1-\bar\alpha_k}$")
    ax.set_title(r"(c) $x_k=\sqrt{\bar\alpha_k}\,x_0+\sqrt{1-\bar\alpha_k}\,\epsilon$")
    ax.set_xlabel("diffusion timestep $k$")
    ax.set_ylabel("mixing coefficient")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"noise_schedule_cosine_T{T}.{ext}"
        fig.savefig(out_path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
