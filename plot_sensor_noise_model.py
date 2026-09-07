"""
Sensor noise model (gaussianprocesstraining.noise_model), reused directly, not
reimplemented: measurement noise added to a sensor reading as a function of
flight altitude. Plotted as standard deviation sigma = sqrt(variance), since
that's the actual scale of the value sampled and added to each measurement
(see sample_correlated_sensor_noise's rng.normal(0.0, sqrt(variance)) call).

This model is a fixed absolute noise curve, independent of map type -
deliberately NOT rescaled for NAIP's narrower value range, a decision the user
explicitly declined to change. See CARRYOVER_NAIP_SWITCH_AND_BASELINES.md
sec 1.5 for the full tradeoff (realistic sensor noise vs. planner-comparison
confounding) left open on purpose.

x-axis is clipped to [ZMIN, ZMAX] (the actual flown altitude band, matching
CMAES_classic_singlemap.py's current module-level constants 10.0/40.0) rather
than extrapolating past it - this plot is meant to sit in an IEEE-style paper
at small print size (see plot_fov_resolution_altitude.py's matching FIG_WIDTH,
combine_noise_fov_resolution.py stacks the two), so font sizes here are large
and annotation text is kept to the minimum needed to stay legible once shrunk.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussianprocesstraining import noise_model

SCRIPT_DIR = Path(__file__).resolve().parent
ZMIN, ZMAX = 10.0, 40.0
OUT_PATH = SCRIPT_DIR / "sensor_noise_model.png"
FIG_WIDTH = 7.5  # inches; matches plot_fov_resolution_altitude.py's FIG_WIDTH for clean vertical stacking

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
COLOR_STD = "#2a78d6"  # categorical slot 1 (blue)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def main():
    altitudes = np.linspace(ZMIN, ZMAX, 400)
    std = np.sqrt(noise_model(altitudes))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3))
    ax.plot(altitudes, std, color=COLOR_STD, linewidth=3.0, zorder=3)

    label_offsets = {ZMIN: (10, -4), ZMAX: (-45, 10)}
    for z in (ZMIN, ZMAX):
        v = float(np.sqrt(noise_model(z)))
        ax.scatter([z], [v], color=COLOR_STD, s=70, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(
            f"{v:.2f}",
            (z, v), textcoords="offset points", xytext=label_offsets[z],
            color=INK_SECONDARY, fontsize=15, fontweight="bold",
        )

    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel("Altitude (m)", color=INK_SECONDARY, fontsize=15)
    ax.set_ylabel("Noise σ", color=INK_SECONDARY, fontsize=15)
    ax.set_title("Sensor noise vs. altitude", color=INK_PRIMARY, fontsize=17)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=13)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
