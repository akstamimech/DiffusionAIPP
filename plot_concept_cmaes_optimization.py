"""
Conceptual (non-data) diagram of the CMA-ES optimization step: a population
of candidate trajectories sampled and evaluated each generation, converging
toward one best trajectory, repeated across many generations. Pure
illustration for a preliminary/motivation slide, not tied to any real run's
data - this is why path planning takes a long optimization time (the lattice
diagram's single lattice pass vs. this repeated population search).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from scipy.interpolate import CubicSpline
import scienceplots

plt.style.use(["science", "no-latex"])

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "concept_cmaes_optimization.png"

rng = np.random.default_rng(3)

ACCENT = "#d1495b"
CANDIDATE_COLOR = "#8a8f98"
TARGET_COLOR = "#f2a541"
START_COLOR = "#22333b"
LABEL_COLOR = "#3d4650"
LOOP_COLOR = "#5c6570"

fig, ax = plt.subplots(figsize=(7, 5.2))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

start = np.array([0.0, 0.0])
target = np.array([7.0, 1.2])

for r, a in [(1.05, 0.10), (0.75, 0.16), (0.45, 0.24)]:
    ax.add_patch(Circle(target, r, color=TARGET_COLOR, alpha=a, lw=0, zorder=1))

N_CANDIDATES = 28
for _ in range(N_CANDIDATES):
    end = target + rng.normal(0, 0.5, size=2)
    mid = np.array([end[0] * 0.5, start[1] + (end[1] - start[1]) * 0.5 + rng.normal(0, 0.45)])
    ctrl_x = np.array([start[0], mid[0], end[0]])
    ctrl_y = np.array([start[1], mid[1], end[1]])
    spline = CubicSpline(ctrl_x, ctrl_y)
    dense_x = np.linspace(ctrl_x[0], ctrl_x[-1], 60)
    ax.plot(dense_x, spline(dense_x), color=CANDIDATE_COLOR, linewidth=0.9, alpha=0.35, zorder=2)

best_ctrl_x = np.array([start[0], 2.3, 4.8, target[0]])
best_ctrl_y = np.array([start[1], 0.9, 1.35, target[1]])
best_spline = CubicSpline(best_ctrl_x, best_ctrl_y)
dense_x = np.linspace(best_ctrl_x[0], best_ctrl_x[-1], 200)
ax.plot(dense_x, best_spline(dense_x), color=ACCENT, linewidth=2.8, zorder=4, solid_capstyle="round")

ax.scatter(*start, s=170, facecolor=START_COLOR, edgecolor="white", linewidth=1.6, zorder=6)

loop_center = (1.1, 3.15)
loop_radius = 0.55
arc = Arc(loop_center, 2 * loop_radius, 2 * loop_radius, angle=0,
          theta1=25, theta2=320, color=LOOP_COLOR, linewidth=2.0, zorder=5)
ax.add_patch(arc)
tip_angle = np.deg2rad(320)
tip = (loop_center[0] + loop_radius * np.cos(tip_angle), loop_center[1] + loop_radius * np.sin(tip_angle))
tail_angle = np.deg2rad(300)
tail = (loop_center[0] + loop_radius * np.cos(tail_angle), loop_center[1] + loop_radius * np.sin(tail_angle))
ax.annotate(
    "", xy=tip, xytext=tail,
    arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color=LOOP_COLOR, linewidth=2.0,
                     shrinkA=0, shrinkB=0),
    zorder=5,
)
ax.text(loop_center[0], loop_center[1] - loop_radius - 0.28, "repeat many\ngenerations",
        ha="center", va="top", fontsize=10, color=LABEL_COLOR)

ax.text(start[0], start[1] - 0.55, "current position", ha="center", va="top", fontsize=10, color=LABEL_COLOR)
ax.text(target[0], target[1] + 1.35, "important region", ha="center", va="bottom", fontsize=10, color=LABEL_COLOR)
ax.annotate(
    "best trajectory\nthis generation",
    xy=(dense_x[150], best_spline(dense_x[150])),
    xytext=(3.6, -1.9),
    fontsize=10, color=ACCENT, ha="center",
    arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.0, alpha=0.6),
)
ax.text(4.5, 3.35, "population of sampled\ncandidate trajectories",
        ha="center", va="bottom", fontsize=10, color=LABEL_COLOR)

ax.set_xlim(-1.0, target[0] + 1.6)
ax.set_ylim(-2.4, 4.1)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, transparent=True)
plt.close(fig)
print(f"Wrote {OUT_PNG}")
