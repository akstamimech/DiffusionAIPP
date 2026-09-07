"""
Conceptual (non-data) diagram of the lattice grid-search planning step:
current position -> a discrete lattice of candidate waypoints evaluated at
each step of a short planning horizon -> one path picked out of the lattice.
Pure illustration for a preliminary/motivation slide, not tied to any real
run's data.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scienceplots

plt.style.use(["science", "no-latex"])

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "concept_lattice_grid_search.png"

rng = np.random.default_rng(7)

N_STEPS = 5
NODES_PER_STEP = 5

ACCENT = "#d1495b"
CANDIDATE_EDGE = "#9aa0a6"
NODE_FACE = "#e9ecef"
NODE_EDGE = "#5c6570"
TARGET_COLOR = "#f2a541"
START_COLOR = "#22333b"
LABEL_COLOR = "#3d4650"

fig, ax = plt.subplots(figsize=(7, 5.2))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

start = np.array([0.0, 0.0])

xs = np.linspace(0.0, N_STEPS, N_STEPS + 1)
center_y = np.linspace(0.0, 2.4, N_STEPS + 1)
spread = np.linspace(0.35, 1.35, N_STEPS + 1)

target_xy = (xs[-1] + 0.9, center_y[-1] + 0.5)
for r, a in [(1.05, 0.10), (0.75, 0.16), (0.45, 0.24)]:
    ax.add_patch(Circle(target_xy, r, color=TARGET_COLOR, alpha=a, lw=0, zorder=1))

nodes = [np.array([[start[0]], [start[1]]])]
for i in range(1, N_STEPS + 1):
    ys = center_y[i] + np.linspace(-spread[i], spread[i], NODES_PER_STEP)
    nodes.append(np.stack([np.full(NODES_PER_STEP, xs[i]), ys]))

for i in range(N_STEPS):
    x0s, y0s = nodes[i]
    x1s, y1s = nodes[i + 1]
    for x0, y0 in zip(x0s, y0s):
        for x1, y1 in zip(x1s, y1s):
            ax.plot([x0, x1], [y0, y1], color=CANDIDATE_EDGE, linewidth=0.6, alpha=0.35, zorder=2)

for i in range(1, N_STEPS + 1):
    xk, yk = nodes[i]
    ax.scatter(xk, yk, s=55, facecolor=NODE_FACE, edgecolor=NODE_EDGE, linewidth=1.1, zorder=3)

chosen = [start]
for i in range(1, N_STEPS + 1):
    xk, yk = nodes[i]
    frac = i / N_STEPS
    aim_y = start[1] + frac * (target_xy[1] - start[1])
    j = int(np.argmin(np.abs(yk - aim_y)))
    chosen.append(np.array([xk[j], yk[j]]))
chosen = np.array(chosen)

ax.plot(chosen[:, 0], chosen[:, 1], color=ACCENT, linewidth=2.6, zorder=4, solid_capstyle="round")
ax.scatter(chosen[1:, 0], chosen[1:, 1], s=75, facecolor=ACCENT, edgecolor="white", linewidth=1.3, zorder=5)
ax.scatter(*start, s=170, facecolor=START_COLOR, edgecolor="white", linewidth=1.6, zorder=6)

ax.text(start[0], start[1] - 0.55, "current position", ha="center", va="top", fontsize=10, color=LABEL_COLOR)
ax.text(xs[2], center_y[2] + spread[2] + 0.35, "candidate waypoints\n(grid lattice)",
        ha="center", va="bottom", fontsize=10, color=LABEL_COLOR)
ax.text(chosen[-1, 0] + 0.15, chosen[-1, 1] - 0.55, "selected path",
        ha="left", va="top", fontsize=10, color=ACCENT)
ax.text((xs[0] + xs[-1]) / 2, min(center_y) - spread[-1] - 0.55, "planning horizon",
        ha="center", va="top", fontsize=10, color=LABEL_COLOR)
ax.annotate(
    "", xy=(xs[-1], min(center_y) - spread[-1] - 0.35), xytext=(xs[0], min(center_y) - spread[-1] - 0.35),
    arrowprops=dict(arrowstyle="-", color=LABEL_COLOR, linewidth=1.0), zorder=2,
)

ax.set_xlim(-0.7, target_xy[0] + 1.4)
ax.set_ylim(min(center_y) - spread[-1] - 1.0, target_xy[1] + 1.3)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, transparent=True)
plt.close(fig)
print(f"Wrote {OUT_PNG}")
