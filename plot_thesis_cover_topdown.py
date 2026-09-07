"""
High-resolution top-down cover-page background: the real ground truth
terrain (map 11, NAIP), viridis (blue -> green -> yellow), with ONE Diffusion
plan (generate_diffusion_denoising_cover_trajectory.py's output) swept over
it as a glowing contrast trail. No axes/legend/title - this is meant as a
pure background image, not a data figure.

The "final trajectory" drawn here is deliberately just that one sampled
plan's own fully-resolved denoising step (step 20 of 20) - not a separately,
independently generated longer flight. Earlier this used a longer flight
from a different script/sampling call, which visibly didn't connect to the
echoes (Diffusion's sampling noise isn't seeded, so two separate sampling
calls never produce the same plan even from identical conditioning). Using
the same sample for both the echoes and the final path fixes that by
construction: every "echo" trail below is an earlier denoising step of the
exact path that's drawn as the final trajectory, not a look-alike.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_ID = 11
MAPTYPE = "NAIP"
GRID_CSV = SCRIPT_DIR / "csv" / f"map_{MAP_ID}_{MAPTYPE}_grid_counts.csv"
DENOISING_STEPS = list(range(21))
FINAL_STEP = DENOISING_STEPS[-1]  # the fully resolved plan - this IS the "final trajectory"
TRAJ_CSV = SCRIPT_DIR / "hero_shot_data" / f"map_{MAP_ID}_denoising_step_{FINAL_STEP:02d}_of_20.csv"
DENOISING_CSV_TEMPLATE = str(
    SCRIPT_DIR / "hero_shot_data" / "map_11_denoising_step_{step:02d}_of_20.csv"
)
DENOISING_SPARSE_CSV_TEMPLATE = str(
    SCRIPT_DIR / "hero_shot_data" / "map_11_denoising_step_{step:02d}_of_20_sparse.csv"
)
POINT_CLOUD_STEPS = {0, 1, 2}  # too noisy for a spline to read as "noise" - show raw points instead
OUT_PATH = SCRIPT_DIR / f"thesis_cover_topdown_map{MAP_ID}_{MAPTYPE}_diffusion_denoising.png"

TRAJECTORY_FRACTION = 1.0  # fraction of each trajectory to plot (main + echoes); 1.0 = full length
SMOOTH_SIGMA = 2.0
DPI = 300
FIGSIZE = (10, 10)  # square; crop/extend to your cover's aspect ratio later

TRAIL_COLOR = "#ffb8a0"  # peach tint, only the final flown trajectory
TRAIL_GLOW_COLOR = "#ffffff"  # glow/halo stays neutral white so it doesn't muddy the tint
DENOISE_COLOR_EARLY = "#ff9d3d"  # warm gold-orange: the chaotic, unresolved steps
DENOISE_COLOR_LATE = "#ffffff"   # echoes (the "previous" diffusion steps) stay plain white


def load_denoising_echoes():
    echoes = []
    for step in DENOISING_STEPS:
        data = np.loadtxt(DENOISING_CSV_TEMPLATE.format(step=step), delimiter=",", skiprows=1)
        sparse = np.loadtxt(DENOISING_SPARSE_CSV_TEMPLATE.format(step=step), delimiter=",", skiprows=1)
        echoes.append((step, data[:, 0], data[:, 1], sparse[:, 0], sparse[:, 1]))
    return echoes


def load_ground_truth_grid(path, step=2.0):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = (
        np.isclose(np.mod(data[:, 0], step), 0.0, atol=1e-9)
        & np.isclose(np.mod(data[:, 1], step), 0.0, atol=1e-9)
    )
    data = data[mask]
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    grid = np.full((len(ys), len(xs)), np.nan)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    for x, y, val in data:
        grid[y_idx[y], x_idx[x]] = val
    return xs, ys, grid


def load_trajectory(path):
    # Denoising-step CSVs are "x,y,z" (no leading timestep column, unlike
    # the old hero_trajectory_*.csv format this used to read).
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]  # x, y


def truncate(*arrays, fraction=TRAJECTORY_FRACTION):
    n = max(int(round(len(arrays[0]) * fraction)), 2)
    return tuple(a[:n] for a in arrays)


def main():
    xs, ys, grid = load_ground_truth_grid(GRID_CSV)
    tx, ty = load_trajectory(TRAJ_CSV)
    tx, ty = truncate(tx, ty)

    plot_x = gaussian_filter1d(tx, sigma=SMOOTH_SIGMA)
    plot_y = gaussian_filter1d(ty, sigma=SMOOTH_SIGMA)
    plot_x[0], plot_y[0] = tx[0], ty[0]
    plot_x[-1], plot_y[-1] = tx[-1], ty[-1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(
        grid, origin="lower", cmap="viridis",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        interpolation="bicubic", zorder=1,
    )

    echoes = load_denoising_echoes()
    n_echo = len(echoes)
    early_rgb, late_rgb = np.array(to_rgb(DENOISE_COLOR_EARLY)), np.array(to_rgb(DENOISE_COLOR_LATE))
    for i, (step, ex, ey, sx, sy) in enumerate(echoes):
        if step in POINT_CLOUD_STEPS:
            continue  # too noisy to read as a path at all; skip rather than dot-scatter
        if step == FINAL_STEP:
            continue  # this IS the main trail (drawn below, more prominently) - don't double-draw it
        ex, ey = truncate(ex, ey)
        frac = i / (n_echo - 1)  # 0 = pure noise, 1 = fully resolved
        color = tuple(early_rgb + min(frac * 1.6, 1.0) * (late_rgb - early_rgb))

        # Steps 3..20: fading in as connected (still visibly wild, early on)
        # paths. Kept deliberately faint through the middle of the sequence
        # so they read as one soft haze rather than N competing loops; only
        # the last few steps step up in weight/opacity toward the final path.
        alpha = 0.05 + 0.75 * frac ** 3.2
        linewidth = 0.8 + 1.8 * max(frac - 0.5, 0) * 2
        sigma = SMOOTH_SIGMA * frac
        px = gaussian_filter1d(ex, sigma=sigma) if sigma > 0.1 else ex
        py = gaussian_filter1d(ey, sigma=sigma) if sigma > 0.1 else ey
        if frac > 0.9:
            for lw, a in [(linewidth + 5, alpha * 0.15), (linewidth + 2.5, alpha * 0.22)]:
                ax.plot(px, py, color=color, linewidth=lw, alpha=a, solid_capstyle="round", zorder=2)
        ax.plot(px, py, color=color, linewidth=linewidth, alpha=alpha,
                 solid_capstyle="round", zorder=3)

    # Soft glow: several wide passes underneath the crisp line, stronger than
    # before for more separation from the terrain.
    for lw, alpha in [(16, 0.09), (12, 0.12), (8, 0.17), (5, 0.26)]:
        ax.plot(plot_x, plot_y, color=TRAIL_GLOW_COLOR, linewidth=lw,
                 alpha=alpha, solid_capstyle="round", zorder=4)

    # Directional taper (thin/dim tail -> thick/bright head) so the line
    # reads as a flown trajectory, not a static string laid on the map.
    # Upsampled first: per-segment width/alpha changes must be imperceptibly
    # small between neighbors, or round line caps at each joint show up as a
    # faint beaded/dashed texture instead of one smooth taper.
    param = np.linspace(0.0, 1.0, len(plot_x))
    fine_t = np.linspace(0.0, 1.0, 12 * len(plot_x))
    fine_x = np.interp(fine_t, param, plot_x)
    fine_y = np.interp(fine_t, param, plot_y)

    points = np.array([fine_x, fine_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.linspace(0.0, 1.0, len(segments))
    widths = 1.1 + 2.6 * t ** 1.3
    alphas = 0.55 + 0.45 * t

    dark_lc = LineCollection(
        segments, linewidths=widths + 1.6, colors=[(0.043, 0.102, 0.169, a * 0.8) for a in alphas],
        capstyle="round", zorder=5,
    )
    ax.add_collection(dark_lc)
    core_lc = LineCollection(
        segments, linewidths=widths, colors=[to_rgb(TRAIL_COLOR) + (a,) for a in alphas],
        capstyle="round", zorder=6,
    )
    ax.add_collection(core_lc)

    # Direction-of-travel arrowheads at evenly spaced points along the arc
    # length (not by index - the flight lingers in some areas, so equal
    # index-spacing would bunch arrows up unevenly in space).
    seg_len = np.hypot(np.diff(fine_x), np.diff(fine_y))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    n_arrows = 9
    margin = 0.06  # keep clear of the very start (too faint) and very end
    arrow_arc = np.linspace(margin, 1 - margin, n_arrows) * arc[-1]
    half_step = 0.006 * arc[-1]

    for a in arrow_arc:
        i0 = np.searchsorted(arc, max(a - half_step, 0.0))
        i1 = np.searchsorted(arc, min(a + half_step, arc[-1]))
        i1 = max(i1, i0 + 1)
        i0, i1 = min(i0, len(fine_x) - 2), min(i1, len(fine_x) - 1)
        frac = a / arc[-1]
        width = 1.1 + 2.6 * frac ** 1.3
        alpha = 0.55 + 0.45 * frac
        ax.annotate(
            "", xy=(fine_x[i1], fine_y[i1]), xytext=(fine_x[i0], fine_y[i0]),
            arrowprops=dict(
                arrowstyle="-|>,head_length=" + f"{0.55 + 0.9*frac:.2f}" + ",head_width=" + f"{0.35 + 0.55*frac:.2f}",
                color=to_rgb(TRAIL_COLOR) + (alpha,), linewidth=width * 0.6,
                shrinkA=0, shrinkB=0,
            ),
            zorder=7,
        )

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {OUT_PATH} ({FIGSIZE[0]*DPI}x{FIGSIZE[1]*DPI}px)")


if __name__ == "__main__":
    main()
