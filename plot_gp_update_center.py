"""
One Kalman/GP update at the center of a real GRF map (map 51), using the
real update mechanics (fov_grid_points -> build_sensor_matrix ->
kalman_update, same pipeline CMAES_classic_singlemap.py runs every
timestep), starting from the same flat prior as plot_gp_prior.py /
plot_gp_prior_variance.py.

The "true" field read by the sensor is map 51's actual ground truth
(csv/map_51_grf_grid_counts.csv), read with the same load+mask convention
CMAES_classic_singlemap.py itself uses. No random sensor noise is added to
z_meas - only to keep this a clean, reproducible, deterministic worked
example of the update mechanics. R itself still uses the real
noise_model(altitude) value, so the Kalman gain (and therefore how far the
mean and variance actually move) is realistic.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from gaussianprocesstraining import (
    initialize_gp,
    fov_grid_points,
    fov_lateral_radius,
    build_sensor_matrix,
    build_correlated_noise_covariance,
    noise_model,
    kalman_update,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "csv" / "map_51_grf_grid_counts.csv"
OUT_MEAN = SCRIPT_DIR / "gp_update_center_mean.png"
OUT_VAR = SCRIPT_DIR / "gp_update_center_variance.png"

UTILITY_THRESHOLD = 0.5
PRIOR_MEAN = UTILITY_THRESHOLD + 0.1  # 0.6, same as plot_gp_prior.py
CENTER = (50.0, 50.0)
ALTITUDE = 25.0  # bigger FOV than the 10m default - mid altitude tier (block 2x2)
ANGLE_OF_VIEW = 60.0

plt.rcParams.update({"mathtext.fontset": "cm"})


def load_true_map_flat(csv_path, X_test, step):
    """Mirrors CMAES_classic_singlemap.py's own ground-truth load block."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pts = data[:, 0:3]
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=1e-9)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=1e-9)
    )
    pts = pts[mask]
    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(np.isclose(X_test[:, 0], x_true) & np.isclose(X_test[:, 1], y_true))[0]
        if idx.size:
            true_map_flat[idx[0]] = value
    return true_map_flat


def add_inset_colorbar(fig, ax, im, label):
    """Colorbar drawn inside the axes bounds instead of appended outside.
    Measures its own actual rendered bounding box (tick labels + axis label
    included) after a draw, then nudges it left/down if that box would spill
    past the axes edge (the plot border) - and only then sizes the white
    backing box to the corrected, now-guaranteed-to-fit bbox."""
    cax = inset_axes(
        ax, width="10%", height="32%", loc="lower right",
        bbox_to_anchor=(-0.06, 0.06, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cax.set_zorder(10)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    ax_bbox_fig = ax.get_position()

    for _ in range(3):
        bbox_px = cax.get_tightbbox(renderer)
        x0, y0 = inv.transform((bbox_px.x0, bbox_px.y0))
        x1, y1 = inv.transform((bbox_px.x1, bbox_px.y1))
        margin = 0.01
        dx = max(0.0, (x1 - (1.0 - margin)) * ax_bbox_fig.width)
        dy = max(0.0, ((0.0 + margin) - y0) * ax_bbox_fig.height)
        if dx <= 1e-6 and dy <= 1e-6:
            break
        pos = cax.get_position()
        cax.set_position([pos.x0 - dx, pos.y0 + dy, pos.width, pos.height])
        fig.canvas.draw()

    bbox_px = cax.get_tightbbox(renderer)
    x0, y0 = inv.transform((bbox_px.x0, bbox_px.y0))
    x1, y1 = inv.transform((bbox_px.x1, bbox_px.y1))
    pad = 0.015
    backing = Rectangle(
        (x0 - pad, y0 - pad), (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
        transform=ax.transAxes, facecolor="white", alpha=0.85,
        edgecolor="black", linewidth=1.0, zorder=9,
    )
    ax.add_patch(backing)


def main():
    gp, X_test, _mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    mu = np.full(X_test.shape[0], PRIOR_MEAN)
    P = cov.copy()

    cx, cy = CENTER
    fov = fov_grid_points(cx, cy, ALTITUDE, xs, ys, angle_of_view=ANGLE_OF_VIEW)
    sensor, block_ids = build_sensor_matrix(fov, ALTITUDE, xs, ys, return_block_ids=True)

    true_map_flat = load_true_map_flat(CSV_PATH, X_test, step)
    z_meas = sensor @ true_map_flat
    print(f"Real map-51 ground truth values sensed in the FOV: "
          f"min={z_meas.min():.3f} max={z_meas.max():.3f} mean={z_meas.mean():.3f}")

    R = noise_model(ALTITUDE)
    R_cov = build_correlated_noise_covariance(block_ids, R)

    mu_post, P_post = kalman_update(mu, P, sensor, z_meas, R_cov, block_ids=block_ids)

    mean_grid = mu_post.reshape(len(ys), len(xs))
    var_grid = np.diag(P_post).reshape(len(ys), len(xs))
    radius = fov_lateral_radius(ALTITUDE, ANGLE_OF_VIEW)

    # (0.3, 0.8) is not an arbitrary tighter crop - it's chosen so that
    # (PRIOR_MEAN - vmin) / (vmax - vmin) still equals PRIOR_MEAN itself
    # (0.3/0.5 = 0.6), i.e. the untouched far field still lands on the exact
    # same viridis(0.6) green as plot_gp_prior.py's fixed 0-1 scale, while
    # the color range is half as wide - so the same real deviation from the
    # prior now covers twice as much of the colormap.
    mean_vmin, mean_vmax = 0.3, 0.8

    for out_path, grid, title, cbar_label, cmap, vrange in [
        (OUT_MEAN, mean_grid, "2D GP mean after center update", "mean", "viridis", (mean_vmin, mean_vmax)),
        (OUT_VAR, var_grid, "2D GP variance after center update", "variance", "viridis", (None, None)),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid, origin="lower", cmap=cmap, extent=[xmin, xmax, ymin, ymax],
                        vmin=vrange[0], vmax=vrange[1])
        # fov_grid_points filters xs and ys independently (a square bounding
        # box), not a circular disc, so this is the real footprint shape.
        ax.add_patch(Rectangle(
            (cx - radius, cy - radius), 2 * radius, 2 * radius,
            facecolor="none", edgecolor="white", linewidth=1.6, linestyle="--",
        ))
        ax.scatter([cx], [cy], color="white", edgecolors="black", s=60, zorder=5)
        add_inset_colorbar(fig, ax, im, cbar_label)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
