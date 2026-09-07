"""
Splits plot_fov_resolution_altitude.py's single combined figure (side-view FOV
triangle + 3-panel resolution block grid, sharing one gridspec) into two
independent standalone PNGs, reusing its exact functions/constants rather
than reimplementing anything. Panel 1 (noise vs. altitude) already exists
standalone as sensor_noise_model.png - no need to touch that one.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fov_resolution_altitude import (
    ALTITUDES, TIER_COLORS, MAP_ID, MAPTYPE, CSV_PATH, CENTER, FIG_WIDTH,
    SEQUENTIAL_CMAP, INK_SECONDARY, INK_PRIMARY, INK_MUTED,
    draw_side_view, observed_image_for_altitude, load_true_map_flat,
)
from gaussianprocesstraining import initialize_gp, fov_lateral_radius, resolution_block_size

SCRIPT_DIR = Path(__file__).resolve().parent
ANGLE_OF_VIEW = 60.0
OUT_FOV = SCRIPT_DIR / f"fov_vs_altitude_panel_map{MAP_ID}_{MAPTYPE}.png"
OUT_RESOLUTION = SCRIPT_DIR / f"resolution_blocks_panel_map{MAP_ID}_{MAPTYPE}.png"


def main():
    # --- Panel: FOV vs. altitude (side-view triangle diagram) alone ---
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, 3.6))
    draw_side_view(ax1)
    fig1.tight_layout()
    fig1.savefig(OUT_FOV, dpi=200)
    plt.close(fig1)
    print(f"Wrote {OUT_FOV}")

    # --- Panel: Resolution vs. altitude (3-panel sensed-value block grid) alone ---
    _, X_test, _, _, xs, ys, _, _, _, _, _, _, _ = initialize_gp()
    true_map_flat = load_true_map_flat(CSV_PATH, X_test)
    cx, cy = CENTER

    panels = []
    for z in ALTITUDES:
        img, extent = observed_image_for_altitude(cx, cy, z, xs, ys, true_map_flat)
        panels.append((z, img, extent))

    vmin = min(np.nanmin(img) for _, img, _ in panels)
    vmax = max(np.nanmax(img) for _, img, _ in panels)
    max_radius = fov_lateral_radius(40.0, ANGLE_OF_VIEW)
    half_window = max_radius + 4

    fig2, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 3.2))
    for ax, ((z, img, extent), color) in zip(axes, zip(panels, TIER_COLORS)):
        ax.imshow(
            img, extent=extent, origin="lower", cmap=SEQUENTIAL_CMAP,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        block = resolution_block_size(z)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(cx - half_window, cx + half_window)
        ax.set_ylim(cy - half_window, cy + half_window)
        ax.set_title(
            f"{z:.0f} m\nblock {block}x{block}",
            color=color, fontsize=14, fontweight="bold", linespacing=1.4,
        )

    mappable = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    mappable.set_array([])
    cbar = fig2.colorbar(mappable, ax=axes, shrink=0.85, pad=0.02, location="right", aspect=20)
    cbar.set_label("Sensed value", color=INK_SECONDARY, fontsize=13)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=11)

    fig2.suptitle("Resolution vs. altitude", color=INK_PRIMARY, fontsize=17)
    fig2.savefig(OUT_RESOLUTION, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {OUT_RESOLUTION}")


if __name__ == "__main__":
    main()
