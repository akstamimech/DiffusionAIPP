"""
Illustrates the forward noising process (threeDSparseTransDiffusion.forward_diffusion_sample)
applied to one real trajectory sampled from the live diffusion planner (Diffusionplanner_singlemap.py),
at five increasingly noisy diffusion steps - rendered two ways from the SAME noise draws:
  1. forward_noise_trajectory.png - the resulting spline-rendered flight path, real-world scale.
  2. trajectory_as_pixels.png - the raw 3x8 tensor itself, rendered as a pixel grid.

x0 is the planner's actual first-replan output (real belief prior, real start pose), taken as the
raw normalized [-1,1]-space tensor the network operates on (i.e. BEFORE
extract_control_waypoints/denormalize_xyz) - not a training-dataset sample - so this reuses the
real planner + real forward-process code directly rather than reimplementing either.

x_K = sqrt(alpha_bar_K) * x0 + sqrt(1 - alpha_bar_K) * epsilon, applied via forward_diffusion_sample
itself, at K = 0, 5, 10, 15, 19 (of T=20; 19 is the true final/most-noised step - K=20 doesn't exist
in a 0-indexed 20-step schedule).

Notation: K is the diffusion timestep (this script's t_val/K), matching the paper's own use of K for
the diffusion step - distinct from t, which the paper reserves for the AIPP replanning/mission time step.
"""
import os

os.environ.setdefault("SELECTED_MAP", "51")
os.environ.setdefault("MAPTYPE", "grf")

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import Diffusionplanner_singlemap as dp
from sample_3d_sparse_trans_diffusion import diffusion

SCRIPT_DIR = Path(__file__).resolve().parent
NOISE_STEPS = [0, 5, 10, 15, 19]  # T=20 means valid steps are 0..19

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_PATH = "#eb6834"   # categorical slot 2 (orange)
COLOR_START = "#1baf7a"  # categorical slot 3 (aqua)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def build_x0(x0_seed):
    """Reproduces sample_diffusion_trajectory's conditioning + ddim_sample call, but returns the
    raw normalized [1,3,8] sample instead of denormalized waypoints - that raw tensor is what
    forward_diffusion_sample actually corrupts during training."""
    model = dp.load_diffusion_model(checkpoint_path=dp.diffusion_path)

    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    data = np.loadtxt(
        csv_path / f"map_{dp.selected_map}_{dp.MAPTYPE}_grid_counts.csv",
        delimiter=",", skiprows=1,
    )
    _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = dp.initialize_gp()
    mean = np.full(X_test.shape[0], dp.utility_threshold + 0.1)

    current_position = (dp.START_X, dp.START_Y, dp.INIT_ALTITUDE)
    current_position_world = torch.as_tensor(current_position, dtype=torch.float32, device=diffusion.device).view(1, 3)
    current_position_model = diffusion.normalize_xyz(current_position_world)

    current_mean = torch.tensor(mean.reshape(X.shape), dtype=torch.float32, device=diffusion.device)
    current_var = torch.tensor(np.diag(cov).reshape(X.shape), dtype=torch.float32, device=diffusion.device)

    mean_map = (current_mean - diffusion.mean_center.to(diffusion.device)) / diffusion.mean_scale.to(diffusion.device)
    var_map = (current_var - diffusion.var_center.to(diffusion.device)) / diffusion.var_scale.to(diffusion.device)
    total_variance_condition = diffusion.normalize_total_variance(
        current_var.reshape(1, -1).sum(dim=1, keepdim=True)
    ).to(diffusion.device)
    marker_map = diffusion.make_position_marker_maps(
        current_position_world[:, :2].cpu(), grid_size=51, marker_radius=2,
    ).to(diffusion.device)[0, 0]
    meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)
    initial_heading_velocity = torch.zeros((1, 3), dtype=torch.float32, device=diffusion.device)

    torch.manual_seed(x0_seed)
    sparse_noise = torch.randn((1, *diffusion.TARGET_SHAPE), device=diffusion.device)
    with torch.no_grad():
        x0, _, _ = diffusion.ddim_sample(
            sparse_noise, meanvarmarker_map,
            current_position=current_position_model,
            initial_heading_velocity=initial_heading_velocity,
            total_variance_condition=total_variance_condition,
            eta=1.0,
        )
    return x0.detach(), current_position_world[0].detach()


def build_noisy_steps(x0, start_world, noise_seed_base):
    """One noise draw per K (seeded K-by-K so the pixel grid and the trajectory render use
    IDENTICAL x_K tensors), returning both the raw normalized array (for the pixel plot) and the
    denormalized-without-clamping spline (for the trajectory plot)."""
    steps = []
    for k_val in NOISE_STEPS:
        torch.manual_seed(noise_seed_base + k_val)
        k = torch.full((1,), k_val, dtype=torch.long)
        x_k, _ = diffusion.forward_diffusion_sample(x0, k)  # normalized space
        sqrt_ab = diffusion.sqrt_alphas_cumprod[k_val].item()

        waypoints_world = diffusion.denormalize_xyz(x_k)[0]  # [3, 8], meters/altitude, unclamped
        with torch.no_grad():
            dense_spline = diffusion.pytorch_cubic_spline(waypoints_world, current_position=start_world)[0]

        steps.append({
            "k": k_val,
            "sqrt_ab": sqrt_ab,
            "normalized": x_k[0].numpy(),
            "waypoints_world": waypoints_world.numpy(),
            "dense_spline": dense_spline.numpy(),
        })
    return steps


def render_trajectory_frames(steps, start_world, out_path):
    all_xy = np.concatenate(
        [s["dense_spline"][:2].reshape(2, -1) for s in steps]
        + [s["waypoints_world"][:2].reshape(2, -1) for s in steps],
        axis=1,
    )
    x_min, x_max = float(np.min(all_xy[0])), float(np.max(all_xy[0]))
    y_min, y_max = float(np.min(all_xy[1])), float(np.max(all_xy[1]))
    pad_x = (x_max - x_min) * 0.08 + 1
    pad_y = (y_max - y_min) * 0.08 + 1

    fig, axes = plt.subplots(1, len(steps), figsize=(3.1 * len(steps), 3.4))
    for ax, s in zip(axes, steps):
        ax.add_patch(plt.Rectangle(
            (0, 0), 100, 100, fill=False, edgecolor=INK_MUTED,
            linewidth=1.0, linestyle="--", alpha=0.7, zorder=1,
        ))
        ax.plot(s["dense_spline"][0], s["dense_spline"][1], color=COLOR_PATH, linewidth=1.8, zorder=3)
        ax.scatter(
            s["waypoints_world"][0], s["waypoints_world"][1], color=COLOR_PATH, s=22,
            zorder=4, edgecolors=SURFACE, linewidths=0.6, alpha=0.85,
        )
        ax.scatter(
            [start_world[0].item()], [start_world[1].item()], color=COLOR_START,
            s=50, zorder=5, edgecolors=SURFACE, linewidths=0.8,
        )

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.set_title(f"K = {s['k']}   ($\\sqrt{{\\bar\\alpha_K}}$ = {s['sqrt_ab']:.2f})", fontsize=12, color=INK_PRIMARY)
        ax.axis("off")

    fig.suptitle(
        "Forward noising of one planner-sampled trajectory, real-world scale "
        "($x_K=\\sqrt{\\bar\\alpha_K}\\,x_0+\\sqrt{1-\\bar\\alpha_K}\\,\\epsilon$, dashed box = the 100x100 m map)",
        fontsize=12, color=INK_PRIMARY, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def render_pixel_grid(steps, out_path):
    # Transposed to [8, 3]: rows = waypoint index 1-8, columns = x/y/z - "vertical" pixel grid.
    vmax = max(np.max(np.abs(s["normalized"])) for s in steps)

    fig, axes = plt.subplots(1, len(steps), figsize=(2.0 * len(steps), 4.6))
    im = None
    for ax, s in zip(axes, steps):
        pixels = s["normalized"].T  # [8, 3]
        im = ax.imshow(
            pixels, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
            interpolation="nearest", aspect="equal",
        )
        ax.set_title(f"K={s['k']}\n" + r"($\sqrt{\bar\alpha_K}$=" + f"{s['sqrt_ab']:.2f})", fontsize=10.5, color=INK_PRIMARY)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["x", "y", "z"], fontsize=8, color=INK_SECONDARY)
        ax.set_yticks(range(8))
        ax.set_yticklabels(range(1, 9), fontsize=7, color=INK_SECONDARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("waypoint index", color=INK_SECONDARY, fontsize=9)

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, aspect=15)
    cbar.set_label("normalized value", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    fig.suptitle("The 8x3 trajectory tensor treated as pixel values, at increasing noise steps", fontsize=12.5, color=INK_PRIMARY, y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def run(x0_seed, noise_seed_base, run_tag):
    x0, start_world = build_x0(x0_seed)
    print(f"[{run_tag}] x0 range: [{x0.min().item():.3f}, {x0.max().item():.3f}]")
    steps = build_noisy_steps(x0, start_world, noise_seed_base)
    render_trajectory_frames(steps, start_world, SCRIPT_DIR / f"forward_noise_trajectory_{run_tag}.png")
    render_pixel_grid(steps, SCRIPT_DIR / f"trajectory_as_pixels_{run_tag}.png")


def main():
    # Both prior runs: (x0_seed=1, noise_seed_base=100) and (x0_seed=7, noise_seed_base=200).
    run(x0_seed=1, noise_seed_base=100, run_tag="run1")
    run(x0_seed=7, noise_seed_base=200, run_tag="run2")


if __name__ == "__main__":
    main()
