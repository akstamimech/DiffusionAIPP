"""
Animates the reverse (denoising) DDIM process for one real planner replan, one frame per
diffusion step, from pure noise down to the final trajectory.

diffusion.ddim_sample only returns the FINAL sample (it builds an internal samples_list but never
returns it), so this calls diffusion.ddim_sample_timestep directly in its own loop - identical
step logic and arguments to ddim_sample's own loop - to capture every intermediate x_t. Everything
else (conditioning setup, denormalize-without-clamp, spline rendering) reuses
plot_forward_noise_trajectory.py's approach for a consistent look between the two figures.

NOTE on eta: Diffusionplanner_singlemap.py's own sample_diffusion_trajectory calls ddim_sample with
eta=ETA, where ETA is THIS file's local `ETA = float(os.environ.get("ETA", "0.0"))` (Diffusionplanner_
singlemap.py's own module-level constant) - i.e. the live planner is DETERMINISTIC DDIM by default.
This is a correction to something said earlier in this session: sample_3d_sparse_trans_diffusion.py
also defines an ETA=1.0, but that constant belongs to that separate standalone test script and is
never imported or used by Diffusionplanner_singlemap.py - the two are unrelated despite the shared
name. This animation uses eta=0 to match what the real planner actually runs by default.
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
import imageio.v2 as imageio

import Diffusionplanner_singlemap as dp
from sample_3d_sparse_trans_diffusion import diffusion

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "reverse_process_animation.gif"
ETA = 0.0  # matches Diffusionplanner_singlemap.py's real default (deterministic DDIM)
FRAME_DURATION = 0.35  # seconds per frame; last frame held longer below

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
COLOR_PATH = "#eb6834"   # categorical slot 2 (orange)
COLOR_START = "#1baf7a"  # categorical slot 3 (aqua)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def build_conditioning():
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

    return meanvarmarker_map, current_position_model, initial_heading_velocity, total_variance_condition, current_position_world[0]


def run_reverse_process(meanvarmarker_map, current_position_model, initial_heading_velocity, total_variance_condition):
    """Mirrors diffusion.ddim_sample's own loop exactly, but keeps every intermediate x."""
    torch.manual_seed(2)
    x = torch.randn((1, *diffusion.TARGET_SHAPE), device=diffusion.device)
    schedule = list(range(diffusion.T - 1, -1, -1))  # 19, 18, ..., 0

    frames = [{"label": "start (pure noise)", "sqrt_ab": 0.0, "x": x.clone()}]
    with torch.no_grad():
        for step_idx, i in enumerate(schedule):
            prev_i = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
            t = torch.full((1,), i, dtype=torch.long, device=x.device)
            t_prev = torch.full((1,), prev_i, dtype=torch.long, device=x.device)
            x, mean_theta, _ = diffusion.ddim_sample_timestep(
                x, t, t_prev, meanvarmarker_map, current_position_model,
                initial_heading_velocity, total_variance_condition,
                clip_x0=True, eta=ETA, sigma_prob_min=0.1,
            )
            sqrt_ab = diffusion.sqrt_alphas_cumprod[max(prev_i, 0)].item() if prev_i >= 0 else 1.0
            label = f"t = {prev_i}" if prev_i >= 0 else "final (t = 0, denoised)"
            frames.append({"label": label, "sqrt_ab": sqrt_ab, "x": x.clone()})
    return frames


def main():
    meanvarmarker_map, current_position_model, initial_heading_velocity, total_variance_condition, start_world = build_conditioning()
    frames = run_reverse_process(meanvarmarker_map, current_position_model, initial_heading_velocity, total_variance_condition)
    print(f"Captured {len(frames)} frames")

    # Denormalize every frame WITHOUT clamping (same rationale as plot_forward_noise_trajectory.py)
    # and build its spline, then compute one shared axis range across all frames up front so the
    # animation visibly "shrinks" from wide-scattered noise down to the converged trajectory.
    rendered = []
    for f in frames:
        waypoints_world = diffusion.denormalize_xyz(f["x"])[0]  # [3, 8]
        dense_spline = diffusion.pytorch_cubic_spline(waypoints_world, current_position=start_world)[0]
        rendered.append((f["label"], f["sqrt_ab"], waypoints_world.numpy(), dense_spline.numpy()))

    all_xy = np.concatenate([r[3][:2].reshape(2, -1) for r in rendered], axis=1)
    x_min, x_max = float(np.min(all_xy[0])), float(np.max(all_xy[0]))
    y_min, y_max = float(np.min(all_xy[1])), float(np.max(all_xy[1]))
    pad_x = (x_max - x_min) * 0.06 + 1
    pad_y = (y_max - y_min) * 0.06 + 1

    gif_frames = []
    for idx, (label, sqrt_ab, waypoints_world, dense_spline) in enumerate(rendered):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.add_patch(plt.Rectangle(
            (0, 0), 100, 100, fill=False, edgecolor=INK_MUTED,
            linewidth=1.0, linestyle="--", alpha=0.7, zorder=1,
        ))
        ax.plot(dense_spline[0], dense_spline[1], color=COLOR_PATH, linewidth=2.0, zorder=3)
        ax.scatter(
            waypoints_world[0], waypoints_world[1], color=COLOR_PATH, s=26,
            zorder=4, edgecolors=SURFACE, linewidths=0.6, alpha=0.9,
        )
        ax.scatter(
            [start_world[0].item()], [start_world[1].item()], color=COLOR_START,
            s=55, zorder=5, edgecolors=SURFACE, linewidths=0.8,
        )

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.set_title(
            f"step {idx}/{len(rendered)-1}   {label}   ($\\sqrt{{\\bar\\alpha}}$={sqrt_ab:.2f})",
            fontsize=11.5, color=INK_PRIMARY,
        )
        ax.set_xlabel("x (m)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("y (m)", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(INK_MUTED)

        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        gif_frames.append(frame_rgba[:, :, :3].copy())
        plt.close(fig)

    durations = [FRAME_DURATION] * len(gif_frames)
    durations[0] = 1.0   # hold the starting pure-noise frame
    durations[-1] = 2.0  # hold the final converged trajectory
    imageio.mimsave(OUT_PATH, gif_frames, duration=durations)
    print(f"Wrote {OUT_PATH} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
