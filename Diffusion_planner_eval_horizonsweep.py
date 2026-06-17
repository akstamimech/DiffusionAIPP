import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from gaussianprocesstraining import initialize_gp
from evalmetrics import (
    compute_coverage_efficiency,
    compute_rmse_time_metrics,
    compute_task_completion,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

import SparseTransDiffusion as diffusion


step = 2.0
timealloted = 200
MAP_ID = 996
RUNS_PER_CHUNK = 5
EXECUTION_CHUNKS = [20, 40, 60, 80]
MAPTYPE = "multiblob"
SENSORNOISE_SEED = 123
DIFFUSION_SAMPLE_SEED = 125
output_root = SCRIPT_DIR / "diffusion_horizon_sweep_metrics"

diffusion_path = DIFFUSION_DIR / "checkpoints" / "sparse_trans_waypoints_epoch_2100_dynamic.pth"


def parse_chunks(raw_chunks):
    chunks = [int(chunk.strip()) for chunk in raw_chunks.split(",") if chunk.strip()]
    if not chunks:
        raise ValueError("At least one execution chunk must be provided.")
    if any(chunk < 1 for chunk in chunks):
        raise ValueError(f"All execution chunks must be >= 1, got {chunks}")
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate diffusion planner on one map while sweeping execution_chunk."
        )
    )
    parser.add_argument("--map-id", type=int, default=MAP_ID)
    parser.add_argument(
        "--execution-chunks",
        default=",".join(str(chunk) for chunk in EXECUTION_CHUNKS),
        help="Comma-separated execution_chunk values, e.g. 20,40,60,80.",
    )
    parser.add_argument("--runs-per-chunk", type=int, default=RUNS_PER_CHUNK)
    parser.add_argument("--output-dir", type=Path, default=output_root)
    parser.add_argument("--summary-name", default="diffusion_horizon_sweep_summary.csv")
    return parser.parse_args()


def iter_chunk_runs(execution_chunks, runs_per_chunk):
    if runs_per_chunk < 1:
        raise ValueError(f"runs_per_chunk must be >= 1, got {runs_per_chunk}")

    for execution_chunk in execution_chunks:
        for run_index in range(runs_per_chunk):
            yield execution_chunk, run_index


def write_metric_summary(metriclist, summary_path):
    if not metriclist:
        print("No metrics to write.")
        return

    fieldnames = list(metriclist[0].keys())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metriclist)
    print(f"\nSaved summary metrics to {summary_path}")


def grid_indices_from_coords(x_coords, y_coords, xmin, ymin, gp_step, width, height):
    x_idx = np.rint((np.asarray(x_coords) - xmin) / gp_step).astype(int)
    y_idx = np.rint((np.asarray(y_coords) - ymin) / gp_step).astype(int)
    valid = (0 <= x_idx) & (x_idx < width) & (0 <= y_idx) & (y_idx < height)
    flat_idx = y_idx[valid] * width + x_idx[valid]
    return x_idx[valid], y_idx[valid], flat_idx


def build_fov_offsets(lateral_coverage, gp_step):
    offsets = np.arange(-lateral_coverage, lateral_coverage + 1e-9, gp_step)
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    return dx.ravel(), dy.ravel()


def fov_indices_and_measurements(cx, cy, fov_dx, fov_dy, true_map, xmin, ymin, gp_step):
    height, width = true_map.shape
    x_coords = cx + fov_dx
    y_coords = cy + fov_dy
    x_idx, y_idx, obs_idx = grid_indices_from_coords(
        x_coords,
        y_coords,
        xmin,
        ymin,
        gp_step,
        width,
        height,
    )
    return obs_idx, true_map[y_idx, x_idx]


def load_map_context(selected_map, xs, ys, xmin, ymin, gp_step):
    data = np.loadtxt(
        SCRIPT_DIR / "csv" / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], gp_step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], gp_step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    x_idx = np.rint((pts[:, 0] - xmin) / gp_step).astype(int)
    y_idx = np.rint((pts[:, 1] - ymin) / gp_step).astype(int)
    valid = (0 <= x_idx) & (x_idx < len(xs)) & (0 <= y_idx) & (y_idx < len(ys))
    true_map[y_idx[valid], x_idx[valid]] = pts[valid, 2]

    return {
        "pts": pts,
        "true_map": true_map,
    }


def kalman_update_indices(mu, P, obs_idx, z_meas, R):
    if len(obs_idx) == 0:
        return mu, P

    v = z_meas - mu[obs_idx]
    S = P[np.ix_(obs_idx, obs_idx)] + R * np.eye(len(obs_idx))
    K = np.linalg.solve(S, P[obs_idx, :]).T

    mu = mu + K @ v
    P = P - K @ P[obs_idx, :]
    return mu, P


def compute_reconstruction_rmse_from_true_map(mu, true_map):
    mean_map = np.asarray(mu, dtype=np.float32).reshape(true_map.shape)
    error = mean_map - true_map
    global_rmse = float(np.sqrt(np.mean(error ** 2)))

    occupied_mask = true_map > 0
    if np.any(occupied_mask):
        occupied_rmse = float(np.sqrt(np.mean(error[occupied_mask] ** 2)))
        weighted_rmse = float(
            np.sqrt(np.sum(true_map * error ** 2) / np.sum(true_map))
        )
    else:
        occupied_rmse = 0.0
        weighted_rmse = 0.0

    return {
        "global_rmse": global_rmse,
        "occupied_rmse": occupied_rmse,
        "weighted_rmse": weighted_rmse,
    }


def load_diffusion_model(checkpoint_path=None):
    model = diffusion.NoisePredictor().to(diffusion.device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=diffusion.device)
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        state_dict = diffusion.remap_legacy_state_dict_keys(state_dict)
        model.load_state_dict(state_dict)
    model.eval()
    diffusion.model = model
    return model


@torch.no_grad()
def sample_diffusion_trajectory(
    model,
    current_position,
    current_mean,
    current_var,
    current_heading_velocity=None,
    grid_step=step,
    bounds=None,
):
    model.eval()
    diffusion.model = model

    current_position_world = torch.as_tensor(
        current_position, dtype=torch.float32, device=diffusion.device
    ).view(1, 2)
    current_position_model = (current_position_world / diffusion.SCALE_FACTOR) * 2.0 - 1.0

    current_mean = torch.as_tensor(current_mean, dtype=torch.float32, device=diffusion.device)
    current_var = torch.as_tensor(current_var, dtype=torch.float32, device=diffusion.device)

    mean_map = (
        current_mean - diffusion.mean_center.to(diffusion.device)
    ) / diffusion.mean_scale.to(diffusion.device)
    var_map = (
        current_var - diffusion.var_center.to(diffusion.device)
    ) / diffusion.var_scale.to(diffusion.device)
    marker_map = diffusion.make_position_marker_maps(
        current_position_world.cpu(),
        grid_size=51,
        marker_radius=2,
    ).to(diffusion.device)[0, 0]
    meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)

    if current_heading_velocity is None:
        initial_heading_velocity = torch.zeros((1, 2), dtype=torch.float32, device=diffusion.device)
    else:
        current_heading_velocity = torch.as_tensor(
            current_heading_velocity,
            dtype=torch.float32,
            device=diffusion.device,
        ).view(1, 2)
        initial_heading_velocity = current_heading_velocity * (2.0 / diffusion.SCALE_FACTOR)

    sparse_noise = torch.randn((1, *diffusion.TARGET_SHAPE), device=diffusion.device)
    sparse_sample = diffusion.ddim_sample(
        sparse_noise,
        meanvarmarker_map,
        current_position=current_position_model,
        initial_heading_velocity=initial_heading_velocity,
    )

    control_waypoints = diffusion.extract_control_waypoints(sparse_sample[0])
    dense_traj = diffusion.pytorch_cubic_spline(
        control_waypoints,
        current_position=current_position_world[0],
    )[0]
    dense_traj = grid_step * torch.round(dense_traj / grid_step)
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        dense_traj[0] = dense_traj[0].clamp(xmin, xmax)
        dense_traj[1] = dense_traj[1].clamp(ymin, ymax)

    return dense_traj.detach().cpu().numpy()


def waypoint(cx, cy, goal_x, goal_y, gp_step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= gp_step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist
    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax, buffer=None):
    if buffer is None:
        buffer = gp_step * 2

    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = gp_step * np.round(cx / gp_step)
    cy = gp_step * np.round(cy / gp_step)
    return cx, cy


def run_map(
    selected_map,
    execution_chunk,
    map_context,
    gp_context,
    fov_offsets,
    diffusion_model,
    run_index=0,
    batch_output_root=output_root,
):
    output_dir = batch_output_root / f"map_{selected_map}_chunk_{execution_chunk}_run_{run_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, mean, cov, xs, ys, X, _, xmin, xmax, ymin, ymax, gp_step = gp_context
    fov_dx, fov_dy = fov_offsets
    pts = map_context["pts"]
    true_map = map_context["true_map"]

    mu = mean.copy()
    P = cov.copy()
    R = 500
    sensor_seed = SENSORNOISE_SEED + selected_map * 1000 + run_index
    diffusion_seed = DIFFUSION_SAMPLE_SEED + selected_map * 1000 + run_index
    rng = np.random.default_rng(sensor_seed)
    torch.manual_seed(diffusion_seed)

    pos_history = []
    rmselist = []
    weighted_rmselist = []
    global_rmselist = []

    lateral_coverage = gp_step * 2
    samplestep = gp_step

    cx, cy = 20.0, 20.0
    grad_x, grad_y = 0.0, 0.0
    current_heading_velocity = np.zeros(2, dtype=np.float32)
    pos_history.append((cx, cy))

    initial_total_variance = np.sum(np.diag(P))
    print(
        f"Map {selected_map} chunk {execution_chunk} run {run_index}: "
        f"Initial total variance: {initial_total_variance:.4f}, "
        f"sensor_seed={sensor_seed}, diffusion_seed={diffusion_seed}"
    )

    spline_path = []
    spline_idx = 0
    executed_since_replan = 0

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, _ = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, gp_step=gp_step)
            prev_x, prev_y = cx, cy
            cx, cy = dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax)
            current_heading_velocity = np.array([cx - prev_x, cy - prev_y], dtype=np.float32)
            pos_history.append((cx, cy))
        else:
            if ts == 2 or executed_since_replan >= execution_chunk or spline_idx >= len(spline_path):
                current_mean = mu.reshape(X.shape)
                current_var = np.diag(P).reshape(X.shape)
                dense_traj = sample_diffusion_trajectory(
                    diffusion_model,
                    current_position=(cx, cy),
                    current_mean=current_mean,
                    current_var=current_var,
                    current_heading_velocity=current_heading_velocity,
                    grid_step=gp_step,
                    bounds=(xmin, xmax, ymin, ymax),
                )
                spline_path = dense_traj.T.tolist()
                spline_idx = 0
                executed_since_replan = 0
                print(
                    f"Map {selected_map} chunk {execution_chunk} run {run_index}: "
                    f"Replanning with {len(spline_path)} spline coordinates."
                )

            if spline_idx < len(spline_path):
                goal_x, goal_y = spline_path[spline_idx]
                grad_x, grad_y, waypoint_reached = waypoint(
                    cx,
                    cy,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    gp_step=gp_step,
                )

                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y = spline_path[spline_idx]
                        grad_x, grad_y, _ = waypoint(
                            cx,
                            cy,
                            goal_x=goal_x,
                            goal_y=goal_y,
                            gp_step=gp_step,
                        )
                    else:
                        grad_x, grad_y = 0.0, 0.0

                if spline_idx < len(spline_path):
                    x_next, y_next = spline_path[spline_idx]
                    padding = gp_step * 2
                    clamped_x = min(max(x_next, xmin + padding), xmax - padding)
                    clamped_y = min(max(y_next, ymin + padding), ymax - padding)
                    if clamped_x != x_next or clamped_y != y_next:
                        spline_path[spline_idx] = [clamped_x, clamped_y]

                prev_x, prev_y = cx, cy
                cx, cy = dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax)
                current_heading_velocity = np.array([cx - prev_x, cy - prev_y], dtype=np.float32)
                pos_history.append((cx, cy))
                executed_since_replan += 1
            else:
                pos_history.append((cx, cy))

        obs_idx, true_values = fov_indices_and_measurements(
            cx,
            cy,
            fov_dx,
            fov_dy,
            true_map,
            xmin,
            ymin,
            gp_step,
        )
        z_meas = true_values + rng.normal(0, np.sqrt(R), size=len(obs_idx))

        mu, P = kalman_update_indices(mu, P, obs_idx, z_meas, R)

        reconstruction_metrics = compute_reconstruction_rmse_from_true_map(mu, true_map)
        rmselist.append(reconstruction_metrics["occupied_rmse"])
        weighted_rmselist.append(reconstruction_metrics["weighted_rmse"])
        global_rmselist.append(reconstruction_metrics["global_rmse"])

    rmse_trace = np.column_stack([global_rmselist, rmselist, weighted_rmselist])
    np.savetxt(
        output_dir / f"map_{selected_map}_chunk_{execution_chunk}_run_{run_index:02d}_rmse_over_time.csv",
        rmse_trace,
        delimiter=",",
        header="global_rmse,occupied_rmse,weighted_rmse",
        comments="",
    )

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance

    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=gp_step,
        lateral_coverage=lateral_coverage,
        xmin=xmin,
        ymin=ymin,
    )
    coverage_efficiency = compute_coverage_efficiency(metrics["task_completion"], timealloted + 1)

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(rmselist)
    weighted_rmse_time = compute_rmse_time_metrics(weighted_rmselist)

    print(
        f"Map {selected_map} chunk {execution_chunk} run {run_index}: "
        f"Task Completion = {metrics['task_completion']:.4%}, "
        f"Global RMSE-AUC = {global_rmse_time['auc_rmse']:.4f}"
    )

    return {
        "map_id": selected_map,
        "execution_chunk": execution_chunk,
        "run_index": run_index,
        "sensor_seed": sensor_seed,
        "diffusion_seed": diffusion_seed,
        "gained_true_utility": metrics["gained_true_utility"],
        "total_true_utility": metrics["total_true_utility"],
        "task_completion": metrics["task_completion"],
        "coverage_efficiency": coverage_efficiency,
        "global_final_rmse": global_rmse_time["final_rmse"],
        "global_mean_rmse": global_rmse_time["mean_rmse"],
        "global_auc_rmse": global_rmse_time["auc_rmse"],
        "occupied_final_rmse": occupied_rmse_time["final_rmse"],
        "occupied_mean_rmse": occupied_rmse_time["mean_rmse"],
        "occupied_auc_rmse": occupied_rmse_time["auc_rmse"],
        "weighted_final_rmse": weighted_rmse_time["final_rmse"],
        "weighted_mean_rmse": weighted_rmse_time["mean_rmse"],
        "weighted_auc_rmse": weighted_rmse_time["auc_rmse"],
        "final_total_variance": final_variance,
        "variance_reduction": variance_delta,
    }


if __name__ == "__main__":
    args = parse_args()
    execution_chunks = parse_chunks(args.execution_chunks)
    chunk_runs = list(iter_chunk_runs(execution_chunks, args.runs_per_chunk))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gp_context = initialize_gp()
    _, _, _, _, xs, ys, _, _, xmin, _, ymin, _, gp_step = gp_context
    fov_offsets = build_fov_offsets(gp_step * 2, gp_step)
    map_context = load_map_context(args.map_id, xs, ys, xmin, ymin, gp_step)

    diffusion_model = load_diffusion_model(checkpoint_path=diffusion_path)
    metriclist = []

    for execution_chunk, run_index in chunk_runs:
        metriclist.append(
            run_map(
                args.map_id,
                execution_chunk,
                map_context,
                gp_context,
                fov_offsets,
                diffusion_model,
                run_index=run_index,
                batch_output_root=args.output_dir,
            )
        )

    print("\nSummary of results:")
    for metrics in metriclist:
        print(
            f"Map {metrics['map_id']} chunk {int(metrics['execution_chunk'])} "
            f"run {int(metrics['run_index'])}: "
            f"Task Completion = {metrics['task_completion']:.4%}, "
            f"Global RMSE-AUC = {metrics['global_auc_rmse']:.4f}, "
            f"Occupied RMSE-AUC = {metrics['occupied_auc_rmse']:.4f}, "
            f"Weighted RMSE-AUC = {metrics['weighted_auc_rmse']:.4f}"
        )

    summary_path = args.output_dir / args.summary_name
    write_metric_summary(metriclist, summary_path)

    if metriclist:
        avg_completion = np.mean([metrics["task_completion"] for metrics in metriclist])
        avg_global_auc = np.mean([metrics["global_auc_rmse"] for metrics in metriclist])
        avg_occupied_auc = np.mean([metrics["occupied_auc_rmse"] for metrics in metriclist])
        avg_weighted_auc = np.mean([metrics["weighted_auc_rmse"] for metrics in metriclist])
        print(f"\nAverage Task Completion = {avg_completion:.4%}")
        print(f"Average Global RMSE-AUC = {avg_global_auc:.4f}")
        print(f"Average Occupied RMSE-AUC = {avg_occupied_auc:.4f}")
        print(f"Average Weighted RMSE-AUC = {avg_weighted_auc:.4f}")
