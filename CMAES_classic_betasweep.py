import numpy as np
import matplotlib
from pathlib import Path
import torch

matplotlib.use("Agg")

from gaussianprocesstraining import (
    importance_filter,
    grid_measure,
    next_best_waypoint,
    cma_es_refine_waypoints,
    build_spline_trajectory,
    create_plots_and_gifs,
    kalman_update,
    initialize_gp,
)
from evalmetrics import compute_task_completion, compute_reconstruction_rmse, compute_rmse_time_metrics


step = 2.0
timealloted = 60
beta_values = np.round(np.arange(0.1, 1.0, 0.1), 1)
alpha = 0.02
utility_threshold = 0.5
planning_horizon = 8
initial_map = 1
mapcount = 31
samples_per_segment = 5
execution_chunk = 20
save_visualizations = False
target_trajectory_len = planning_horizon * samples_per_segment + 1
SENSORNOISE_SEED = 123
MAPTYPE = "multiblob"
script_dir = Path(__file__).resolve().parent
temp_chunk_dir = script_dir / "CMAES_classic_betasweep_temp_chunks"
final_dataset_path = script_dir / "CMAES_classic_betasweep_dataset.pt"


def resample_trajectory(path, target_len):
    path = np.asarray(path, dtype=np.float32)

    if len(path) == target_len:
        return path

    if len(path) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)

    if len(path) == 1:
        return np.repeat(path, target_len, axis=0)

    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, target_len)
    x = np.interp(target_t, source_t, path[:, 0])
    y = np.interp(target_t, source_t, path[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def make_empty_dataset():
    return {
        "trajectories": [],
        "control_waypoints": [],
        "current_position": [],
        "current_mean": [],
        "current_var": [],
        "map_id": [],
        "beta": [],
        "timestep": [],
    }


def tensorize_dataset(dataset):
    return {
        "trajectories": torch.tensor(np.asarray(dataset["trajectories"], dtype=np.float32)).permute(0, 2, 1),
        "control_waypoints": torch.tensor(np.asarray(dataset["control_waypoints"], dtype=np.float32)),
        "current_position": torch.tensor(np.asarray(dataset["current_position"], dtype=np.float32)),
        "current_mean": torch.tensor(np.asarray(dataset["current_mean"], dtype=np.float32)),
        "current_var": torch.tensor(np.asarray(dataset["current_var"], dtype=np.float32)),
        "map_id": torch.tensor(np.asarray(dataset["map_id"], dtype=np.int64)),
        "beta": torch.tensor(np.asarray(dataset["beta"], dtype=np.float32)),
        "timestep": torch.tensor(np.asarray(dataset["timestep"], dtype=np.int64)),
    }


def save_beta_chunk(selected_map, beta_value, dataset):
    temp_chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = temp_chunk_dir / f"map_{selected_map:03d}_beta_{beta_value:.1f}.pt"
    torch.save(tensorize_dataset(dataset), chunk_path)
    print(f"Saved beta chunk to {chunk_path} with {len(dataset['trajectories'])} samples")
    return chunk_path


def delete_chunk(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def consolidate_chunks(chunk_paths, output_path):
    payloads = [torch.load(path, map_location="cpu") for path in chunk_paths]
    if not payloads:
        print("No selected chunks found; final dataset was not written.")
        return

    final_payload = {}
    for key in payloads[0]:
        final_payload[key] = torch.cat([payload[key] for payload in payloads], dim=0)

    torch.save(final_payload, output_path)
    print(f"Saved consolidated dataset to {output_path} with {len(final_payload['map_id'])} samples")


def cleanup_temp_chunks():
    temp_chunk_dir.mkdir(parents=True, exist_ok=True)
    for path in temp_chunk_dir.glob("map_*_beta_*.pt"):
        path.unlink()


def record_replan_sample(dataset, selected_map, beta_value, ts, cx, cy, mu, P, spline_path, control_waypoints, map_shape):
    dataset["trajectories"].append(resample_trajectory(spline_path, target_trajectory_len))
    dataset["control_waypoints"].append(np.asarray(control_waypoints, dtype=np.float32))
    dataset["current_position"].append(np.asarray([cx, cy], dtype=np.float32))
    dataset["current_mean"].append(mu.reshape(map_shape).astype(np.float32))
    dataset["current_var"].append(np.diag(P).reshape(map_shape).astype(np.float32))
    dataset["map_id"].append(selected_map)
    dataset["beta"].append(beta_value)
    dataset["timestep"].append(ts)


def real_receding_horizon_planner(
    cx,
    cy,
    mu,
    P,
    xs,
    ys,
    utility_threshold,
    beta,
    planning_horizon,
    alpha=0.1,
    R=None,
    lateral_coverage=None,
):
    flight_plan = []
    filtered_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_at_gridpoints = grid_measure(filtered_utility, xs, ys)
    curr_x, curr_y = cx, cy

    for _ in range(planning_horizon):
        best_waypoint = next_best_waypoint(utility_at_gridpoints, curr_x, curr_y, alpha=alpha)
        flight_plan.append(best_waypoint)
        curr_x, curr_y = best_waypoint
        utility_at_gridpoints = [
            (util, point) for util, point in utility_at_gridpoints if point != best_waypoint
        ]

    return cma_es_refine_waypoints(
        flight_plan,
        mu,
        P,
        xs,
        ys,
        cx,
        cy,
        beta,
        utility_threshold,
        R=R,
        lateral_coverage=lateral_coverage,
        predictive_variance=True,
    )


def waypoint(cx, cy, goal_x, goal_y, step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist
    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=None):
    if buffer is None:
        buffer = step * 2

    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def run_map(selected_map, beta_value):
    beta_dataset = make_empty_dataset()
    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    data = np.loadtxt(rf"{csv_path}/map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)
    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    mu = mean.copy()
    P = cov.copy()
    R = 4

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []
    planned_path_history = []
    control_waypoint_history = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)

    initial_utility = importance_filter(mu, P, beta_value, threshold=utility_threshold)
    utility_history.append(initial_utility.copy())
    planned_path_history.append([])
    control_waypoint_history.append([])

    lateral_coverage = step * 2
    samplestep = step

    cx, cy = 20.0, 20.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    initial_var_field = np.diag(P).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))
    sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

    initial_total_variance = np.sum(np.diag(P))
    print(f"Map {selected_map}, beta {beta_value:.1f}: Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility
    control_waypoints = []
    spline_path = []
    spline_idx = 0
    executed_since_replan = 0
    global_rmselist = []
    occupied_rmselist = []
    weighted_rmselist = []

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 2 or executed_since_replan >= execution_chunk or spline_idx >= len(spline_path):
                control_waypoints = real_receding_horizon_planner(
                    cx,
                    cy,
                    mu,
                    P,
                    xs,
                    ys,
                    utility_threshold,
                    beta_value,
                    planning_horizon,
                    alpha=alpha,
                    R=R,
                    lateral_coverage=lateral_coverage,
                )
                spline_path = build_spline_trajectory(
                    cx, cy, control_waypoints, samples_per_segment=samples_per_segment
                )
                record_replan_sample(
                    beta_dataset,
                    selected_map,
                    beta_value,
                    ts,
                    cx,
                    cy,
                    mu,
                    P,
                    spline_path,
                    control_waypoints,
                    X.shape,
                )
                spline_idx = 0
                executed_since_replan = 0
                print(
                    f"Map {selected_map}: Replanning with {len(control_waypoints)} control waypoints "
                    f"and {len(spline_path)} spline coordinates at beta {beta_value:.1f}."
                )

            if spline_idx < len(spline_path):
                goal_x, goal_y = spline_path[spline_idx]
                grad_x, grad_y, waypoint_reached = waypoint(
                    cx, cy, goal_x=goal_x, goal_y=goal_y, step=step
                )

                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y = spline_path[spline_idx]
                        grad_x, grad_y, _ = waypoint(
                            cx, cy, goal_x=goal_x, goal_y=goal_y, step=step
                        )
                    else:
                        grad_x, grad_y = 0.0, 0.0

                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
                executed_since_replan += 1
                planned_path_history.append(list(spline_path[spline_idx:]))
                control_waypoint_history.append(list(control_waypoints))
            else:
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
                planned_path_history.append([])
                control_waypoint_history.append([])

        fov = [
            (x, y)
            for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
        ]

        sensor = np.zeros((len(fov), N))

        for i, (x_meas, y_meas) in enumerate(fov):
            idx = np.where(
                np.isclose(X_test[:, 0], x_meas) & np.isclose(X_test[:, 1], y_meas)
            )[0]

            if len(idx) == 0:
                continue

            idx = idx[0]
            sensor[i, idx] = 1.0

        measurement_list = []
        for x_fov, y_fov in fov:
            idx = np.where(
                np.isclose(pts[:, 0], x_fov) & np.isclose(pts[:, 1], y_fov)
            )[0]

            if idx.size > 0:
                measurement_list.append(pts[idx[0], 2])
            else:
                measurement_list.append(0.0)

        z_meas = np.array(measurement_list) + rng.normal(0, np.sqrt(R), size=len(fov))

        mu, P = kalman_update(mu, P, sensor, z_meas, R)
        utility = importance_filter(mu, P, beta_value, threshold=utility_threshold)

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility_history.append(utility.copy())
        sorted_util_values_list.append(grid_measure(utility, xs, ys))

        var_field = np.diag(P).reshape(X.shape)
        gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
        grad_history.append((gx, gy))

        reconstruction_metrics = compute_reconstruction_rmse(
            mu=mu,
            pts=pts,
            xs=xs,
            ys=ys,
            step=step,
            xmin=xmin,
            ymin=ymin,
        )
        global_rmselist.append(reconstruction_metrics["global_rmse"])
        occupied_rmselist.append(reconstruction_metrics["occupied_rmse"])
        weighted_rmselist.append(reconstruction_metrics["weighted_rmse"])

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance
    print(f"Map {selected_map}, beta {beta_value:.1f}: Final total variance: {final_variance:.4f}")
    print(f"Map {selected_map}, beta {beta_value:.1f}: Variance reduction: {variance_delta:.4f}")

    if save_visualizations:
        output_dir = (
            r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts"
            rf"\classic_map_{selected_map}_beta_{beta_value:.1f}_viz"
        )
        create_plots_and_gifs(
            output_dir,
            mu_history,
            P_history,
            step_numbers,
            grad_history,
            pos_history,
            utility_history,
            sorted_util_values_list,
            X,
            Y,
            xs,
            ys,
            cx,
            cy,
            lateral_coverage,
            xmin,
            xmax,
            ymin,
            ymax,
            plot_utility=True,
            plot_grad=False,
            planned_path_history=planned_path_history,
            control_waypoint_history=control_waypoint_history,
        )

    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        lateral_coverage=lateral_coverage,
        xmin=xmin,
        ymin=ymin,
    )

    print(
        f"Map {selected_map}, beta {beta_value:.1f}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
        f"Total Utility = {metrics['total_true_utility']:.4f}, "
        f"Task Completion = {metrics['task_completion']:.4%}"
    )

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(occupied_rmselist)
    weighted_rmse_time = compute_rmse_time_metrics(weighted_rmselist)
    print(
        f"Map {selected_map}, beta {beta_value:.1f}: Global RMSE AUC = "
        f"{global_rmse_time['auc_rmse']:.4f}, Mean = {global_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}, beta {beta_value:.1f}: Occupied RMSE AUC = "
        f"{occupied_rmse_time['auc_rmse']:.4f}, Mean = {occupied_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}, beta {beta_value:.1f}: Weighted RMSE AUC = "
        f"{weighted_rmse_time['auc_rmse']:.4f}, Mean = {weighted_rmse_time['mean_rmse']:.4f}"
    )
    chunk_path = save_beta_chunk(selected_map, beta_value, beta_dataset)

    return (
        selected_map,
        beta_value,
        metrics["gained_true_utility"],
        metrics["total_true_utility"],
        metrics["task_completion"],
        global_rmse_time["auc_rmse"],
        occupied_rmse_time["auc_rmse"],
        weighted_rmse_time["auc_rmse"],
        chunk_path,
    )


if __name__ == "__main__":
    metriclist = []
    selected_chunk_paths = []

    cleanup_temp_chunks()
    for selected_map in range(initial_map, initial_map + mapcount):
        map_results = []
        for beta_value in beta_values:
            result = run_map(selected_map, float(beta_value))
            metriclist.append(result)
            map_results.append(result)

        best_result = min(map_results, key=lambda item: item[5])
        selected_chunk_paths.append(best_result[8])
        print(
            f"Map {selected_map}: keeping beta {best_result[1]:.1f} chunk "
            f"with Global RMSE AUC = {best_result[5]:.4f}"
        )

        for result in map_results:
            if result[8] != best_result[8]:
                delete_chunk(result[8])

    print("\nSummary of results:")
    for (
        map_id,
        beta_value,
        gained_utility,
        total_utility,
        completion,
        global_auc,
        occupied_auc,
        weighted_auc,
        _,
    ) in metriclist:
        print(
            f"Map {map_id}, beta {beta_value:.1f}: Gained Utility = {gained_utility:.4f}, "
            f"Total Utility = {total_utility:.4f}, Task Completion = {completion:.4%}, "
            f"Global RMSE AUC = {global_auc:.4f}, Occupied RMSE AUC = {occupied_auc:.4f}, "
            f"Weighted RMSE AUC = {weighted_auc:.4f}"
        )

    print("\nAverage global RMSE AUC by beta:")
    for beta_value in beta_values:
        global_aucs = [
            global_auc
            for _, result_beta, _, _, _, global_auc, _, _, _ in metriclist
            if np.isclose(result_beta, beta_value)
        ]
        avg_global_auc = np.mean(global_aucs)
        print(f"Beta {beta_value:.1f}: Average Global RMSE AUC = {avg_global_auc:.4f}")

    print("\nSelected best beta by map:")
    for map_id in range(initial_map, initial_map + mapcount):
        map_results = [result for result in metriclist if result[0] == map_id]
        best_result = min(map_results, key=lambda item: item[5])
        print(f"Map {map_id}: beta {best_result[1]:.1f}, Global RMSE AUC = {best_result[5]:.4f}")

    consolidate_chunks(selected_chunk_paths, final_dataset_path)

    for path in selected_chunk_paths:
        delete_chunk(path)
