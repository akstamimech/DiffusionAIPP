import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
import cma

from scipy.interpolate import CubicSpline




step = 2.0
CMA_SEED = 42



def sampler(cx, cy, X, Y, P_history, samplestep):
    var_field = np.diag(P_history[-1]).reshape(X.shape)

    # gradient in physical coordinates
    gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
    grad_mag = np.hypot(gx, gy)

    
    mask = grad_mag >= 0.95 * grad_mag.max()
    rows, cols = np.where(mask)

    candidate_x = X[rows, cols]
    candidate_y = Y[rows, cols]

    distances = np.hypot(candidate_x - cx, candidate_y - cy)
    closest_idx = np.argmin(distances)

    target_x = candidate_x[closest_idx]
    target_y = candidate_y[closest_idx]

    dx = target_x - cx
    dy = target_y - cy

    norm = np.hypot(dx, dy)
    if norm > 1e-12:
        dx = dx / norm * samplestep
        dy = dy / norm * samplestep
    else:
        dx, dy = 0.0, 0.0

    return np.array([dx, dy]), (gx, gy), (target_x, target_y)

"""
messing around with utility function at the moment
"""


#importance_filter returns variance form for easy utility deduction!
def importance_filter(mu, P, beta, threshold = 0.7, eps=1e-12): 
    sigma = np.sqrt(np.diag(P))

    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    mu_norm = (mu - mu_min) / max(mu_max - mu_min, eps)
    sigma_norm = (sigma - sigma_min) / max(sigma_max - sigma_min, eps)
    importance = mu_norm + beta * sigma_norm
    importance_threshold = np.quantile(importance, threshold)

    importance_mask = importance >= importance_threshold

    if not np.any(importance_mask):
        importance_mask = np.ones_like(importance, dtype=bool)


    sigma2 = np.diag(P)

    sigma2_min, sigma2_max = np.min(sigma2), np.max(sigma2)
    sigma2_norm = (sigma2 - sigma2_min) / max(sigma2_max - sigma2_min, eps)

    utility = np.zeros_like(mu)
    utility[importance_mask] = sigma2_norm[importance_mask]

    return utility


def grid_measure(filtered_utility, xs, ys, margin=None):

    if margin is None:
        margin = step * 2   # match your dynamics buffer

    X, Y = np.meshgrid(xs, ys)

    nx = 8
    ny = 8

    # sample ONLY inside valid region
    x_positions = np.linspace(X.min() + margin, X.max() - margin, nx)
    y_positions = np.linspace(Y.min() + margin, Y.max() - margin, ny)

    selected_points = []
    for x in x_positions:
        xg = X[0, np.argmin(np.abs(X[0, :] - x))]
        for y in y_positions:
            yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
            selected_points.append((xg, yg))

    selected_points = list(dict.fromkeys(selected_points))

    util_values = []
    for xg, yg in selected_points:
        idx = np.where(
            np.isclose(X.ravel(), xg) &
            np.isclose(Y.ravel(), yg)
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        util_value = filtered_utility[idx]

        # shape preserved exactly
        util_values.append((util_value, (xg, yg)))

    return util_values


def next_best_waypoint(grid_util_values, curr_x, curr_y, alpha = 0.1): 
    scored = []
    for util, (x, y) in grid_util_values:
        dist = np.hypot(x - curr_x, y - curr_y)
        if dist == 0:
            dist = 1e-6
        score = util * np.exp(-alpha * dist)
        scored.append((score, (x, y)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
    
def flatten_waypoints(waypoints):
    return np.array([coord for pt in waypoints for coord in pt], dtype=float)

def unflatten_waypoints(z):
    return [(z[i], z[i + 1]) for i in range(0, len(z), 2)]


def clip_waypoints_continuous(waypoints, xs, ys, margin=None):
    if margin is None:
        margin = step * 2

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    clipped = []
    for x, y in waypoints:
        x = np.clip(x, xmin + margin, xmax - margin)
        y = np.clip(y, ymin + margin, ymax - margin)
        clipped.append((x, y))

    return clipped


def snap_waypoints_to_grid(waypoints, xs, ys):
    snapped = []
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    for x, y in waypoints:
        x_snap = xs[np.argmin(np.abs(xs - x))]
        y_snap = ys[np.argmin(np.abs(ys - y))]
        snapped.append((float(x_snap), float(y_snap)))

    return snapped


def future_observation_indices(spline_path, xs, ys, lateral_coverage, trajectory_stride=10, max_obs=64):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    nx = len(xs)
    selected_path = list(spline_path[::trajectory_stride])
    if spline_path and selected_path[-1] != spline_path[-1]:
        selected_path.append(spline_path[-1])

    obs_indices = []
    for px, py in selected_path:
        cx = xs[np.argmin(np.abs(xs - px))]
        cy = ys[np.argmin(np.abs(ys - py))]

        for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step):
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step):
                xi = int(np.argmin(np.abs(xs - x)))
                yi = int(np.argmin(np.abs(ys - y)))
                if np.isclose(xs[xi], x) and np.isclose(ys[yi], y):
                    obs_indices.append(yi * nx + xi)

    obs_indices = list(dict.fromkeys(obs_indices))
    if len(obs_indices) > max_obs:
        sample_idx = np.linspace(0, len(obs_indices) - 1, max_obs, dtype=int)
        obs_indices = [obs_indices[i] for i in sample_idx]

    return np.array(obs_indices, dtype=int)


def masked_expected_variance_reduction(P, mask, obs_indices, R):
    if len(obs_indices) == 0:
        return 0.0

    mask_indices = np.flatnonzero(mask)
    if len(mask_indices) == 0:
        mask_indices = np.arange(P.shape[0])

    S = P[np.ix_(obs_indices, obs_indices)] + R * np.eye(len(obs_indices))
    cross_cov = P[np.ix_(mask_indices, obs_indices)]

    try:
        solved = np.linalg.solve(S, cross_cov.T)
        reduction = np.sum(cross_cov.T * solved, axis=0)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(S) @ cross_cov.T
        reduction = np.sum(cross_cov.T * solved, axis=0)

    prior_diag = np.diag(P)[mask_indices]
    reduction = np.clip(reduction, 0.0, prior_diag)
    return float(np.sum(reduction))


def trajectory_objective(
    z,
    mu,
    P,
    xs,
    ys,
    start_x,
    start_y,
    beta,
    utility_threshold,
    R=None,
    lateral_coverage=None,
    predictive_variance=False,
    cached_utility=None,
    cached_importance_mask=None,
):
    raw_control_waypoints = unflatten_waypoints(z)
    control_waypoints = clip_waypoints_continuous(raw_control_waypoints, xs, ys)

    spline_path = build_spline_trajectory(
        start_x, start_y, control_waypoints, samples_per_segment=5
    )

    utility = cached_utility
    if utility is None:
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)

    total_score = 0.0
    total_distance = 0.0
    curr_x, curr_y = start_x, start_y
    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    for wx, wy in raw_control_waypoints:
        if wx < xmin + margin or wx > xmax - margin or wy < ymin + margin or wy > ymax - margin:
            total_score -= 1000.0

    for px, py in spline_path:
        if px < xmin + margin or px > xmax - margin or py < ymin + margin or py > ymax - margin:
            total_score -= 1000.0
        total_distance += np.hypot(px - curr_x, py - curr_y)
        curr_x, curr_y = px, py

    if predictive_variance:
        if R is None or lateral_coverage is None:
            raise ValueError("R and lateral_coverage are required for predictive variance scoring.")

        importance_mask = cached_importance_mask
        if importance_mask is None:
            importance_mask = utility > 0
        obs_indices = future_observation_indices(spline_path, xs, ys, lateral_coverage)
        variance_reduction = masked_expected_variance_reduction(P, importance_mask, obs_indices, R)
        total_score += variance_reduction
        total_score -= 0.01 * total_distance
        return -total_score

    X, Y = np.meshgrid(xs, ys)
    curr_x, curr_y = start_x, start_y
    for px, py in spline_path:
        idx = np.argmin((X.ravel() - px) ** 2 + (Y.ravel() - py) ** 2)
        total_score += utility[idx]
        total_score -= 0.01 * np.hypot(px - curr_x, py - curr_y)
        curr_x, curr_y = px, py

    return -total_score


def cma_es_refine_waypoints(
    initial_waypoints,
    mu,
    P,
    xs,
    ys,
    cx,
    cy,
    beta,
    utility_threshold,
    R=None,
    lateral_coverage=None,
    predictive_variance=False,
):
    x0 = flatten_waypoints(initial_waypoints)
    sigma0 = 4.0
    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    lower_bounds = []
    upper_bounds = []
    for _ in initial_waypoints:
        lower_bounds.extend([xmin + margin, ymin + margin])
        upper_bounds.extend([xmax - margin, ymax - margin])

    maxiter = 18 if predictive_variance else 40
    popsize = 8 if predictive_variance else 12
    cached_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    cached_importance_mask = cached_utility > 0

    es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": [lower_bounds, upper_bounds],
        "maxiter": maxiter,
        "popsize": popsize,
        "seed": CMA_SEED,
        "verb_disp": 0,
        "verb_log": 0,
    })

    while not es.stop():
        solutions = es.ask()
        values = [
            trajectory_objective(
                sol,
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
                predictive_variance=predictive_variance,
                cached_utility=cached_utility,
                cached_importance_mask=cached_importance_mask,
            )
            for sol in solutions
        ]
        es.tell(solutions, values)

    best = es.result.xbest
    best_waypoints = clip_waypoints_continuous(unflatten_waypoints(best), xs, ys, margin=margin)
    return snap_waypoints_to_grid(best_waypoints, xs, ys)




def build_spline_trajectory(cx, cy, control_waypoints, samples_per_segment=10):
    waypoints = np.array([(cx, cy)] + list(control_waypoints), dtype=float)

    if len(waypoints) < 2:
        return [tuple(waypoints[0])]

    deltas = np.diff(waypoints, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    t = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # Remove duplicate points that create zero-length segments
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    waypoints = waypoints[keep]
    t = t[keep]

    if len(t) < 2:
        return [tuple(waypoints[0])]

    cs_x = CubicSpline(t, waypoints[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, waypoints[:, 1], bc_type="natural")

    trajectory = []
    for i in range(len(t) - 1):
        t_segment = np.linspace(t[i], t[i + 1], samples_per_segment, endpoint=False)
        for ts in t_segment:
            trajectory.append((float(cs_x(ts)), float(cs_y(ts))))

    trajectory.append((float(waypoints[-1, 0]), float(waypoints[-1, 1])))
    return trajectory





def utility_function(mu, P, threshold, beta, eps=1e-12):
    sigma = np.sqrt(np.diag(P))
    beta = np.clip(beta, 0.0, 1.0)

    # Keep mean and uncertainty on comparable scales so beta controls the
    # tradeoff instead of whichever term happens to have the larger magnitude.
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)

    mu_norm = (mu - mu_min) / max(mu_max - mu_min, eps)
    sigma_norm = (sigma - sigma_min) / max(sigma_max - sigma_min, eps)

    utility = ((1 - beta) * mu_norm) + (beta * sigma_norm) - threshold
    return np.asarray(utility)

def kalman_update(mu, P, sensor, z_meas, R):
    v = z_meas - (sensor @ mu)

    S = sensor @ P @ sensor.T + R * np.eye(sensor.shape[0])
    K = P @ sensor.T @ np.linalg.inv(S)

    mu = mu + (K @ v).flatten()
    P = P - K @ sensor @ P

    return mu, P


def initialize_gp(sigma2=10.0, lengthscale=10.0, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0):
    kernel = ConstantKernel(
        sigma2, constant_value_bounds="fixed"
    ) * Matern(
        length_scale=lengthscale,
        length_scale_bounds="fixed",
        nu=1.5
    )

    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)

    X, Y = np.meshgrid(xs, ys) #X, Y are 2D arrays of shape (len(ys), len(xs)) and represent the grid of points in the 2D space
    X_test = np.column_stack([X.ravel(), Y.ravel()])

    # initialization
    mean, cov = gp.predict(X_test, return_cov=True)

    return gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step


def create_plots_and_gifs(path, mu_history, P_history, step_numbers, grad_history, pos_history, utility_history, sorted_util_values_list, X, Y, xs, ys, cx, cy, lateral_coverage, xmin, xmax, ymin, ymax, plot_utility=True, plot_grad=True, planned_path_history=None, control_waypoint_history=None):
    def overlay_planned_path(ax, path_points):
        if not path_points:
            return
        path_arr = np.asarray(path_points, dtype=float)
        if path_arr.ndim != 2 or path_arr.shape[1] != 2:
            return
        ax.plot(path_arr[:, 0], path_arr[:, 1], color="cyan", linewidth=1.5, alpha=0.9)

    def overlay_control_waypoints(ax, control_points):
        if not control_points:
            return
        control_arr = np.asarray(control_points, dtype=float)
        if control_arr.ndim != 2 or control_arr.shape[1] != 2:
            return
        ax.scatter(
            control_arr[:, 0],
            control_arr[:, 1],
            c="magenta",
            s=28,
            marker="x",
            linewidths=1.5,
            alpha=0.95,
        )

    def append_current_figure(writer):
        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        writer.append_data(frame[:, :, :3])

    utility_vmin = 0.0
    utility_vmax = max(float(np.max(u)) for u in utility_history) if utility_history else 1.0
    if utility_vmax <= utility_vmin:
        utility_vmax = utility_vmin + 1.0
    frame_stride = 1
    video_figsize = (5, 4)

    # variance plots
    P_before = np.diag(P_history[0]).reshape(X.shape)
    P_after = np.diag(P_history[-1]).reshape(X.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(
        P_before,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[0].set_title("Variance before update")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(
        P_after,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[1].set_title("Variance after update")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # mean plots
    mu_before = mu_history[0]
    mu_after = mu_history[-1]

    Z_before = mu_before.reshape(len(ys), len(xs))
    Z_after = mu_after.reshape(len(ys), len(xs))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(
        Z_before,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[0].set_title("Mean before update")

    im1 = ax[1].imshow(
        Z_after,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )

    x_min = cx - lateral_coverage
    y_min = cy - lateral_coverage

    rect = Rectangle(
        (x_min, y_min),
        2 * lateral_coverage,
        2 * lateral_coverage,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax[1].add_patch(rect)
    ax[1].plot(cx, cy, "wo", markersize=6)
    if planned_path_history:
        overlay_planned_path(ax[1], planned_path_history[-1])
    if control_waypoint_history:
        overlay_control_waypoints(ax[1], control_waypoint_history[-1])

    ax[1].set_title("Mean after update")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # utility plots
    if plot_utility:
        util_before = utility_history[0]
        util_after = utility_history[-1]

        U_before = util_before.reshape(len(ys), len(xs))
        U_after = util_after.reshape(len(ys), len(xs))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        im0 = ax[0].imshow(
            U_before,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=utility_vmin,
            vmax=utility_vmax,
        )
        ax[0].set_title("Masked Information Gain Before Update")
        plt.colorbar(im0, ax=ax[0], label="Masked information gain")

        im1 = ax[1].imshow(
            U_after,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=utility_vmin,
            vmax=utility_vmax,
        )

        x_min = cx - lateral_coverage
        y_min = cy - lateral_coverage

        rect = Rectangle(
            (x_min, y_min),
            2 * lateral_coverage,
            2 * lateral_coverage,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax[1].add_patch(rect)
        ax[1].plot(cx, cy, "wo", markersize=6)
        if planned_path_history:
            overlay_planned_path(ax[1], planned_path_history[-1])
        if control_waypoint_history:
            overlay_control_waypoints(ax[1], control_waypoint_history[-1])

        ax[1].set_title("Masked Information Gain After Update")
        plt.colorbar(im1, ax=ax[1], label="Masked information gain")

        plt.tight_layout()
        plt.show()

    vmin = min(m.min() for m in mu_history)
    vmax = max(m.max() for m in mu_history)
    mp4_path = os.path.join(path, "gp_mean_evolution.mp4")
    with imageio.get_writer(mp4_path, fps=4, codec='libx264') as writer:
        for i, (mu_snap, step_num) in enumerate(zip(mu_history, step_numbers)):
            if i % frame_stride != 0:
                continue

            mu_field = mu_snap.reshape(X.shape)

            plt.figure(figsize=video_figsize)
            plt.imshow(
                mu_field,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                aspect="equal",
                cmap="hot",
                vmin=vmin,
                vmax=vmax
            )

            px, py = pos_history[min(i, len(pos_history) - 1)]
            plt.plot(px, py, "wo", markersize=5)
            plt.gca().add_patch(
                Rectangle(
                    (px - lateral_coverage, py - lateral_coverage),
                    2 * lateral_coverage,
                    2 * lateral_coverage,
                    linewidth=2,
                    edgecolor="cyan",
                    facecolor="none"
                )
            )
            if planned_path_history and i < len(planned_path_history):
                overlay_planned_path(plt.gca(), planned_path_history[i])
            if control_waypoint_history and i < len(control_waypoint_history):
                overlay_control_waypoints(plt.gca(), control_waypoint_history[i])

            plt.colorbar(label="GP mean")
            plt.title(f"GP mean after {step_num} measurements")
            plt.xlabel("x")
            plt.ylabel("y")
            append_current_figure(writer)
            plt.close()

    print(f"MP4 saved to: {mp4_path}")

    # GIF generation for utility
    if plot_utility:
        print(f"Generating utility MP4 with {len(utility_history)} frames, sorted_util_values_list has {len(sorted_util_values_list)} entries")

        util_mp4_path = os.path.join(path, "gp_utility_evolution.mp4")
        with imageio.get_writer(util_mp4_path, fps=4, codec='libx264') as writer:
            for i, (util_snap, step_num) in enumerate(zip(utility_history, step_numbers)):
                if i % frame_stride != 0:
                    continue

                util_field = util_snap.reshape(X.shape)

                plt.figure(figsize=video_figsize)
                plt.imshow(
                    util_field,
                    origin="lower",
                    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    aspect="equal",
                    cmap="viridis",
                    vmin=utility_vmin,
                    vmax=utility_vmax
                )

                print(f"Frame {i}: len(sorted_util_values_list)={len(sorted_util_values_list[0])}")
                if i < len(sorted_util_values_list) and isinstance(sorted_util_values_list[i], list) and len(sorted_util_values_list[i]) > 0 and isinstance(sorted_util_values_list[i][0], tuple):
                    # print(f"Plotting {len(sorted_util_values_list[i])} points")
                    for util, (x, y) in sorted_util_values_list[i]:
                        plt.scatter(x, y, c='yellow', s=35, edgecolors='black', alpha=0.9)

                px, py = pos_history[min(i, len(pos_history) - 1)]
                plt.plot(px, py, "wo", markersize=5)
                plt.gca().add_patch(
                    Rectangle(
                        (px - lateral_coverage, py - lateral_coverage),
                        2 * lateral_coverage,
                        2 * lateral_coverage,
                        linewidth=2,
                        edgecolor="cyan",
                        facecolor="none"
                    )
                )
                if planned_path_history and i < len(planned_path_history):
                    overlay_planned_path(plt.gca(), planned_path_history[i])
                if control_waypoint_history and i < len(control_waypoint_history):
                    overlay_control_waypoints(plt.gca(), control_waypoint_history[i])

                plt.colorbar(label="Masked information gain")
                plt.title(f"Masked Information Gain after {step_num} measurements")
                plt.xlabel("x")
                plt.ylabel("y")
                append_current_figure(writer)
                plt.close()

        print(f"Utility MP4 saved to: {util_mp4_path}")

    varmin = min(np.diag(P_snap).min() for P_snap in P_history)
    varmax = max(np.diag(P_snap).max() for P_snap in P_history)

    var_mp4_path = os.path.join(path, "gp_variance_evolution.mp4")
    with imageio.get_writer(var_mp4_path, fps=4, codec='libx264') as writer:
        for i, (P_snap, step_num) in enumerate(zip(P_history, step_numbers)):
            if i % frame_stride != 0:
                continue

            var_field = np.diag(P_snap).reshape(X.shape)

            plt.figure(figsize=video_figsize)
            plt.imshow(
                var_field,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                aspect="equal",
                cmap="viridis",
                vmin=varmin,
                vmax=varmax
            )

            px, py = pos_history[min(i, len(pos_history) - 1)]
            plt.plot(px, py, "wo", markersize=5)

            plt.colorbar(label="Variance")
            plt.title(f"Variance after {step_num} measurements")
            plt.xlabel("x")
            plt.ylabel("y")
            append_current_figure(writer)
            plt.close()

    print(f"Variance MP4 saved to: {var_mp4_path}")


def grid_search(X, Y, cx, cy, utility_history, margin=None):
    if margin is None:
        margin = step * 2   # match your dynamics buffer

    nx = 8
    ny = 8

    # sample ONLY inside valid region
    x_positions = np.linspace(X.min() + margin, X.max() - margin, nx)
    y_positions = np.linspace(Y.min() + margin, Y.max() - margin, ny)

    selected_points = []
    for x in x_positions:
        xg = X[0, np.argmin(np.abs(X[0, :] - x))]
        for y in y_positions:
            yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
            selected_points.append((xg, yg))

    # remove duplicates (important when snapping)
    selected_points = list(dict.fromkeys(selected_points))

    util_values = []
    for xg, yg in selected_points:
        idx = np.where(
            np.isclose(X.ravel(), xg) &
            np.isclose(Y.ravel(), yg)
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        util_value = utility_history[-1][idx]

        # shape preserved exactly
        util_values.append((util_value, (xg, yg)))

    util_values.sort(key=lambda x: x[0], reverse=True)
    return util_values

# #grid_search is set to greedy step, searching for next best.
# def grid_search(X, Y, cx, cy, utility_history, grid_edge = step*2):
#     # Select 20 equidistant points over the grid: 5 in x, 4 in y
#     nx = 5
#     ny = 4
#     x_positions = np.linspace(X.min() - grid_edge, X.max() + grid_edge, nx)
#     y_positions = np.linspace(Y.min() - grid_edge, Y.max() + grid_edge, ny)
    
#     selected_points = []
#     for x in x_positions:
#         # Find closest grid x
#         xg = X[0, np.argmin(np.abs(X[0, :] - x))]
#         for y in y_positions:
#             # Find closest grid y
#             yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
#             selected_points.append((xg, yg))
    
#     util_values = []
#     for xg, yg in selected_points:
#         idx = np.where(
#             np.isclose(X.ravel(), xg) &
#             np.isclose(Y.ravel(), yg)
#         )[0]

#         if len(idx) == 0:
#             continue

#         idx = idx[0]
#         util_value = utility_history[-1][idx]
#         util_values.append((util_value, (xg, yg)))
#         # print(f"Utility at ({xg}, {yg}): {util_value:.4f}")
    
#     util_values.sort(key=lambda x: x[0], reverse=True)
#     return util_values  # Return sorted list of utility values and their corresponding coordinates




if __name__ == "__main__":
    path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    data = np.loadtxt(rf"{path}\map_1_blob_grid_counts.csv", delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    # discretized points
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
    R = 1e-6

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)
    utility_threshold = 4.0

    initial_utility = utility_function(mu, P, utility_threshold)
    utility_history.append(initial_utility)
    
    # sorted_util_values_list.append(sorted_util_values)
    

    save_every = 5
    lateral_coverage = step * 2
    samplestep = 4.0
    timealloted = 150

   

    # initial stating position and gradient
    cx, cy = 50.0, 50.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    sorted_util_values_list.append(grid_search(X, Y, cx, cy, utility_history))

    # store initial gradient field
    initial_var_field = np.diag(P_history[-1]).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))

    #from here we consider movement and measurement updates

    for ts in range(0, timealloted):
        cx = np.clip(cx + grad_x + np.ceil(np.random.uniform(-step, step)), xmin, xmax)
        cy = np.clip(cy + grad_y + np.ceil(np.random.uniform(-step, step)), ymin, ymax)
        cx = step * np.round(cx / step)
        cy = step * np.round(cy / step)
        pos_history.append((cx, cy))

        print(cx, cy)
        step_numbers.append(ts + 1)

        fov = [
            (x, y)
            for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
        ]

        sensor = np.zeros((len(fov), N))

        for i, (x_meas, y_meas) in enumerate(fov):
            idx = np.where(
                np.isclose(X_test[:, 0], x_meas) &
                np.isclose(X_test[:, 1], y_meas)
            )[0]

            if len(idx) == 0:
                continue

            idx = idx[0]
            sensor[i, idx] = 1.0

        measurement_list = []
        for x_fov, y_fov in fov:
            idx = np.where(
                np.isclose(pts[:, 0], x_fov) &
                np.isclose(pts[:, 1], y_fov)
            )[0]

            if idx.size > 0:
                measurement_list.append(pts[idx[0], 2])
            else:
                measurement_list.append(0.0)

        z_meas = np.array(measurement_list)

        mu, P = kalman_update(mu, P, sensor, z_meas, R)

        

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility = utility_function(mu, P, utility_threshold)
        utility_history.append(utility)
        sorted_util_value = grid_search(X, Y, cx, cy, utility_history)
        sorted_util_values_list.append(sorted_util_value)
        

        grad, grad_field, target = sampler(cx, cy, X, Y, P_history, samplestep)
        grad_x, grad_y = grad
        target_x, target_y = target

        grad_history.append(grad_field)

        print("step vector:", grad_x, grad_y)
        print("target:", target_x, target_y)
        print(sorted_util_value)

    create_plots_and_gifs(path, mu_history, P_history, step_numbers, grad_history, pos_history, utility_history, sorted_util_values_list, X, Y, xs, ys, cx, cy, lateral_coverage, xmin, xmax, ymin, ymax)
