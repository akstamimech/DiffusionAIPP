import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MAP_START = int(os.environ.get("MAP_START", 30))
MAP_END = int(os.environ.get("MAP_END", 35))
MAPS = range(MAP_START, MAP_END + 1)
MAPTYPE = "multiblob"


def run_singlemap(script_name, map_id, rmse_path):
    force_rerun = os.environ.get("FORCE_RERUN", "0") == "1"
    force_cmaes = (
        os.environ.get("FORCE_CMAES", "0") == "1"
        and script_name == "CMAES_classic_singlemap.py"
    )

    if rmse_path.exists() and not force_rerun and not force_cmaes:
        print(f"Reusing {rmse_path}")
        return

    env = os.environ.copy()
    env["SELECTED_MAP"] = str(map_id)
    env["MAPTYPE"] = MAPTYPE
    env["SKIP_VIZ"] = "1"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name)],
        cwd=str(SCRIPT_DIR),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path = SCRIPT_DIR / f"map_{map_id}_{Path(script_name).stem}.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed for map {map_id}. See {log_path}"
        )


def load_rmse(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data


def plot_comparison(map_id, cmaes_rmse, lawnmower_rmse, output_dir):
    labels = ["Global RMSE", "Occupied RMSE", "Weighted RMSE"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    for idx, ax in enumerate(axes):
        ax.plot(cmaes_rmse[:, idx], label="CMA-ES", linewidth=2)
        ax.plot(lawnmower_rmse[:, idx], label="Lawnmower", linewidth=2)
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Map {map_id} - RMSE over Time ({MAPTYPE})")
    fig.tight_layout()

    output_path = output_dir / f"map_{map_id}_lawnmower_vs_cmaes_rmse.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def summarize(map_id, cmaes_rmse, lawnmower_rmse):
    names = ["global", "occupied", "weighted"]
    rows = []
    for idx, name in enumerate(names):
        rows.append({
            "map": map_id,
            "metric": name,
            "cmaes_final": float(cmaes_rmse[-1, idx]),
            "lawnmower_final": float(lawnmower_rmse[-1, idx]),
            "cmaes_mean": float(np.mean(cmaes_rmse[:, idx])),
            "lawnmower_mean": float(np.mean(lawnmower_rmse[:, idx])),
        })
    return rows


def main():
    comparison_dir = SCRIPT_DIR / f"rmse_comparisons_{MAP_START}_{MAP_END}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for map_id in MAPS:
        print(f"Running map {map_id}: lawnmower")
        lawnmower_rmse_path = (
            SCRIPT_DIR / f"lawnmower_map_{map_id}_viz" / f"map_{map_id}_rmse_over_time.csv"
        )
        cmaes_rmse_path = (
            SCRIPT_DIR / f"{MAPTYPE}_map_{map_id}_viz" / f"map_{map_id}_rmse_over_time.csv"
        )

        run_singlemap("lawnmower_singlemap.py", map_id, lawnmower_rmse_path)

        print(f"Running map {map_id}: CMA-ES")
        run_singlemap("CMAES_classic_singlemap.py", map_id, cmaes_rmse_path)

        lawnmower_rmse = load_rmse(lawnmower_rmse_path)
        cmaes_rmse = load_rmse(cmaes_rmse_path)

        output_path = plot_comparison(map_id, cmaes_rmse, lawnmower_rmse, comparison_dir)
        print(f"Saved {output_path}")
        summary_rows.extend(summarize(map_id, cmaes_rmse, lawnmower_rmse))

    summary_path = comparison_dir / "rmse_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("map,metric,cmaes_final,lawnmower_final,cmaes_mean,lawnmower_mean\n")
        for row in summary_rows:
            f.write(
                f"{row['map']},{row['metric']},"
                f"{row['cmaes_final']},{row['lawnmower_final']},"
                f"{row['cmaes_mean']},{row['lawnmower_mean']}\n"
            )

    print(f"Saved summary {summary_path}")


if __name__ == "__main__":
    main()
