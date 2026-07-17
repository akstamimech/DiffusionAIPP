import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Set which map to visualize here ---
MAPTYPE = os.environ.get("MAPTYPE", "multiblob")  # "NAIP", "multiblob", "halffield", "blob",
                                              # "safecast_poly_normalized_multiblob", or ""
                                              # for the untyped default grid_counts.csv
selected_map = int(os.environ.get("SELECTED_MAP", 20))

CSV_DIR = Path(__file__).resolve().parent / "csv"


def load_map(selected_map, maptype):
    suffix = f"_{maptype}" if maptype else ""
    csv_path = CSV_DIR / f"map_{selected_map}{suffix}_normalized_grid_counts.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No grid_counts CSV for map {selected_map} maptype={maptype!r} at {csv_path}"
        )

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    grid = data[:, 2].reshape(len(ys), len(xs))
    return xs, ys, grid


if __name__ == "__main__":
    xs, ys, grid = load_map(selected_map, MAPTYPE)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        grid,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im, label="count")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Map {selected_map} ({MAPTYPE or 'default'})")
    plt.tight_layout()
    plt.show()
