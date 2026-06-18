# 3D Dataset Rank Plotter Design

## Purpose

Create a small interactive Python plotter for
`CMAES_beamsearch_dataset_3d.pt`. The user edits configuration constants at
the top of the script rather than supplying command-line arguments.

## Selection

- `SELECTED_MAP` filters `map_id`.
- `SELECTED_RANK` filters `timestep`, which is the collector's planning rank.
- Matching rows are sorted by `RMSE_correction` from largest to smallest.
- `TOP_N` limits how many matching action trajectories are displayed.

Each matching row is an independent candidate action from the selected
planning condition. The dataset does not contain enough child-beam lineage
metadata to reconstruct a guaranteed multi-rank flight history.

## Figure

The script creates one interactive Matplotlib figure with:

- A 3D subplot showing the saved 41-point trajectory, eight control
  waypoints, and current 3D position.
- A top-down XY subplot showing the same candidates.
- A shared color scale based on `RMSE_correction`.
- Labels identifying each candidate's sorted position and RMSE correction.

The figure is displayed with `plt.show()` and is not saved automatically.

## Validation And Errors

The loader verifies required dataset keys and expected coordinate shapes.
It raises a clear error when the selected map/rank has no samples. Plotting
uses CPU tensors converted to NumPy arrays and does not modify the dataset.

## Testing

Run the script with an existing map/rank and verify that both views appear,
the current positions and control waypoints align with their trajectories,
and the displayed ordering matches descending `RMSE_correction`.
