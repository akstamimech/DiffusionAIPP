# 3D Dataset Rank Plotter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an editable, interactive Python script that selects a map and planning rank from the saved 3D CMAES dataset and plots the highest-RMSE-correction candidate trajectories.

**Architecture:** Keep loading, validation, selection, and plotting in small functions inside one script. A focused unittest module will construct synthetic tensor dictionaries to verify selection and validation without opening a GUI; a final smoke test will load the real dataset with Matplotlib's noninteractive backend.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, unittest

---

### Task 1: Dataset Selection

**Files:**
- Create: `plot_3d_dataset_rank.py`
- Create: `tests/test_plot_3d_dataset_rank.py`

- [ ] **Step 1: Write the failing selection tests**

Create tests that import `select_samples`, build a synthetic dataset, and assert that it filters by `map_id` and `timestep`, sorts descending by `RMSE_correction`, and respects `top_n`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_plot_3d_dataset_rank -v`

Expected: import failure because `plot_3d_dataset_rank.py` does not exist.

- [ ] **Step 3: Implement loading, validation, and selection**

Add editable constants, `load_dataset`, `validate_dataset`, and `select_samples`. Required keys are `trajectories`, `control_waypoints`, `current_position`, `map_id`, `timestep`, and `RMSE_correction`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_plot_3d_dataset_rank -v`

Expected: all selection and validation tests pass.

### Task 2: Interactive Plot

**Files:**
- Modify: `plot_3d_dataset_rank.py`
- Modify: `tests/test_plot_3d_dataset_rank.py`

- [ ] **Step 1: Write a failing figure-construction test**

Test that `build_figure` returns a figure containing one 3D axis and one top-down axis for a valid selected dataset.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_plot_3d_dataset_rank -v`

Expected: failure because `build_figure` is not defined.

- [ ] **Step 3: Implement plotting**

Plot each saved trajectory as a colored line, its control waypoints as markers, and its current position as a start marker in both views. Add a shared `RMSE_correction` colorbar, concise labels, equal XY limits, and call `plt.show()` from `main`.

- [ ] **Step 4: Run all tests**

Run: `python -m unittest tests.test_plot_3d_dataset_rank -v`

Expected: all tests pass.

### Task 3: Real Dataset Smoke Test

**Files:**
- Verify: `plot_3d_dataset_rank.py`

- [ ] **Step 1: Compile the script**

Run: `python -m py_compile plot_3d_dataset_rank.py tests/test_plot_3d_dataset_rank.py`

Expected: exit code 0.

- [ ] **Step 2: Build a figure from the real dataset without opening a window**

Load `CMAES_beamsearch_dataset_3d.pt`, select map 1/rank 0, build the figure under the `Agg` backend, and close it.

Expected: selected candidates and both axes are produced without errors.

- [ ] **Step 3: Check the final diff**

Run: `git diff --check -- plot_3d_dataset_rank.py tests/test_plot_3d_dataset_rank.py`

Expected: exit code 0.
