# Project handoff for a new agent (no prior context assumed)

Written 2026-10-08 by the previous agent (Claude) so that a different agent service on a different
machine can continue this project. Everything below was verified against the files on disk on that
date unless a line says "unverified" or "estimate". Treat every "current default" in any document in
this repo, including this one, as provisional until you re-check it (see section 9, trap 2).

---

## 0. Read this first

Suggested reading order for a new agent:

1. This file (all of it, especially sections 2, 3 and 9).
2. `CARRYOVER_PROJECT_SCOPE_OVERVIEW_2026_09_30.md` (orientation written 2026-09-30; some "current state"
   sentences in it are now stale, see section 2 here for what changed on 2026-10-01).
3. `CARRYOVER_NAIP_NORMALIZATION_BUG_AND_MULTIMODAL_INVESTIGATION.md` (the normalization-at-import bug,
   execution-chunk mismatch, multimodal investigation).
4. The other `CARRYOVER_*.md` files only if you need history (index in section 5.8).
5. `README.md` (short public-facing overview of the repo).

Before touching anything, run the settings audit in section 9 (trap 2). This project has flipped between
two map configurations (synthetic GRF and real NAIP) at least six times and every flip left something
inconsistent.

---

## 1. The project in brief

**Domain.** Adaptive informative path planning (AIPP) for an aerial robot (UAV). The UAV flies over a
100 m x 100 m area to map an unknown scalar field (real NAIP-derived NDVI vegetation-health tiles, or
synthetic Gaussian random fields). It keeps a Gaussian Process (GP) belief (posterior mean and variance
maps), updates it with a Kalman-style measurement update as it flies, and must choose where to look next
to reduce uncertainty fastest, especially in the "important" regions (cells whose value crosses a
threshold). It can change altitude (10 to 40 m): high altitude sees a larger footprint at coarser
resolution and higher noise, low altitude sees a small, sharp, low-noise footprint.

**Expert.** An online planner after Popovic et al. (2020): a coarse 3D lattice grid search proposes
waypoints, then CMA-ES refines 8 continuous 3D waypoints to maximize expected masked variance reduction
(divided by path length to the power `COST_EXPONENT`). It is good but slow (10 to 60 s per replan in the
original NAIP-era measurements, CPU only).

**The research idea (what the paper is about).** Train a diffusion policy by imitation on expert
trajectories so that:
1. Online planning is replaced by fast policy inference (amortization).
2. The policy learns the multimodal action distribution of the expert: for the same belief state there
   can be several near-equivalent informative trajectories. A deterministic regressor trained with MSE
   averages them ("mode averaging") into a trajectory that matches none of them; a diffusion model can
   sample the individual modes.

The policy conditions on the GP belief (mean map, variance map, a one-hot map of the agent position, total
remaining variance) and outputs 8 ordered 3D control waypoints, which are interpolated by a cubic spline
into a smooth continuous trajectory. It is re-queried at every replanning event (receding horizon).

**Planners compared throughout** (they all share the same simulator, belief update and metrics):

| Planner | Script | Role |
|---|---|---|
| CMA-ES (live) | `CMAES_classic_singlemap.py` | the expert, run online (slow) |
| Diffusion policy | `Diffusionplanner_singlemap_headingremoved.py` | the proposed method |
| ImitateTrans | `ImitateTrans_singlemap_headingremoved.py` | deterministic BC baseline, same network, MSE loss |
| Lawnmower | `lawnmower_singlemap.py` | non-adaptive coverage baseline |
| Greedy | `realgreedy_singlemap.py`, `greedygradient_singlemap.py` | myopic baselines |

**Paper.** A conference-style RA-L (IEEE Robotics and Automation Letters) submission, written in the IPB
(Bonn) LaTeX template (`ieeeconf`, "Our Approach" section etc.). The LaTeX source is NOT in this folder
(the user keeps it elsewhere, probably Overleaf); ask the user for it. Status of naming:

* Old working name: "DIPPer". It clashes with DiPPeR (Liu et al. 2023) and DiPPeST, both cited in the
  related work, so it should be replaced.
* Leading candidate: **MODAL** used as a plain name (not a letter-by-letter acronym), title
  "MODAL: Learning Multimodal Action Distributions with Diffusion for Aerial Adaptive Informative Path
  Planning". The user has not made a final decision; confirm before editing the draft. Other candidates
  discussed: MAD-AIPP (strict acronym), AMPED, MIMIC, MIDAS.

Contribution paragraph and claims in the user's latest wording (still says "DIPPer"; to be renamed):

> The main contribution of this paper is a diffusion-based policy for amortized informative path
> planning that replaces computationally expensive online planning with fast policy inference. By
> modelling multimodal action distributions of expert trajectories, the policy can exploit
> demonstrations containing multiple simultaneously valid actions without collapsing them through mode
> averaging. Conditioned on the Gaussian Process belief over the environment and the robot state, the
> policy directly generates smooth continuous 3D waypoint trajectories for informative exploration.
>
> (i) DIPPer approaches the information gain of the online CMA-ES expert at a fraction of its replanning
> cost (measured as occupied-region variance reduction over time and wall-clock replanning time per
> step). (ii) DIPPer outperforms a deterministic behavioral cloning policy of identical architecture in
> occupied-region variance and reconstruction error at a fixed time budget. (iii) Under a multimodal
> expert action distribution, DIPPer samples the individual expert modes, whereas the deterministic
> policy predicts trajectories between them. (iv) These results hold when the policy is executed on a
> real UAV.

Sections already drafted in the paper (text exists only in the user's LaTeX / chat history): Introduction,
Related Work (three subsections), Problem Formulation, Expert Data Generation (with Algorithm 1), DIPPer
architecture (with figure), Closed-Loop Execution (with Algorithm 2). The experiments section is a
commented-out skeleton. **Algorithm 1 in the paper is now out of date** relative to the code (it still
describes a diversity penalty); the up-to-date procedure is in Appendix B.

User's definition of "multimodal" (use this term consistently): the expert's action distribution is
multimodal when, for the same belief state, several distinct trajectories achieve near-equivalent
information gain. It is NOT the observations that are multimodal.

---

## 2. Current state (snapshot 2026-10-08)

### 2.1 Git

* Remote: `origin` = `https://github.com/akstamimech/DiffusionAIPP.git`. Default branch `main`.
* Latest commits on `main` (newest first): `6827312` "randomized mode continuation" (2026-10-01, also
  contains the rank-splitting rewrite of the collector), `8b55417` "removed diversity penalty from
  collector...", `85df899` "added jaccard distance to collection, flipped to NAIP", `a2bbb7f` readme edit,
  `cbcf5dd` project README, ...
* At the time of writing `main` equals `origin/main` and there are 0 modified tracked files. 529
  untracked entries exist (generated outputs, carryover docs, figures); this handoff file is also
  untracked until committed.
* Other branches: `prototype`, `backup-before-prototype-reset`, `temp` (history/backup, do not delete).
* Git identity on the old machine: `user.name = Akshat`, `user.email = akshatj02@gmail.com`.

### 2.2 Map configuration: NAIP everywhere in the pipeline (done 2026-10-01)

The user decided to use real NAIP data for the paper. The whole pipeline was flipped:

| Setting | NAIP value (current) | Old GRF value |
|---|---|---|
| `LCB` in `gaussianprocesstraining.py` AND `evalmetrics.py` (two separate copies, keep in sync) | `True` | `False` |
| `MAPTYPE` default | `"NAIP"` | `"grf"` |
| utility threshold | `0.3` | `0.5` |
| important cell rule | `mu - beta*sigma <= threshold` (low NDVI = stressed crop = important) | `mu + beta*sigma >= threshold` |
| prior mean of belief | `threshold - 0.1` (pessimistic: everything starts important) | `threshold + 0.1` |
| GP kernel (`initialize_gp` defaults and collector `GP_KERNEL_*`) | `sigma2 = 0.0079`, `lengthscale = 4.78` | `0.05`, `6.08` |

Not flipped on purpose (all documented in section 9): the trained models/datasets the planners point to
(still GRF), the PPO/DPPO side-experiment scripts, GRF-named figure scripts, GP-prior figure scripts,
legacy `multiblob`/`halffield` scripts, `COST_EXPONENT` (still 1.0), and the live CMA-ES generation count
(still 25).

### 2.3 Data collector configuration (`DataCollector_3D_randomstart_CMAESregularized.py`)

| Parameter | Value | Notes |
|---|---|---|
| solutions per round `CMA_SOLUTIONS_PER_BRANCH` | 8 | plain CMA-ES, different seeds, NO diversity penalty (penalty code was removed) |
| CMA-ES generations `CMA_GENERATIONS` | 200 (env override) | collector only; popsize 12, step 20 m xy / 8 m z unchanged |
| retention ratio `NEIGHBOURHOOD_THRESHOLD` | 0.95 | candidate kept if realized masked variance reduction >= 95% of the best |
| Jaccard minimum `JACCARD_MIN_DISTANCE` | 0.3 (env override) | footprint distinctness filter; user judged 0.5 too strict |
| chain continuation `CONTINUE_RANDOM_KEPT_MODE` | on (env `0` restores winner-only) | chain follows a uniformly random KEPT mode each round |
| rounds per chain `RANKLIM` | 16 | |
| maps / starts | `initial_map = 0`, `mapcount = 60`, `STARTS_PER_MAP = 4` | 240 chains |
| `BRANCH_COUNT` | 1 | one grid-search warm start per round |
| scoring / committed execution chunk | 40 / 20 spline points | |
| parallel layout | all 240 (map, start) chains dealt round-robin over all MPI ranks | `COLLECTOR_RESUME=1` skips finished chains |
| winner and retention metric | TOTAL masked variance reduction | the user says the real goal is reduction PER DISTANCE; not yet switched in code |

### 2.4 What was running / pending

* The user launched the collector on the HPC cluster (log showed `MPI size=8`, `RANKLIM=16`, map 0).
  The result is unknown to the previous agent. Chunks land in
  `CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks/` (one file per chain) and the final
  consolidated file goes to `CMAES_beamsearch_dataset_3d_synthetic_final.pt` (see trap 3 below).
* No NAIP model has been trained on data from the new collector yet. Models in `checkpoints/` and
  `Diffusion/checkpoints/` were trained earlier (GRF data, or older NAIP data with heading).

---

## 3. Standing rules and preferences of the user (the previous agent's memory, copied here)

Writing and communication:

* **No em dashes or double hyphens used as a pause in prose.** Do not join clauses with a dash anywhere the
  user reads writing (chat replies, LaTeX, docs). Use a period, comma, colon, semicolon or a conjunction. En-dash number ranges in LaTeX (`91--100`) are fine. The user says dashes make text sound
  machine-written.
* **No explanatory caption text baked into matplotlib figures** (no `fig.text` blocks with methodology or
  caveats). Titles, axis labels, legends and direct annotations are fine. Put caveats in docstrings or
  chat.
* The user writes a thesis and an RA-L paper. For prose, prefer formal but plain wording, few colons and
  semicolons, short sentences. They dislike inflated phrasing.
* For the paper, claims must be measurable, and the multimodal terminology above must be consistent.

Git and attribution:

* **Never add AI attribution to commits or PR text.** No `Co-Authored-By: ...` trailers and no
  "Generated with ..." lines, whatever your service's default is. The user makes the decisions and is the
  sole author. (The user previously had trailers stripped from three historical commits.)
* Commit messages so far are short, lowercase, descriptive sentences, for example
  "randomized mode continuation". Commit and push only when the user asks.

Decisions already made (do not re-propose):

* **Sensor noise is NOT to be rescaled for NAIP.** `noise_model(altitude)` returns a fixed absolute
  variance; NAIP's value range is narrow so noise looks proportionally large and occupied-region RMSE
  curves look flat. The user was asked and declined ("no it's okay").
* The collector's diversity penalty was removed on purpose (section 8).
* The user decided: 8 plain seeded CMA-ES runs per round, 200 generations, retention 0.95, Jaccard 0.3,
  16 rounds, random continuation among kept modes, rank splitting across all chains.
* The user accepted short paths ("shorter paths just mean replanning more often").

Compute reality: the user's laptop is an HP EliteBook 840 G9 (i7-1255U, 10 cores / 12 threads, 32 GB RAM,
**no GPU**; `torch.cuda.is_available()` is False). All local timing is CPU-only. The cluster is Slurm
(partitions `memory` and `gpu-a100-small`, modules `2025`, `cuda`, `python`, `openmpi`).

Working style that was effective: verify before asserting (re-grep, run a tiny test), report honestly
when an estimate turned out wrong, keep edits scoped, and ask before changing selection metrics or
anything that alters which data gets collected.

---

## 4. Technical background

### 4.1 Simulator and belief (`gaussianprocesstraining.py` is the shared source of truth)

* World 100 x 100 m, grid step 2 m, so 51 x 51 = 2601 cells. Agent altitude 10 to 40 m, starts at 10 m.
* Camera: field of view 60 degrees, so footprint half-width `r = z * tan(30 deg)`.
* Resolution blocks by altitude: z <= 20 m uses 1x1 cells, z <= 30 m uses 2x2, above that 4x4. Cells in a
  block are averaged into one measurement row, and measurement noise is correlated within a block.
* Sensor noise variance `noise_model(z) = 0.01 + 0.03 * (1 - exp(-b (z - 10)))` with `b = ln(4)/30`
  (about 0.0325 at 40 m, asymptote 0.04).
* GP prior: `ConstantKernel(sigma2) * Matern(nu=1.5, lengthscale)`, belief tracked as (mu, full
  covariance P) with a Kalman measurement update (`kalman_update`).
* Importance: `importance_filter(mu, P, beta, threshold)` returns the variance on cells that count as
  important (rule in section 2.2, `beta = 1.0`). Utility = variance on important cells.
* Waypoint to path: current pose plus 8 control waypoints are interpolated by a natural cubic spline,
  parametrized by cumulative 3D chord length, 5 samples per segment, so 41 path points
  (`build_spline_trajectory_3d`, and `pytorch_cubic_spline` in the diffusion modules). There is no
  dynamics or smoothness term anywhere in the objective; the natural boundary condition makes the second
  derivative zero at both ends. Execution: the agent steps 2 m toward the next spline point, rounded to the
  grid (`dynamics_3d`).
* Time model in the single-map evaluation scripts: each step has a minimum physical duration
  (`STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS`, 1/3 s at 3 m/s) so replanning latency eats mission time. A
  flight ends at `TIMEALLOTED` steps (default 4000) or `WALLCLOCK_SECONDS` real seconds (default 150),
  whichever comes first.

### 4.2 Expert planner and its objective

* Grid search: `build_pyramid_lattice_3d` lattice, greedy sequential selection by variance gain / distance
  (`_grid_search_3d_impl`, `branch_grid_search_3d` in the collector).
* CMA-ES refinement (`cma_es_refine_waypoints_3d`): 24 variables (8 waypoints x 3), population 12, initial
  step 20 m (x, y) and 8 m (z), bounds x,y in [4, 96], z in [10, 40]; defaults **25 generations**
  (`CMA_PREDICTIVE_MAXITER`), up to 1000 evaluations. Objective (minimized) =
  `-(masked expected variance reduction / path_length^COST_EXPONENT) + boundary penalties`, mirroring
  Popovic's `mav_ipp` implementation. The predicted gain uses a subsampled sensor model
  (`future_sensor_model_3d` with stride 4 and at most 64 measurement rows), so it is a coarse proxy of the
  realized reduction.
* Important property (measured, section 8): optimizing gain/length to convergence collapses to short
  paths; the 25-generation default stops long before convergence.

### 4.3 Data collection (canonical script: `DataCollector_3D_randomstart_CMAESregularized.py`)

Per (map, start) chain, per round: build the warm start, run 8 plain seeded CMA-ES refinements, roll each
out for 40 spline points against the true map with simulated noisy measurements, score each by realized
masked variance reduction (belief-only, no ground truth in the decision), keep the best plus every
candidate within the retention ratio that is also Jaccard-distinct (observed important-cell footprint),
record all kept candidates as labels for the same belief state, then advance the chain by 20 spline points
along a uniformly random kept mode. Seeds are deterministic per (map, start, round, purpose), so results do
not depend on how chains are split across MPI ranks (verified: identical dataset for 1 vs 3 simulated
ranks). Pseudocode in Appendix B, dataset schema in Appendix A.

### 4.4 Models

* **Diffusion model** (current = "headingremoved" variants): conditional DDPM over a (3 x 8) tensor of
  normalized waypoints (x,y by 100 m, z by [10, 40] m, all in [-1, 1]). Cosine noise schedule with
  `T = 20` training steps (a stale docstring in the file says 200). Sampling: DDIM over all 20 steps by
  default, `eta` from env `ETA` (default 0.0 in the planner script).
* **Network** (about 1.54 M parameters per the thesis text): CNN encoder on a 3 x 51 x 51 map (standardized
  mean, standardized variance, one-hot agent marker) down to 12 x 12 = 144 map tokens of width 196;
  waypoint tokens (linear embedding plus sinusoidal order embedding) get diffusion-timestep and
  total-variance embeddings added; 2 blocks of 4-head self-attention over waypoints then cross-attention to
  map tokens; output MLP predicts noise. Heading conditioning was removed (hence "headingremoved").
* **Training** (`Diffusion/threeDSparseTransDiffusion_headingremoved.py`): AdamW, lr 3e-4, weight decay
  1e-4, gradient clipping 1.0, min lr 1e-5, batch 256, 2000 epochs by default.
* **ImitateTrans**: same encoder/tokenizer/attention stack, no diffusion timestep, zero tensor in place of
  the noisy sample, MSE loss on the waypoints (`ImitateTrans_headingremoved.py`).
* **Normalization constants** (`mean_center`, `mean_scale`, `var_center`, `var_scale`,
  `total_variance_*`) are recomputed at import time from the dataset named by `DIFFUSION_DATASET_PATH` /
  `IMITATE_DATASET_PATH`. A checkpoint is only valid together with the dataset it was trained on (trap 1).

### 4.5 Metrics

* `evalmetrics.py`: `compute_reconstruction_rmse` (global and occupied RMSE against the true map; ground
  truth is used for evaluation only), `compute_task_completion`, coverage efficiency.
* **Occupied-region variance** (the headline metric): variance summed over cells that are important in the
  ground truth, over time. Computed post hoc by replaying a saved `executed_trajectory.csv` through the
  Kalman update (`important_region_variance_from_trajectories.py`; the analysis scripts reuse its helpers).
  Reported as final value, percent drop, or time-averaged AUC (lower AUC = variance stayed low longer).
  Time to 80% or 90% reduction is also used.

---

## 5. Directory map

### 5.1 Layout on the old machine

```
C:\Users\Aksha\OneDrive\Year 6\Thesis\            (parent folder; NOT a git repo)
|-- scripts\                  9.9 GB on disk, THE GIT REPO ROOT (only 3.1 MB / 510 files are tracked)
|-- Datasets\                 119 GB. Raw data. Only Datasets\NAIP_dataset matters for this project
|-- literature\ SOTA\ "other theses"\    reference PDFs (diffusion planning, AIPP, diffusion policy, ...)
|-- imgs\ explorerimgs\ diffex\ 051125meeting\ "Important communications"\    thesis material
|-- diffusionimg.svg, networkimg.svg      architecture drawings (draw.io exports)
`-- slurm-*.out, periodic_eval_map_39*.csv   old cluster logs and eval CSVs
```

Datasets\NAIP_dataset (raw NDVI source, 6+ GB, only needed to regenerate maps or draw raw-tile overlays):

| Folder | Size | Purpose |
|---|---|---|
| `selected_tiles/` (65 .tif) and `selected_tiles_csv/` (65 .csv) | 16 MB, 14 MB | the verified NDVI tiles behind the 65 simulator maps. CSVs are named by OLD id, `naip_shuffle_mapping_2026-08-15.csv` maps old to new id |
| `entire_map/`, `ndvi_whole/`, `ndvidata/`, `ndvi_tiles_100m/`, `ndvi_tiles_below_0p2/` | 1 to 2 GB each | full-area NDVI rasters and tiling intermediates |
| `NAIPmapmaker.py` | small | script that cuts tiles into the simulator's grid CSVs |

Method to find the real raw tile for simulator map N (correlation against the selected tiles) is in
section 5.7.

### 5.2 `scripts/` root: libraries and configuration

| File | What it is |
|---|---|
| `gaussianprocesstraining.py` | **core library**: GP init, Kalman update, FOV/sensor matrices, importance filter, lattice, grid search, CMA-ES objective and refinement, spline builders. `LCB`, `COST_EXPONENT`, CMA budget env vars live here |
| `evalmetrics.py` | RMSE, task completion, coverage metrics (has its own `LCB` copy) |
| `hpc_sweep_common.py` | shared helpers for the `sweep_*_hpc.py` drivers |
| `groundtruthgrid.py`, `visualize_map.py`, `normalize_multiblob_maps.py` | map utilities (multiblob/legacy) |
| `assetplacement.py` | legacy generator of the synthetic "plant focus" maps (200 maps, up to 20 foci each) |

### 5.3 `scripts/` root: data collection and datasets

| File | What it is |
|---|---|
| `DataCollector_3D_randomstart_CMAESregularized.py` | **canonical expert data collector** (MPI-parallel). Despite "Regularized" in the name, the diversity regularization has been removed |
| `consolidate_first_40_maps.py` | consolidates per-map chunk files (older chunk naming) into one dataset |
| `make_unimodal_dataset.py` | collapses a multimodal dataset to `parent_beam_index == 0` rows (one path per chain node). Its source and output paths are hard-coded to the OLD multimodal dataset; the winner-only NAIP file `FINAL_NAIP_DATASET_FILTERED.pt` was made the same way. **Caveat:** with random mode continuation (since 2026-10-01) index 0 is the best mode but the chain may have followed another kept mode, so index-0-only no longer reproduces the chain the data actually traversed; use the `continued` field if you need the visited chain |
| `plot_3d_dataset_rank.py`, `tests/` | dataset-rank plotter and its tests (spec in `docs/superpowers/`) |
| `smoketest_cmaes_datacollector_multimodal.py` | one-chain smoke test of the collector |
| `smoketest_diversity_distance_calibration.py` | multi-seed diversity diagnostic (gain ratios, RMS and Jaccard distances, kept counts) |
| `smoketest_variant_winner_check.py` | which seed wins each round |
| `DataCollector+datasets/` | OLD collector generations, old datasets and the HPC `run_*_hpc.sh` sweep scripts. Kept as history; do not edit |

Datasets in `scripts/` (all git-ignored): `FINAL_NAIP_DATASET.pt` (105 MB, 2800 conditions, collected
before the heading removal and before the 2026-10-01 collector changes), `FINAL_NAIP_DATASET_FILTERED.pt`
(85 MB, winner-only), `dataset_grf_60.pt` (146 MB, 3840 conditions, GRF maps 0 to 59, what the current
GRF checkpoints were trained on), `dataset_grf_39.pt`, `CMAES_beamsearch_dataset_3d_synthetic_final.pt`
(52 MB, older; also the collector's default OUTPUT name, trap 3).

### 5.4 `scripts/` root: planners (single-map evaluation, all env-var configurable)

| File | Notes |
|---|---|
| `CMAES_classic_singlemap.py` | live CMA-ES; exports `compute_fov`, `dynamics_3d`, `waypoint_3d` used by others |
| `Diffusionplanner_singlemap_headingremoved.py` | **current** diffusion evaluation planner. Imports the model module `Diffusion/sample_3d_sparse_trans_diffusion_headingremoved.py` as `diffusion` and sets `DIFFUSION_DATASET_PATH` default to `dataset_grf_60.pt` |
| `Diffusionplanner_singlemap.py` | previous version (with heading) |
| `ImitateTrans_singlemap_headingremoved.py`, `ImitateTrans_singlemap.py` | deterministic baseline, current and previous |
| `lawnmower_singlemap.py`, `lawnmower_multimap.py` | non-adaptive coverage |
| `realgreedy_singlemap.py`, `greedygradient_singlemap.py` | myopic baselines |
| `random_singlemap.py` | random baseline. Has a pre-existing SyntaxError (also in git HEAD) |
| `Diffusionplanner_PPO.py`, `PPO_residual_training.py` | PPO/DPPO side experiments, separate from the paper, still GRF |
| `CMAES_classic_eval.py`, `Lawnmower_eval.py`, `Diffusionplanner_eval.py`, `ImitateTran_eval.py` | older batch evaluators (the last two use `multiblob` and are legacy) |

### 5.5 `scripts/` root: training

| File | Notes |
|---|---|
| `Diffusion/threeDSparseTransDiffusion_headingremoved.py` | **current** diffusion model definition and training loop (run from inside `Diffusion/`) |
| `Diffusion/threeDSparseTransDiffusion_periodic_eval_headingremoved.py` | same training with periodic single-map evaluation |
| `Diffusion/sample_3d_sparse_trans_diffusion_headingremoved.py` | inference/sampling module imported by the planners |
| `Diffusion/continue_sparse_trans_training.py` | resume training from a checkpoint (cluster jobs) |
| `ImitateTrans_headingremoved.py`, `ImitateTrans_periodic_eval_headingremoved.py` | ImitateTrans training |
| non-headingremoved twins, `SparseDiffusion.py`, `SparseTransDiffusion.py`, `DPPO*.py`, `dppo_rollout_worker.py` | earlier generations and DPPO side experiments |
| `Diffusion/old/` (25 files, git-ignored), `Diffusion/2D/` (3.2 GB, 12 old 2D datasets, git-ignored), `Diffusion/data/` (566 MB incl. CIFAR) | legacy, ignore unless reproducing 2D history |

### 5.6 `scripts/` root: sweeps, drivers, analysis, figures

* Sweep drivers (run planners many times): `sweep_cmaes_*.py` (beta, maxiter, stepsize, lattice),
  `sweep_*_hpc.py` and `hpc_sweep_common.py` (Slurm array drivers), `checkpoint_sweep_*.py` (one run per
  checkpoint epoch and repeat), `sweep_imitatetrans_stop_at_80pct.py`, `rerun_failed_80pct_checkpoints.py`,
  `execution_chunk_sweep.py`, `evaluate_diffusion_eta_sweep.py`, `compare_*_planners.py`,
  `compare_diffusion_imitation_maps.py`, `run_map51_planner_heatmaps.py`.
* Analysis: `analyze_*.py` (occupied-variance AUC, beta/maxiter/stepsize sweeps, horizon sweeps),
  `important_region_variance_from_trajectories.py` (the shared Kalman-replay helpers),
  `compute_imitatetrans_150s_v2_occupied_variance_auc.py`, `diagnose_cmaes_naip_plateau.py`,
  `test_multimodal_branch_grf60.py`, `test_multimodal_branch_batch.py` (mode-collapse investigation).
* Figures: `plot_*.py` (about 60 thesis/paper figure scripts; many are GRF- or NAIP-specific one-offs),
  `record_*_animation.py` (GIF/MP4 animations), `generate_hero_trajectory*.py` and
  `plot_thesis_cover_topdown.py` (deliberately NAIP, for the thesis cover), `combine_noise_fov_*.py`.
* Shell scripts at root: `diff.sh` (diffusion training job) and `diffcont.sh` (continues training from an
  epoch-1800 checkpoint to 3000 via `continue_sparse_trans_training.py`), both partition `gpu-a100-small`,
  `mapeval_multimodal.sh` (the collector job: Slurm, 3 tasks and 24 h in the repo copy, needs more tasks),
  `run_realgreedy_sweep_hpc.sh`.
* Other folders with code: `Kerneltraining/` (fits GP kernel hyperparameters to real map statistics; source
  of the NAIP values 0.0079 / 4.78), `receding_gridsearch/` (older receding-horizon prototypes),
  `situation_comparison/` (freezes one real belief state and compares what CMA-ES, Diffusion and
  ImitateTrans each propose; has its own README), `old/CMAES.py`, `docs/superpowers/` (one plan and spec
  for the dataset-rank plotter), `tests/`.

### 5.7 Data and outputs that are NOT in git

| Path | Size | What | Needed for |
|---|---|---|---|
| `csv/map_<id>_NAIP_grid_counts.csv` (65 files, ids 1 to 65, plus an extra `map_1_NAIP_grid_counts_normalized_0_100.csv`) | 14 MB | the 65 NAIP maps | **everything on NAIP** |
| `csv/map_<id>_grf_grid_counts.csv` (201 files, ids 0 to 200) | 10 MB | synthetic GRF maps (map 200 = four-corner stress test) | GRF experiments |
| rest of `csv/` (multiblob, halffield, blob, ...) | about 600 MB | legacy synthetic maps | legacy only |
| `naip_shuffle_mapping_2026-08-15.csv` | tiny | old to new NAIP map id | raw-tile lookup |
| `*.pt` datasets (list in 5.3) | 52 to 146 MB each | collected expert data | training |
| `checkpoints/` (14 files, 193 MB) | | ImitateTrans checkpoints (`*_headingremoved.pth` epochs 1000 to 1500, `NAIP_FINAL_IMIT.pth`, older) | evaluation |
| `Diffusion/checkpoints/` (50 files, 830 MB) | | diffusion checkpoints (`sparse_trans_waypoints_epoch_*_headingremoved.pth`, `NAIP_FINAL_DIFF*.pth`, `UNIMODAL_NAIP_FINAL_DIFF_*.pth`, DPPO iterations, older) | evaluation |
| `CMAES_2D/` (1.2 GB) | | old 2D-era datasets, has its own README | legacy |

Real-tile lookup for NAIP map N (from the user's note): use `naip_shuffle_mapping_2026-08-15.csv` to find the
old id; `Datasets/NAIP_dataset/selected_tiles_csv/map_<OLD>_NAIP_grid_counts.csv` should be byte-identical to
`csv/map_<N>_NAIP_grid_counts.csv`; correlate each `selected_tiles/*.tif` (resampled to 101 x 101, trying
identity / flips / transpose) against it. A true match has r about 0.97, others about 0.6 to 0.7.

### 5.8 Generated results (all untracked, safe to leave behind)

`results_*/` (CMA-ES beta/maxiter/stepsize sweeps, horizon sweeps, smoke tests), `Vizualization*/`
(about 517 run folders named `<planner>_map_<MAPTYPE>_<id>_viz_<tag>/`, each with trajectory CSVs, plots,
GIFs; note the spelling "Vizualization"), `planner_outputs/` (HPC repeat runs, 226 folders containing
`executed_trajectory.csv`), `*_logs/`, `*_batch_metrics/`, `sweep_results/`, `plots/`, `hero_shot_data/`,
`start_frames/`, root-level `*.png`, `*.gif`, `*.mp4`, `*.csv`, `slurm-*.out`, LaTeX table fragments
`results_*_table.tex` (these are generated for the paper's tables).

The carryover documents (untracked, hand-written, high value):

| File | Topic |
|---|---|
| `CARRYOVER_PROJECT_SCOPE_OVERVIEW_2026_09_30.md` | top-level orientation, the GRF/NAIP flip pattern, paper positioning, statistics plan |
| `CARRYOVER_NAIP_NORMALIZATION_BUG_AND_MULTIMODAL_INVESTIGATION.md` | normalization-at-import bug, execution chunk, multimodal vs unimodal data, checkpoint table |
| `CARRYOVER_NAIP_SWITCH_AND_BASELINES.md` | first NAIP switch, baselines, occupied-variance AUC method |
| `CARRYOVER_CHECKPOINT_SWEEPS_DIFFUSION_VS_IMITATETRANS.md` | checkpoint sweeps |
| `CARRYOVER_GRF_SWITCH_DPPO_REWARD_AND_INFOGRAPHICS.md` | GRF switch, DPPO reward, infographics |
| `CARRYOVER_CMAES_REWARD_2.md`, `CARRYOVER_CMAES_REWARD_3.md` | CMA-ES objective and hyperparameters (earliest) |

---

## 6. How to run things

Environment variables configure every single-map script; defaults are in the file headers. In bash:

```bash
# evaluate one planner on one NAIP map (150 s wall-clock budget)
SELECTED_MAP=5 MAPTYPE=NAIP WALLCLOCK_SECONDS=150 python Diffusionplanner_singlemap_headingremoved.py
SELECTED_MAP=5 MAPTYPE=NAIP WALLCLOCK_SECONDS=150 python CMAES_classic_singlemap.py
SELECTED_MAP=5 MAPTYPE=NAIP WALLCLOCK_SECONDS=150 python lawnmower_singlemap.py
```

PowerShell equivalent: `$env:SELECTED_MAP=5; $env:MAPTYPE="NAIP"; python Diffusionplanner_singlemap_headingremoved.py`.

Useful variables: `SELECTED_MAP`, `MAPTYPE`, `UTILITY_THRESHOLD`, `WALLCLOCK_SECONDS` (`<=0` runs to a fixed step
count), `TIMEALLOTED`, `EXECUTION_CHUNK`, `ENFORCE_MIN_STEP_TIME`, `RUN_SEED`, `RUN_OUTPUT_TAG` (keeps runs
from overwriting each other under `Vizualization/`), `SKIP_VIZ=1`, `ETA`, `DIFFUSION_CHECKPOINT`,
`DIFFUSION_DATASET_PATH`, `IMITATE_DATASET_PATH`, `CSV_DIR`, `RESULTS_ROOT`.

**Quick sanity check after porting** (a minute or two; uses a tiny CMA-ES budget):

```bash
CMA_GENERATIONS=2 SMOKETEST_ROUNDS=1 SMOKETEST_MAP=5 python smoketest_diversity_distance_calibration.py
```

**Collector locally, tiny test** (do not point it at the default output file; copy the pattern used by the
previous agent: import the module, override `dc.RANKLIM`, `dc.mapcount`, `dc.STARTS_PER_MAP`,
`dc.chunk_dir`, `dc.final_dataset_path`, then call `dc.main()`).

**Collector on the cluster** (Slurm, MPI): see `mapeval_multimodal.sh`; the launch line is
`srun --mpi=pmix python -u DataCollector_3D_randomstart_CMAESregularized.py`. Resource sizing (estimate from
laptop timings, about 2 min per 200-generation CMA-ES run, about 4 h per 16-round chain, unmeasured on the
cluster): `--ntasks=240` about 4 to 5 h, 120 about 8 to 9 h, 60 about 17 h, 30 about 34 h (too long for a
24 h job; use `COLLECTOR_RESUME=1` and resubmit). Pin `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1`. The `module: command not found` lines seen in one job log were harmless (the job
ran); if it ever matters, use `#!/bin/bash -l` as the first line.

**Training:**

```bash
cd Diffusion
DIFFUSION_DATASET_PATH=../<dataset>.pt python threeDSparseTransDiffusion_headingremoved.py
# ImitateTrans (from scripts/):
IMITATE_DATASET_PATH=<dataset>.pt python ImitateTrans_headingremoved.py
```

Checkpoints are written as `Diffusion/checkpoints/sparse_trans_waypoints_epoch_<N>_headingremoved.pth` and
`checkpoints/imitate_trans_waypoints_epoch_<N>_headingremoved.pth`. Then point the planner at the model with
`DIFFUSION_CHECKPOINT` AND set `DIFFUSION_DATASET_PATH` to the same dataset.

**Python environment** (versions on the old laptop): Python 3.12.3, torch 2.5.1 (CPU), numpy 2.2.6, scipy
1.13.1, scikit-learn 1.6.1, matplotlib 3.9.0, cma 4.4.4, imageio 2.37.0, tqdm 4.67.1, pandas 2.2.3,
mpi4py 3.1.6 (needs an MPI runtime; absent on the laptop, so ranks were simulated in tests). No
`requirements.txt` exists. Cluster env: conda env `diffusion_env`.

```bash
pip install torch numpy scipy scikit-learn matplotlib cma imageio tqdm pandas
pip install mpi4py        # only where an MPI runtime exists
```

---

## 7. Key facts about the collector (so you do not have to rediscover them)

* Each round: grid-search warm start (`BRANCH_COUNT = 1`), 8 CMA-ES runs from the same warm start with 8
  different seeds, 40-step rollouts, selection and recording as in section 4.3.
* The winner is index 0 in the recorded group (`parent_beam_index` 0); the field `continued` marks which
  recorded mode the chain actually followed. `condition_id` groups all recorded modes for one belief state.
* Filters are applied in this order: realized reduction >= 0.95 of the best, then greedy Jaccard filter
  (sorted by reduction, keep if footprint Jaccard distance to every kept one >= 0.3).
* Footprint = union of camera fields of view along the 40 flown spline points, restricted to the
  belief-based important mask (no ground truth).
* Chunks: `map_XXX_start_YY_ranked_3d.pt` per chain, written atomically; rank 0 consolidates the paths
  gathered via `comm.gather`. Resume: `COLLECTOR_RESUME=1`.
* The 3 smoke-test / diagnostic scripts in section 5.3 and the temporary experiment scripts of the
  previous agent (see section 10) are the fastest way to see the collector's behaviour.

---

## 8. Findings so far

Planner comparisons (from the carryover docs, GRF and older NAIP runs; re-run under NAIP before citing):

* GRF map 51, 300 s wall-clock: diffusion beats ImitateTrans and CMA-ES on occupied-variance AUC (AUC
  3.764 vs 5.516 vs 8.495), but CMA-ES replans take 10 to 60 s vs about 0.1 to 0.3 s, so a fixed
  wall-clock budget structurally favors learned planners. Keep that caveat explicit.
* Synthetic four-corner map 200: ranking inverts (CMA-ES AUC 1.933, best; ImitateTrans 3.460, worst):
  joint 8-waypoint planning routes between far-apart regions, learned planners dwell near the start. So
  "it depends on map structure", not "diffusion wins".
* The raw CMA-ES demonstrations themselves beat the lawnmower, so the remaining gap to the expert lives in
  the imitation step, not in data quality. A CMA-ES budget mismatch between collection and live
  comparison, and a kernel-lengthscale explanation, were checked and ruled out (earlier NAIP era).
* Multimodality of diffusion is condition-dependent: at epoch 3000 some conditions collapse to one mode
  (0 of 80 samples near the losing branch); epoch 1000 collapses less but is not accurate. Plain
  denoising-MSE training trades mode coverage for accuracy. `dataset_grf_60.pt` has 753 of 3840 conditions
  with two modes; `FINAL_NAIP_DATASET.pt` has 608 of 2800.
* Using `EXECUTION_CHUNK = 20` at inference (matching the collector's committed chunk) improved results
  versus 30; it was not made the default at that time (see trap 5).

Findings from the 2026-10-01 session (all on NAIP map 5, start 1; small samples, treat as indicative):

* **CMA-ES at the default budget is not converged.** 25 generations with step 20 m in 24 dimensions gives
  widely different runs per seed (reduction spread 0.5 to 1.0 of best); many equivalent-looking
  "diverse" solutions were really unfinished searches.
* **Converged CMA-ES collapses to short paths.** The objective is gain divided by path length
  (`COST_EXPONENT = 1.0`). At 200 generations all 8 seeds ended at nearly the same 40 to 65 m path,
  objective improved about 4x but total masked variance reduction fell from about 8 to about 3 (of 20.1).
  The 25-generation default only produced long, high-reduction paths because it stopped early. The user
  accepted short paths (more frequent, better replans).
* **The distance penalty biased results.** With the old penalty, the last (most constrained) variant won
  every round, and the plain run ranked near the bottom. It was removed (user decision).
* **Yield of multimodal rounds is low.** In 3-round chains, 1 mode was kept of 8 in every round at 98%
  retention (runner-ups at 0.81 to 0.96 of best). On reduction per metre, round 0 had 3 near-ties within
  10% and mutually distinct; later rounds had a single dominant short path. Without the penalty, 1 round of
  3 kept 2 modes at 98% (on reduction per metre). This is the main open risk for claims (ii) and (iii):
  measure the multimodal yield on the real collection before relying on it.
* **Jaccard calibration**: across 84 candidate pairs the closest was 0.25, median 0.64; a "weakly
  distinct" pair the user approved sat at 0.53; hence the 0.3 default (rejects about 2% of pairs; 0.5
  rejected about 25%).
* **Parallel efficiency lesson**: 8 processes each running multi-threaded BLAS made 200-generation runs take
  about 890 s each; pinning one thread per process and evaluating the 12 candidates of a generation across
  12 workers gave 15 to 20 s per run (and 57 s per 100-generation run with 8 single-thread processes).
* Covariate-shift discussion: keeping only the best mode's successor states means alternate-mode states
  never appear in training. Implemented fix: random continuation among kept modes (zero extra cost).
  Discussed next options: short capped side branches from multimodal rounds, and DAgger-style
  on-policy relabeling once a first policy exists.

---

## 9. Known issues and traps (read before running anything)

1. **Normalization-at-import trap (the single most damaging past bug).** The model modules recompute their
   normalization constants from `DIFFUSION_DATASET_PATH` / `IMITATE_DATASET_PATH` at import. If the path
   points at a different dataset than the checkpoint was trained on, every inference silently runs with
   wrong scales (about 1.6 to 1.8x, loss inflated 7.4x in the original incident). Never re-default a dataset
   path because `MAPTYPE` changed. Always pair a checkpoint with its own training dataset. Cheap check:
   recompute a checkpoint's training loss under the intended normalization and compare with its recorded
   loss.
2. **GRF/NAIP flip pattern.** Re-grep before trusting any state:
   ```bash
   grep -n "^LCB" gaussianprocesstraining.py evalmetrics.py
   grep -n "MAPTYPE\", *\"\|MAPTYPE = " *.py Diffusion/*.py | grep -v "NAIP"
   grep -n "UTILITY_THRESHOLD\|^utility_threshold\|EVAL_UTILITY_THRESHOLD" *.py | grep -v "0\.3"
   grep -n "threshold + 0\.1\|THRESHOLD + 0\.1" *.py Diffusion/*.py
   grep -n "def initialize_gp" gaussianprocesstraining.py
   ```
   Also confirm the important-mask direction (`<=` under LCB) wherever a script builds its own mask.
3. **Collector output name collision.** `final_dataset_path` is `CMAES_beamsearch_dataset_3d_synthetic_final.pt`
   in `scripts/`, and a file with that name already exists (older GRF/synthetic data, 52 MB). A finished
   collection will overwrite it. Back it up or change `final_dataset_path` before the first full run. Also
   `consolidate_chunks` deletes chunk files after consolidating.
4. **Map id range.** On this laptop `csv/` has NAIP map ids 1 to 65 (no map 0), but the collector defaults
   to `initial_map = 0`, `mapcount = 60` and the cluster job log showed it processing map 0. Verify which
   numbering the cluster `csv/` uses before submitting. Maps 1, 9 and 49 (old numbering in the notes) have
   almost no important region.
5. **Train/test overlap.** The default collection range (maps 0 to 59) includes the maps used by the
   evaluation sweeps (51 to 59, 41 to 50, and singles such as 30 and 56). The paper's claims say
   "held-out maps". Decide the split before collecting or training.
6. **Execution chunk mismatch.** `EXECUTION_CHUNK` default is 20 in `Diffusionplanner_singlemap.py` and 40 in
   `Diffusionplanner_singlemap_headingremoved.py`; the collector commits 20. Use 20 for all reported
   inference results unless there is a reason not to.
7. **Budget mismatch.** The collector uses 200 CMA-ES generations; the live CMA-ES planner (the paper's
   "online expert") still defaults to 25. Decide which budget the baseline should have and state it.
8. **Models vs. maps.** The planners now default to NAIP maps but their default checkpoints and datasets are
   GRF-trained. A NAIP, no-heading model must be trained on data from the new collector.
9. **`COST_EXPONENT`** default 1.0 (the NAIP carryover recommended 0.5 for NAIP's disjoint important
   clusters). The objective also drives short-path collapse (section 8). The selection metric (total vs
   per-distance) is undecided.
10. **Flight-speed fairness (as of 2026-08, unverified now):** lawnmower used 2 m/s and diffusion 3 m/s in
    their scripts, which confounds shared-wall-clock comparisons. Check before reporting.
11. **Unreviewed or broken scripts:** `random_singlemap.py` (SyntaxError at line 103), `lawnmower_multimap.py`
    (utility threshold 0.0), `scratch_*.py`, stale docstrings (diffusion file says T = 200, code uses 20).
12. **Paper text out of date:** Algorithm 1 still describes the diversity penalty and picks the best mode;
    the claims paragraph still says "DIPPer".
13. **Line endings.** The old machine is Windows with `core.autocrlf=true`; working files are CRLF, git
    blobs are LF. On Linux/cluster make sure shell scripts are LF (`sed -i 's/\r$//' script.sh`). Git prints
    "LF will be replaced by CRLF" warnings; they are harmless.
14. **`.gitattributes` says `*.pt filter=lfs`** but `.gitignore` excludes `*.pt`, `*.pth`, `checkpoints/`,
    `csv/`, so none of the data is in git or in LFS. Do not assume the repo contains data.
15. `torch.load` prints `weights_only` FutureWarnings; harmless.
16. The folder path contains spaces and sits inside OneDrive; quote paths.

---

## 10. Open items and suggested next steps

1. Confirm what the cluster collection produced; consolidate chunks into a NAIP dataset (back up the old
   `CMAES_beamsearch_dataset_3d_synthetic_final.pt` first).
2. Measure the multimodal yield on that dataset (conditions with more than one recorded mode; Jaccard and
   RMS spread between modes; `continued` frequencies). This decides how strong claims (ii) and (iii) can be.
3. Decide the selection metric (total reduction vs reduction per distance), `COST_EXPONENT`, and the live
   CMA-ES budget. The user stated the goal is reduction per distance but has not asked for the code change.
4. Decide the train/test map split (trap 5), then train NAIP diffusion and ImitateTrans (no heading) on the
   new data, with dataset paths matching the checkpoints.
5. Re-run the planner comparison (CMA-ES, Diffusion, ImitateTrans, lawnmower, greedy) on held-out NAIP maps;
   statistics plan: paired by map, Wilcoxon signed-rank for two planners, Friedman plus Holm-corrected
   pairwise Wilcoxon for several; average diffusion's 3 seeds per map first; treat "never reached the
   threshold" as censored (reach-rate test with McNemar or Fisher, plus a continuous metric at a fixed time
   budget); report effect sizes.
6. Hardware experiment (claim iv): replay a real flown trace through the Kalman-replay tool and compare to
   the simulated curve for the same planner and map; report onboard replanning latency and tracking error.
   Only keep claim (iv) and "UAV" in the title if this is done.
7. Optional methodology: short capped side branches from multimodal rounds; DAgger-style relabeling once a
   first policy exists; parallelize CMA-ES candidate evaluation inside the collector (about 7x faster per
   run on 12 cores in tests, not implemented in the repo).
8. Paper: pick the name and title, replace "DIPPer", update Algorithm 1 (Appendix B), fill the experiments
   section.

Temporary experiment scripts written by the previous agent live OUTSIDE the repo, in a temp scratchpad on
the old machine (`...\AppData\Local\Temp\claude\...\scratchpad\`): `chain3.py` / `chain3_nopen.py` (3-round
chain driver with pooled, BLAS-pinned CMA-ES evaluation), `long_cma_100.py` (parallel seeds at a given
generation count), `plot_chain2.py` / `plot_set2.py` (figures of the candidate sets). Ask the user whether
to preserve them; they may be gone.

---

## 11. Git: state, porting to a new device, and controlling git safely

### 11.1 What git does and does not hold

* Tracked: 510 files, 3.1 MB total (source code, a few docs, shell scripts). History from 2026-06 on.
* NOT tracked, NOT recoverable from GitHub: all datasets (`*.pt`), all checkpoints (`*.pth`), the whole
  `csv/` folder (the maps!), all result and visualization folders, the `CARRYOVER_*.md` documents, this
  handoff file, and the raw NAIP data in the parent `Datasets/` folder.
* A fresh `git clone` therefore gives you code only. The new machine also needs the data tiers below.

### 11.2 Step 0: choose how the data travels

* **Option A (simplest, if the new device uses the same Microsoft account):** the project lives in OneDrive,
  so signing in on the new device can sync everything. Warning: do not run git on the SAME synced `.git`
  from two machines at once; OneDrive can corrupt a repo. Safer: use OneDrive only as a file carrier, then
  clone the repo fresh to a non-synced folder (for example `C:\work\DiffusionAIPP` or `~/DiffusionAIPP`) and
  copy the data tiers into it.
* **Option B:** zip the tiers (11.4) onto an external drive or a private cloud bucket.
* Never put datasets or checkpoints into normal git history (GitHub rejects files over 100 MB; the 146 MB
  `dataset_grf_60.pt` would be rejected, and the repo would bloat). If you want them versioned, use Git LFS
  or a release asset, which is a deliberate decision for the user.

### 11.3 Step 1 (old device): save the context documents in git (user's decision)

Only if the user agrees. These are small and carry most of the project knowledge:

```bash
cd "C:/Users/Aksha/OneDrive/Year 6/Thesis/scripts"
git status -sb                           # expect: main, no modified tracked files
git switch -c handoff                    # keep it off main until reviewed
git add PROJECT_HANDOFF_FOR_NEW_AGENT.md CARRYOVER_*.md
git commit -m "add project handoff and carryover docs"
git push -u origin handoff
```

Then merge to `main` through GitHub or `git switch main && git merge --ff-only handoff && git push`.
Do not use `git add -A` or `git add .`: 529 untracked entries (GIFs, MP4s, PNGs, result folders) are not
all ignored and would be added.

### 11.4 Step 2 (old device): package the untracked assets by priority

Run from `scripts/`. Adjust `$dest` to your staging folder or drive.

| Tier | Contents | Size | Needed for |
|---|---|---|---|
| 1 | `csv/map_*_NAIP_grid_counts.csv`, `csv/map_*_grf_grid_counts.csv`, `naip_shuffle_mapping_2026-08-15.csv` | about 25 MB | running any simulation |
| 2 | datasets: `FINAL_NAIP_DATASET.pt`, `FINAL_NAIP_DATASET_FILTERED.pt`, `dataset_grf_60.pt`, `dataset_grf_39.pt`, `CMAES_beamsearch_dataset_3d_synthetic_final.pt`; checkpoints you actually use: `Diffusion/checkpoints/*_headingremoved.pth`, `Diffusion/checkpoints/NAIP_FINAL_DIFF*.pth`, `Diffusion/checkpoints/UNIMODAL_NAIP_FINAL_DIFF_*.pth`, `checkpoints/*_headingremoved.pth`, `checkpoints/NAIP_FINAL_IMIT.pth` | about 0.6 to 1.2 GB | training and evaluating without recollecting |
| 3 | `CARRYOVER_*.md`, `PROJECT_HANDOFF_FOR_NEW_AGENT.md` (if not committed), `results_*_table.tex`, key CSV summaries at root | small | context, paper tables |
| 4 | `Datasets/NAIP_dataset/selected_tiles` and `selected_tiles_csv` (30 MB) | | raw-tile overlays only |
| 5 (optional) | `results_*/`, `Vizualization*/`, `planner_outputs/`, figures | several hundred MB | reproducing figures without rerunning |
| skip | `Diffusion/2D/` (3.2 GB), `CMAES_2D/` (1.2 GB), `Diffusion/old/`, `__pycache__/`, `Datasets/` raw rasters (about 6 GB, only for regenerating maps) | | legacy |

PowerShell (preserves folder structure for the checkpoint subsets):

```powershell
$dest = "D:\port"
New-Item -ItemType Directory -Force "$dest\csv","$dest\checkpoints","$dest\Diffusion\checkpoints" | Out-Null
Copy-Item csv\map_*_NAIP_grid_counts.csv, csv\map_*_grf_grid_counts.csv "$dest\csv\"
Copy-Item naip_shuffle_mapping_2026-08-15.csv "$dest\"
Copy-Item FINAL_NAIP_DATASET.pt, FINAL_NAIP_DATASET_FILTERED.pt, dataset_grf_60.pt, dataset_grf_39.pt, CMAES_beamsearch_dataset_3d_synthetic_final.pt "$dest\"
robocopy Diffusion\checkpoints "$dest\Diffusion\checkpoints" *_headingremoved.pth NAIP_FINAL_DIFF*.pth UNIMODAL_NAIP_FINAL_DIFF_*.pth
robocopy checkpoints "$dest\checkpoints" *_headingremoved.pth NAIP_FINAL_IMIT.pth
Copy-Item CARRYOVER_*.md, PROJECT_HANDOFF_FOR_NEW_AGENT.md "$dest\"
```

### 11.5 Step 3 (new device): set up and verify

```bash
# 1. tools: git (and git-lfs only if you decide to use it), Python 3.12, an env
git clone https://github.com/akstamimech/DiffusionAIPP.git
cd DiffusionAIPP
git config user.name  "Akshat"                  # or the identity the user wants
git config user.email "akshatj02@gmail.com"
git config core.autocrlf input                  # Linux/macOS; on Windows keep true
python -m venv .venv && source .venv/bin/activate        # or a conda env
pip install torch numpy scipy scikit-learn matplotlib cma imageio tqdm pandas
# 2. copy the tiers from 11.4 into the clone with the SAME relative paths (csv/, checkpoints/, ...)
# 3. verify
git status -sb && git log --oneline -5
grep -n "^LCB" gaussianprocesstraining.py evalmetrics.py            # both True
CMA_GENERATIONS=2 SMOKETEST_ROUNDS=1 SMOKETEST_MAP=5 python smoketest_diversity_distance_calibration.py
```

`.gitignore` already hides `csv/`, `*.pt`, `*.pth`, `checkpoints/`, `*.log`, `*_viz/`, `*_chunks/`, so the
copied data will not show up in `git status`.

### 11.6 Rules for the new agent's use of git (give these to it verbatim)

1. Work on a branch (`git switch -c agent/<topic>`); merge to `main` only when the user says so.
2. **Commit and push only when the user asks.** Show `git status -sb` and `git diff --stat` first.
3. **Stage explicitly**: `git add <path> ...` or `git add -u` (tracked files only). Never `git add -A` or
   `git add .`, because hundreds of generated outputs are untracked.
4. **No AI attribution of any kind** in commit messages or PR text (no co-author trailer, no "generated
   by"). Turn off any such default in your service's settings. Commit messages: short, lowercase,
   descriptive, imperative or past tense, matching the history.
5. Never commit files over 50 MB or anything under `csv/`, `checkpoints/`, `*.pt`, `*.pth`.
6. Before pushing: `git fetch origin && git status -sb`. If behind, `git pull --rebase origin <branch>`;
   never `push --force` or `--force-with-lease` to `main` without the user's explicit instruction (a local
   history rewrite was done once on 2026-09-13 to strip trailers; do not repeat unprompted).
7. Do not rewrite or delete the branches `prototype`, `backup-before-prototype-reset`, `temp`.
8. Do not commit settings changes that flip GRF/NAIP defaults without the audit in section 9, trap 2.
9. Keep experiment scripts out of the repo root unless the user wants them tracked; prefer a `tools/`
   folder and ask first.

### 11.7 Suggested `.gitignore` additions (not applied; propose to the user)

```gitignore
results_*/
Vizualization*/
planner_outputs/
sweep_results/
*_batch_metrics/
*_logs/
hero_shot_data/
start_frames/
situation_comparison/belief_cache/
slurm-*.out
slurm-*.err
*.gif
*.mp4
*.npz
```

Check first with `git ls-files | grep -E "\.(gif|mp4|npz)$"` that none of these patterns would hide files
that are already tracked and wanted (ignoring does not untrack existing files, but it avoids surprises).

### 11.8 If the cluster also needs the update

On the cluster: `git pull --rebase origin main` inside its clone, strip CRs from shell scripts if they were
edited on Windows, and keep its own `csv/` and checkpoint folders (they are not in git).

---

## Appendix A. Dataset schema (what the collector writes; tensors with N samples)

| Key | Shape / type | Meaning |
|---|---|---|
| `trajectories` | N x 3 x 41 | dense spline path of the recorded candidate |
| `control_waypoints` | N x 3 x 8 | the 8 CMA-ES waypoints (training target) |
| `current_position` | N x 3 | agent pose at the start of the round |
| `current_mean`, `current_var`, `current_util` | N x 51 x 51 | belief mean, variance, importance utility |
| `map_id`, `start_index`, `timestep` | N | map, start index within map, round index |
| `beta`, `RMSE_correction`, `variance_correction` | N | beta; RMSE improvement (diagnostic only); masked variance reduction (selection score) |
| `condition_id`, `parent_beam_id` | N | group id: all recorded modes for one belief state share it |
| `parent_beam_index` | N | 0 = best mode, 1.. = other kept modes |
| `initial_heading_velocity`, `start_position` | N x 3, N x 2 | legacy / bookkeeping (heading no longer used by the model) |
| `continued` | N, 0/1 | 1 if the chain continued from this mode (new 2026-10-01; absent in older datasets) |

## Appendix B. Current data-collection procedure (replace the paper's Algorithm 1 with this)

```
Require: maps {M_m}, start poses and prior (x0, b0), starts S, rounds R, solutions J, scoring chunk K,
         retention ratio eta, Jaccard threshold d
Ensure:  dataset D
for each map M_m and start s in 1..S:
    (x, b) <- (x0, b0)
    for round r in 1..R:
        psi0 <- GridSearch(x, b)                                 # greedy lattice warm start
        for j in 1..J:
            psi_j <- CMA-ES(psi0, J_var; seed_j)                 # plain runs, different seeds, no penalty
            (delta_j, x_j, b_j) <- Rollout(psi_j, K)             # realized masked variance reduction
        A <- { j : delta_j >= eta * max_i delta_i }              # near-optimal set
        A <- GreedyJaccard(A, d)                                 # sort by delta, keep if footprint distance >= d
        D <- D union { (x, b, psi_j) : j in A }                  # all kept modes are labels for (x, b)
        draw j' uniformly from A
        (x, b) <- (x_j', b_j')                                   # chain follows one kept mode
return D
```

Defaults: J = 8, eta = 0.95, d = 0.3, R = 16, K = 40 (commit the first 20 points), 200 CMA-ES generations.
Selection score is still total masked variance reduction (see section 10, item 3).

## Appendix C. Settings and variables quick reference

| Where | Variable | Default | Meaning |
|---|---|---|---|
| collector | `CMA_GENERATIONS` | 200 | CMA-ES generations (collector only) |
| collector | `JACCARD_MIN_DISTANCE` | 0.3 | footprint distinctness |
| collector | `CONTINUE_RANDOM_KEPT_MODE` | 1 | random kept-mode continuation |
| collector | `COLLECTOR_RESUME` | 0 | skip finished chains |
| gp library | `CMA_PREDICTIVE_MAXITER`, `_POPSIZE`, `_MAXFEVALS` | 25, 12, 1000 | live CMA-ES budget |
| gp library | `CMA_STEP_SIZE_XY`, `CMA_STEP_SIZE_Z` | 20, 8 | CMA-ES initial steps |
| gp library | `COST_EXPONENT` | 1.0 | power on path length in the objective |
| planners | `SELECTED_MAP`, `MAPTYPE`, `UTILITY_THRESHOLD`, `WALLCLOCK_SECONDS`, `TIMEALLOTED`, `EXECUTION_CHUNK`, `ENFORCE_MIN_STEP_TIME`, `RUN_SEED`, `RUN_OUTPUT_TAG`, `SKIP_VIZ`, `ETA`, `SENSORNOISE_SEED`, `START_X`, `START_Y`, `PLANNING_HORIZON` | see file headers | single-map evaluation |
| models | `DIFFUSION_CHECKPOINT`, `DIFFUSION_DATASET_PATH`, `IMITATE_DATASET_PATH` | see traps 1 and 8 | model and its matching dataset |
