# Carryover — NAIP switch-back, baseline 3D plots, occupied-variance-AUC methodology, diffusion multimodality testing

Written for a future agent with no memory of this session. This is a large, multi-thread session,
unrelated in content to the earlier `CARRYOVER_CMAES_REWARD*.md` files (those cover CMA-ES
objective/cost-normalization bugfixes and hyperparameter sweeps from an earlier session — still
accurate, not superseded, just a different topic). **The single most important thing in this file is
§1 (the NAIP switch) — it's the most recent work, it changes default behavior across many scripts,
and the user gave an explicit "do not make any mistakes" instruction for it that a future agent should
take seriously by verifying rather than assuming.**

## 0. Repo/git state

Still no commits made — consistent with every prior session on this project. Everything below is
uncommitted working-tree state on branch `prototype`. Many new scripts and result files were created
this session; none are tracked in git.

## 1. NAIP switch-back — READ THIS FIRST, VERIFY BEFORE TRUSTING

The user is going back to NAIP-based (real aerial imagery) data collection, moving away from this
session's earlier GRF (synthetic Gaussian random field) work. **"We are only doing NAIP now on"** —
this is a stated, permanent-going-forward decision, not an experiment.

### 1.1 The design decisions the user explicitly confirmed (these are settled, not open questions)

- **LCB (lower confidence bound) is correct for NAIP**, replacing UCB (upper confidence bound). The
  mechanism: `gaussianprocesstraining.py` has a module-level `LCB` flag read by `importance_filter()`
  (originally at lines ~356-388): `if LCB: importance_mask = (mu - beta*sigma) <= threshold` (LCB) `else:
  importance_mask = (mu + beta*sigma) >= threshold` (UCB, was the sole behavior all session since `LCB`
  was hardcoded `False` everywhere). The task list for this session shows "Fix core mechanism: LCB flags
  in gaussianprocesstraining.py and evalmetrics.py" marked completed — **a future agent should grep for
  `LCB` in both files and confirm it now reads `True` (or is wired to something that resolves to `True`
  for the NAIP-only workflow) rather than trusting this note alone.**
- **NAIP's value range is correct as-is and should NOT be rescaled to [0,1].** Empirically confirmed this
  session (direct CSV inspection): NAIP ground truth spans roughly **[-0.07, 0.6]**, mean **~0.27-0.34**,
  never approaching 1.0 — genuinely different from GRF's clean [0,1] range (mean ~0.495). This is
  intentional, not a bug to fix.
- **`mean = utility_threshold - 0.1` ("pessimistic" prior) is correct and intentional** — the stated
  reasoning: "the idea is that we initialize the entire area as important at first." (This pairs with
  LCB: a low initial mean, under an LCB criterion, marks the whole map as below-threshold/important at
  the start, matching the "explore everything, narrow down" intent.)
- **`utility_threshold` should be `0.3` everywhere now** (not the 0.5 used for GRF work all session).
  This was already precedented pre-session in `CMAES_classic_eval.py`'s comment ("For NAIP its 0.3") but
  needed standardizing across every script.
- **The primary/canonical DataCollector script is `scripts/DataCollector_3D_randomstart_CMAESregularized.py`**
  — directly in `scripts/`, **not** the similarly-named files inside `scripts/DataCollector+datasets/`
  (`DataCollector_3D.py`, `DataCollector_3D_randomstart.py`, etc.). Those latter files already happened to
  hardcode `MAPTYPE="NAIP"` pre-session (found during investigation) but are **not** the ones the user
  cares about going forward — don't confuse the two when someone says "the DataCollector file."

### 1.2 What was found broken/inconsistent (investigated via a dedicated Explore agent pass, then fixed)

- `evalmetrics.py` has its **own separate, unsynced** `LCB` flag (was `False`, independent copy from
  `gaussianprocesstraining.py`'s), used by `compute_reconstruction_rmse()`'s occupied-mask direction
  (`true_map <= threshold` for LCB vs `>= threshold` for UCB). Must move in lockstep with the planner's
  flag or the metric and the planner disagree about what "important" means.
- **`utility_function()`** (a *second*, structurally different utility function in
  `gaussianprocesstraining.py`, used by `lawnmower_singlemap.py`, `lawnmower_multimap.py`, and the whole
  `receding_gridsearch_*` family) **has no LCB/UCB concept at all** — "higher mean = more important" is
  unconditional. Flipping the `LCB` flag does **nothing** for these scripts; if they need NAIP/LCB support
  they need actual new code, not a flag flip. Worth checking whether these scripts are even in scope for
  the user's current NAIP work before spending time on them.
- **`hpc_sweep_common.py`** hardcoded `"MAPTYPE": "grf"` into every subprocess env unconditionally — would
  silently force GRF for any sweep launched through the shared HPC helper regardless of caller intent.
  Task list shows this was fixed this session.
- **`important_region_variance_from_trajectories.py`** (the script behind every occupied-variance-AUC
  table/plot built this session — see §2) was **entirely GRF-hardcoded**: `GROUND_TRUTH_THRESHOLD = 0.5`
  with no override, ground-truth CSV path had `_grf_` literally baked in (would `FileNotFoundError` on a
  NAIP map id), mask direction hardcoded `>` with no LCB branch, and its result-directory-matching regex
  only recognized `map_grf_...` folders (tied to `hpc_sweep_common.py`'s own naming). Task list shows this
  was fixed this session — **verify the threshold value it now uses (should be 0.3, matching §1.1) and
  that the LCB direction matches whatever `gaussianprocesstraining.py`'s flag now is.**
- Many scripts read `MAPTYPE` from an env var with a hardcoded Python-level *default* string, split
  roughly evenly between defaulting to `"grf"` and defaulting to `"NAIP"` pre-session (see the earlier
  investigation's bucket A/B/C survey for the exhaustive list, not repeated here — grep `MAPTYPE` and
  `utility_threshold`/`UTILITY_THRESHOLD` across `scripts/` if a specific file's status needs checking).
  Task list item "Update MAPTYPE defaults to NAIP across single-map/compare scripts" is marked complete.

### 1.3 What was actually verified directly (highest confidence)

**`CMAES_classic_singlemap.py`**'s resulting state was shown directly via a system reminder after the
edits: `utility_threshold` default now `"0.3"`, `MAPTYPE` default now `"NAIP"`, `selected_map` default now
`11`, `WALLCLOCK_SECONDS` default now `"50"`, `ENFORCE_MIN_STEP_TIME` default now `"1"`. This one file's
post-edit state is confirmed correct and consistent with §1.1.

**Everything else in §1.2 is reported via the session's own task list (all 8 planned fix tasks marked
`completed`: verify current state → fix core LCB mechanism → fix
`DataCollector_3D_randomstart_CMAESregularized.py` → standardize threshold=0.3 → fix
`important_region_variance_from_trajectories.py` → fix `hpc_sweep_common.py` → update MAPTYPE defaults
across single-map/compare scripts → final verification sweep) but the literal diffs were not re-observed
by whichever agent is reading this file next. Given the user's explicit "do NOT make any mistakes"
instruction, the highest-value first action in a new session touching this code is a fresh verification
pass**: grep for any remaining `LCB = False`, `threshold.*0\.5`, or `MAPTYPE.*grf` defaults across the
files listed in §1.2 before trusting that the switch is fully consistent, rather than assuming the task
list's "completed" status is sufficient on its own.

### 1.4 NAIP map quality — which of the 65 maps are usable

Computed the fraction of grid cells with value `< 0.3` for every `map_{id}_NAIP_grid_counts.csv` (ids
0-64). Result has a clean natural gap, not an ambiguous middle: **3 maps (1, 9, 49) have a negligible
important region (3.26-3.85% of cells)** — exclude these. **All other 62 maps clear 10%+** (the next
lowest after the excluded three is map 34 at 11.01%), so nothing sits ambiguously in the requested
5-10% floor zone. "Good" map list = all ids except `{1, 9, 49}`.

Within those 62, the spread is large and non-uniform — worth knowing before picking maps for future runs.
A cluster of maps (**31, 36, 61, 39, 32, 56, 25, 38** — all >95% below-threshold) have almost the *entire*
map flagged as important, meaning very little actual search/discrimination problem for a planner to
solve. The more typical/moderate band (roughly 20-50% below-threshold — e.g. maps 0, 3, 4, 11-13, 15, 16,
19, 20, 27, 30, 42, 46-48, 51-54) exercises the "distinguish important from unimportant" capability more
meaningfully. Neither is wrong, but they test different things — near-100% maps ≈ "fly anywhere and win,"
moderate maps ≈ genuine informative-path-planning.

### 1.5 Open, explicitly-declined item: sensor noise is not scaled to NAIP's narrower range

`noise_model(altitude)` in `gaussianprocesstraining.py` (originally lines ~90-97) returns a **fixed
absolute measurement-noise variance** (0.01 at 10m altitude rising to 0.04 at higher altitude — std
0.10-0.20), completely independent of map type, never touched by the NAIP switch. Since NAIP's real value
range (~0.6 wide, §1.1) is much narrower than GRF's [0,1], this same absolute noise is proportionally much
larger relative to NAIP's signal — at high altitude the noise std alone is 30-70% of NAIP's entire value
range. Confirmed empirically against two real post-fix runs (`Vizualization/classic_map_NAIP_11_viz/`,
`Vizualization/classic_map_NAIP_32_viz/`): global RMSE trends down reasonably over ~100 timesteps, but
`occ_rmse` (occupied-region RMSE, the metric that actually matters for the thesis) is flat/noisy —
bouncing in a narrow band without much visible improvement — consistent with the SNR argument, not a
sign of a broken filter.

**User was asked whether to fix this (e.g. scale `noise_model()`'s variance to be proportional to each
map's own value range, keeping NAIP native/unrescaled per §1.1) and explicitly declined ("no it's
okay").** This is a deliberate, acknowledged decision to leave as-is, not an oversight — don't re-propose
it without new information. If it comes up again, the live framing question is: is noisier NAIP RMSE
*realistic* (real sensor data is genuinely noisier than a clean synthetic field — arguably shouldn't be
"fixed" to look nicer) or does it *undermine planner-vs-planner comparability* (a metric dominated by
measurement noise makes it harder to detect genuine planning-quality differences)? Left unresolved,
by choice.

## 2. Occupied-variance-AUC methodology, and its two headline findings

Built on top of `important_region_variance_from_trajectories.py`'s coarse-grid Kalman-replay helpers
(`build_coarse_grid`, `_important_mask_for_map`, `fov_grid_points`, `build_sensor_matrix`, `noise_model`,
`compress_shared_sensor_rows`, `_covariance_step` — this toolkit, reused directly rather than
reimplemented, has now been the standard post-hoc analysis approach across *two* sessions). The core
metric: replay a saved `executed_trajectory.csv` through the Kalman update to reconstruct the
occupied-region (ground-truth-masked) variance curve over real wall-clock time, then report either the
final value, the percentage drop from initial, or the wall-clock-time-averaged AUC (lower AUC = variance
stayed low for more of the flight, not just at the end).

**Scripts**: `analyze_map51_occupied_variance_auc.py`, `analyze_map200_occupied_variance_auc.py` (adapt
per map/run-tag; both write a CSV and print a table). LaTeX table versions exist too
(`results_map51_occupied_variance_auc_table.tex`, `results_map200_occupied_variance_auc_table.tex`).

**Finding 1 — standard GRF map 51 (pre-existing data, run-tag `mission300_steps3000`, 300s wall-clock
each)**: Diffusion wins decisively (AUC 3.764, final 0.526) over ImitateTrans (AUC 5.516) and CMA-ES
(AUC 8.495, worst). Directly supports the thesis's "diffusion matches/exceeds the expert" claim — **but
with an important caveat**: CMA-ES's replanning costs 10-60s per round vs. the learned planners' ~0.1-0.3s
under this session's earlier findings, so a fixed-300s-wall-clock comparison structurally advantages
whichever planner is cheaper per replan — this figure supports "diffusion achieves broader
real-time-budgeted coverage," not cleanly "diffusion plans better as an algorithm." Worth keeping that
distinction explicit in the thesis write-up (this exact critique was given to the user directly this
session re: the map-51 combined FOV+position heatmap figure).

**Finding 2 — synthetic four-corner map (map 200, new this session, see §3)**: the ranking **inverts** —
CMA-ES wins (AUC 1.933, best), ImitateTrans worst (AUC 3.460). Mechanistic explanation, visible directly
in the FOV heatmaps: CMA-ES's joint 8-waypoint planning routes efficiently between the four sparse,
far-apart informative corners, while the learned planners default to dwelling near the (non-informative)
center-start, likely because they were never trained on maps with information concentrated in disjoint
distant regions. This is a legitimate, useful stress-test finding — diffusion's advantage over the expert
planner is conditional on map structure resembling its training distribution, not universal.

## 3. Synthetic four-corner test map (map 200) and matched baseline-comparison figures

`csv/map_200_grf_grid_counts.csv` — new synthetic map, four 20×20 corner squares at value 0.6, background
0.1 elsewhere (484 of 2601 cells informative). Purpose: stress-test planners against sparse, spatially
disjoint informative regions rather than GRF's naturalistic spread-out fields.

Added `START_X`/`START_Y` env-var overrides (default unchanged at `4.0`, so no other script/session is
affected) to `CMAES_classic_singlemap.py`, `Diffusionplanner_singlemap.py`, `ImitateTrans_singlemap.py` —
all three ran from `(50,50)` (map center) for this test. **Note**: `CMAES_classic_singlemap.py` was
subsequently reused/reconfigured for the NAIP work (§1.3) and its defaults (map id, wallclock, MAPTYPE)
have since changed away from the map-200 test's settings — the `START_X`/`START_Y` mechanism itself is
still there and still defaults to 4.0, just not actively in use for the current NAIP default config.

Ran all three planners for 300s wall-clock each via `run_map51_planner_heatmaps.py --map-id 200
--run-tag corners_start5050` (this script is generic/reusable across map ids via CLI args — no
map-51-specific hardcoding despite the name). **Hit and fixed a real bug during this run**:
`Diffusionplanner_singlemap.py`'s own default `DIFFUSION_CHECKPOINT` pointed at
`scripts/checkpoints/current_best_updated.pth`, which doesn't exist — the correct file is at
`scripts/Diffusion/checkpoints/current_best_updated.pth` (confirmed via SHA-256 that these would have been
two genuinely different models, not just a naming mismatch, had the wrong one existed). Fixed the default
path; the bad first attempt's diffusion output was moved (not deleted) to
`diffusion_map_grf_200_viz_corners_start5050_WRONG_CHECKPOINT_bak/` and the run redone correctly.

Reconstructed a **position/time-spent heatmap** plotting script, `plot_map200_executed_position_heatmap.py`
— this style of figure existed pre-session (`Vizualization/map51_planner_heatmaps/map_51_planner_time_spent_heatmaps.png`
and its per-planner `*_time_spent_heatmap.csv` files) but its generating script no longer existed on disk;
rebuilt it by reverse-engineering the exact CSV schema (`x,y,time_spent_steps` — direct grid-cell
position-visit counting, **not** FOV-footprint-radius counting like the sibling FOV heatmap) and matching
the magma/grayscale visual convention. Both the FOV heatmap (`run_map51_planner_heatmaps.py`'s existing
`make_plot()`) and this reconstructed position heatmap were generated for map 200 and combined vertically
(FOV on top) via a small PIL script into `map_200_planner_heatmaps_combined.png` — same treatment done for
the pre-existing map-51 figures → `map_51_planner_heatmaps_combined.png`.

Also built matched 3D single-planner trajectory visualizations for the classical-baseline comparison
(lawnmower vs. greedy vs. full CMA-ES expert) on **map 53**, timesteps 0-747: `plot_lawnmower_3d_map53.py`,
`plot_cmaes_3d_map53.py`, `plot_greedy_3d_map53.py`. All three share styling (dataviz-skill categorical
palette, sequential-blue ground-truth floor, Gaussian-smoothed flight-path line since `dynamics_3d` snaps
executed positions to the 2m grid every timestep — real simulated behavior, not a plotting artifact) and
were made interactive (`plt.show()`, no forced `Agg` backend) per explicit request. Underlying trajectory
data: `Vizualization/lawnmower_map_grf_53_viz/`, `results_cmaes_viz_map53/`,
`Vizualization/greedy_map_grf_53_viz/`.

## 4. Diffusion multimodality / mode-collapse investigation (via `dataset_grf_60.pt`)

Open thread, **not finished** — left mid-investigation when the session pivoted to NAIP. Genuinely
interesting and probably worth returning to.

**Setup**: `dataset_grf_60.pt` is the training dataset behind the diffusion and ImitateTrans models.
Grouped by `condition_id` (identical starting pose/belief state), most conditions have exactly 1 saved
trajectory, but 753 of 3840 have exactly 2 (branching points where CMA-ES data collection kept two
meaningfully different candidate solutions — see the "Multimodal CMA-ES Data Collection" algorithm this
session also helped condense for the thesis, unrelated content-wise but same underlying data-collection
process). Test scripts: `test_multimodal_branch_grf60.py` (single condition, deep dive + plot),
`test_multimodal_branch_batch.py` (batch scan across N conditions, model loaded once, no plot),
`plot_batch_condition_scan.py` (bar-chart visualization of a batch scan's CSV).

**Confirmed facts**:
- ImitateTrans is **exactly deterministic** — max pairwise distance across repeated samples from the
  identical condition is `0.000000`, every time tested. Expected (no sampling noise in its forward pass)
  but good to have empirically nailed down rather than assumed.
- No seeding bug: `torch.manual_seed`/`RUN_SEED` is never set by either test script, and independently
  verified the diffusion samples collected are genuinely distinct (zero exact duplicates among 190
  pairwise comparisons per condition, real spread in pairwise distances) — the collapse findings below are
  real, not an artifact of accidentally reusing the same noise draw.
- **Diffusion's multimodality is condition-dependent, not reliable.** Tested 3 individual conditions
  (map 0 cond 3000000, map 1 cond 103001000, map 2 cond 200002000) with the `epoch_3000` checkpoint at up
  to 80 samples each: one showed genuine spread across both training-data branches, the other two showed
  **total collapse** — 0 of 80 samples ever landed closer to the "losing" branch than the "winning" one,
  with a clean gap between the two distance distributions (not sampling noise on a rare mode; ruled out
  explicitly by the 80-sample rerun).
- Ruled out unequal training-loss weighting as the cause: checked the dataset's own
  `variance_correction`/`RMSE_correction` per-branch weights for the collapsed conditions — nearly
  identical between the two branches in every case checked (e.g. 25.9 vs 25.3), so it's not that one
  branch was simply down-weighted in training.
- **Batch-scanned 20 conditions with the `epoch_1000` checkpoint** (less-trained): only 2/20 fully
  collapsed vs. 3/3 with `epoch_3000` — but this is **not** straightforwardly "less training preserves
  diversity better." Checked the actual distances: `epoch_1000`'s mean distance to its own "winning"
  branch (114.1) is *larger* than the branch-to-branch distance itself (106.6) — i.e. it isn't accurately
  reproducing *either* mode, just scattered broadly because it hasn't converged to anything precise yet.
  `epoch_3000`, by contrast, gets to ~44-47% of the branch-separation distance for its preferred mode —
  accurate, but committed to one mode only. **Real tradeoff, not a free lunch**: more training buys
  accuracy at the cost of mode-coverage, consistent with a standard per-sample denoising-MSE loss having
  no explicit incentive to preserve multimodality. Weak correlations checked in the batch (not dominant):
  winning branch has lower `parent_beam_index` (the original, non-diversity-regularized CMA-ES solution)
  in 14/20, higher training weight in 14/20, lower `RMSE_correction` in only 7/20 (no real signal there).
- **Diffusion checkpoint currently configured** in `Diffusionplanner_singlemap.py`'s `diffusion_path`
  default: as of the last point this was directly touched, it was `sparse_trans_waypoints_epoch_1000.pth`
  (switched from `epoch_3000.pth` mid-investigation, at the user's request, to run the batch scan in §4).
  **This has not been re-verified since** — a future session should check the current default before
  assuming which checkpoint is active, especially since it's irrelevant to the CMA-ES-only NAIP runs done
  afterward (§1) but very relevant if resuming diffusion-planner work of any kind. Both `epoch_1000.pth`
  and `epoch_3000.pth` exist on disk (`scripts/Diffusion/checkpoints/`); training was continued via a
  SLURM job (`diffcont.sh`, calling `continue_sparse_trans_training.py --checkpoint ... --target-epochs
  3000`) — the existence of `epoch_3000.pth` on disk confirms this continuation succeeded, though whether
  `continue_sparse_trans_training.py` itself was fully authored during this session's visible history is
  unclear from the transcript alone (the checkpoint-loading/resuming logic in
  `threeDSparseTransDiffusion_periodic_eval.py`'s `train()` was read and understood — checkpoint dict keys
  are `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `loss`,
  `epoch_train_loss` — but the resulting continuation script's own content wasn't reconfirmed later).
  **Worth checking `scripts/continue_sparse_trans_training.py` exists and is correct before assuming it's
  ready to reuse for further continued training.**

**Natural next step if resuming this thread**: pick a well-separated condition and plot it with whichever
checkpoint is currently active (the previous session was about to do exactly this — pick from the 20-map
batch-scan candidates by combined score of low win-fraction + low distance-ratio — see
`results_multimodal_branch_test/batch_condition_scan.csv` for the ranked candidates) to visually confirm
the accuracy/mode-coverage tradeoff on a specific, illustrative example rather than just aggregate
statistics. An intermediate checkpoint (~2000 epochs) was also proposed as informative (find where the
tradeoff crosses over) but never run.

## 5. Smaller items

- Condensed an overly-long LaTeX algorithm block ("Multimodal CMA-ES Data Collection with
  Diversity-Regularized Refinement") for the thesis — replaced inline piecewise-math and a full
  diversity-penalty equation with prose descriptions, collapsed nested loops. Given directly in
  conversation, not saved to a file — copy from transcript into the thesis document if not already done.
- Finished a thesis paragraph on the greedy one-step planner's primary drawback vs. the full 8-waypoint
  expert (no cross-step lookahead / cost-benefit reasoning across a sequence of moves, compounded by a much
  shorter execution horizon — 5 steps vs. 40 — forcing ~8x more frequent replanning), grounded directly in
  `realgreedy_singlemap.py`'s actual `planning_horizon=1`/`execution_chunk=5` defaults. Also given directly
  in conversation, not yet copied into the thesis document.
- Reported the user's laptop specs for thesis methodology reporting: HP EliteBook 840 14" G9, Intel Core
  i7-1255U (10C/12T, 1.7GHz base), 32GB RAM, integrated Intel UHD Graphics (no discrete GPU — confirmed
  `torch.cuda.is_available()` is `False`), Windows 10 Pro. Relevant since several tables this session
  report wall-clock timings that are implicitly CPU-only.

## 6. Priority-ordered next steps

1. **Verify the NAIP fix (§1.3)** — don't trust the task-list-says-completed status blindly; grep for
   stray `LCB = False` / `threshold` defaults still at 0.5 / `MAPTYPE` still defaulting to `grf` across the
   files listed in §1.2, given the user's explicit "do not make any mistakes."
2. Decide whether to actually run new NAIP data collection with
   `DataCollector_3D_randomstart_CMAESregularized.py` now that the switch is believed complete — this was
   the whole point of the NAIP thread and hasn't happened yet as of this writing.
3. If continuing the multimodality thread (§4): confirm which diffusion checkpoint is currently active,
   then pick and plot a specific well-separated condition from the existing batch-scan ranking, and/or run
   the proposed ~2000-epoch intermediate checkpoint to locate the accuracy/mode-coverage crossover.
4. Copy the two thesis-writing items from §5 (condensed algorithm, greedy-drawback paragraph) into the
   actual thesis document — both currently exist only in this session's transcript.
5. The sensor-noise/SNR question (§1.5) is explicitly closed for now (user declined the fix) — don't
   revisit without new information prompting it.
