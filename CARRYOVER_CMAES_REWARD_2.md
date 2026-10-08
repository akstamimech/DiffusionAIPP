# Carryover — CMA-ES objective bugfixes, multi-map hyperparameter sweeps, DataCollector changes, thesis writing

Written for a future agent with no memory of this session. Follows on from `CARRYOVER_CMAES_REWARD.md`
(the "Popovic comparison, mask staleness, beta/sigma/noise tuning" carryover), which this session opened
by loading and verifying. That file's own priority list (§8) is now largely superseded by what happened
in *this* session - read this file first, only fall back to the old one for background on the
mask-staleness/non-submodularity material, which is still accurate and not repeated here.

## 0. Repo/git state - READ THIS FIRST

Repo: `C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts`, branch `prototype`.

**No commits were made this entire session.** Everything below is uncommitted working-tree state.
`git status` is dominated by a large pre-existing pile of unrelated modified/deleted files from other
work threads (DPPO, synthetic map pipeline, classic_batch_metrics/ deletions) - never touched this
session, not discussed further here, exactly as the previous carryover already noted.

**Files this session actually modified (tracked, currently showing `M`):**
- `gaussianprocesstraining.py` - the big one, see §1-3. Four independent fixes landed here:
  low-tier lattice densification, CMA-ES cost normalization, CMA-ES boundary penalty realignment,
  `kalman_update` Cholesky-alignment, plus `CMA_STEP_SIZE_XY`/`CMA_STEP_SIZE_Z` made env-configurable.
- `CMAES_classic_singlemap.py` - added `CMAES_REFINE` env toggle (0 = skip CMA-ES, execute raw grid
  search output only, for an isolated lattice-only baseline) and imported `clip_waypoints_continuous_3d`.
- `Diffusion/threeDSparseTransDiffusion_periodic_eval.py` and `ImitateTrans_periodic_eval.py` - periodic
  eval's `variance_drop`/`final_variance` now measures ground-truth *occupied-region* variance
  (`true_map_flat > EVAL_UTILITY_THRESHOLD`), not global/unmasked variance. Same dict keys, so no
  downstream consumer (logging, CSV, plots) needed changes - only what the numbers measure changed.

**Untracked, new this session, not yet git-added:**
- `DataCollector_3D_randomstart_CMAESregularized.py` - **still not tracked in git at all**, exactly as
  flagged mid-session (§7 of the previous carryover raised this too - it's now been directly edited
  four separate times this session, see §7 below, and STILL isn't committed). If anything happens to
  the working tree, this file's entire edit history (execution_chunk split, RANKLIM, uniform start,
  xyz diversity) is at risk. `git add` + commit this before anything destructive happens near it.
- `realgreedy_singlemap.py` - untracked, not touched this session, not investigated, mentioned only
  because it showed up in status.
- A large number of new sweep/analyze/smoketest scripts (all in `scripts/`, not `scripts/Diffusion/`):
  `sweep_cmaes_beta_multimap.py` (extended this session with repeats + `ProcessPoolExecutor` parallel
  execution - was sequential, single-repeat before), `sweep_cmaes_maxiter_multimap.py`,
  `sweep_cmaes_stepsize_multimap.py`, `sweep_cmaes_lattice_check.py`, `sweep_cmaes_stepsize_map55.py`,
  and one `analyze_*.py` counterpart per sweep (`analyze_cmaes_beta_sweep.py`,
  `analyze_cmaes_beta_sweep_multimap.py`, `analyze_cmaes_lattice_check.py`,
  `analyze_cmaes_stepsize_sweep_multimap.py`, `analyze_beta_sweep_maps51-55.py` +
  `analyze_beta_sweep_maps51-55_combined.py`, `analyze_maxiter_sweep_maps51-55.py`,
  `analyze_stepsize_map55.py`), plus `consolidate_first_40_maps.py` (standalone dataset-chunk
  concatenation utility for the DataCollector's chunk output, deliberately dependency-light - only
  needs `torch`), and three `smoketest_*.py` diagnostic scripts (see §7).
- Many `results_cmaes_*/` directories holding raw run outputs + `analysis/` subfolders (CSVs, PNGs,
  `.tex` tables) from every sweep this session ran. **None of this is committed or backed up anywhere
  else** - `results_cmaes_beta_sweep_maps51-55/`, `results_cmaes_maxiter_sweep_maps51-55/`,
  `results_cmaes_stepsize_map55/`, `results_cmaes_beta_sweep_multimap/`,
  `results_cmaes_stepsize_multimap/`, `results_cmaes_lattice_check/` are the ones with genuine
  standing findings worth keeping; the others (`results_cmaes_beta_sweep_threshold0.4/`,
  `results_cmaes_beta_sweep_fixed_objective_maxiter20/`) were intermediate/superseded runs from
  earlier in the session, safe to delete if disk space matters.
- `stash@{0}` is unchanged from before this session (`autoregressive-mask CMA-ES objective` shelved
  work) - never touched, still fully recoverable via `git stash pop`, still not re-justified needed
  given what §1-2 below found (the real bug was cost normalization, not mask staleness - see §1).

## 1. Headline finding: the CMA-ES objective was missing Popovic's own cost term - now fixed

The single most consequential thing that happened this session. The previous carryover's "mask
staleness" mechanism (§1 of that file) was real and mechanistically confirmed, but turned out **not**
to be the primary cause of CMA-ES underperforming the grid search it's supposed to refine. The actual
cause, found by fetching Popovic et al. (2020)'s real MATLAB source
(`github.com/marija-p/mav_ipp`, `mav_ipp_sim/src/MATLAB/tools/planning/`), not just re-reading the
paper text: **their CMA-ES fitness function divides information gain by travel cost**
(`compute_objective.m`: `cost = max(trajectory_total_time, 1/measurement_frequency); obj = -gain/cost
+ penalty`), exactly mirroring their own grid search's cost-benefit ratio (`search_lattice.m`:
`obj = -gain/cost`). This codebase's `trajectory_objective_3d` had **no cost term at all** - pure
`variance_reduction`, unnormalized by distance. This was previously (wrongly) "confirmed" as matching
Popovic's design in the earlier carryover, based on reading the paper's prose (§5.4), which indeed never
mentions a cost term - the source code does, and the paper's prose is incomplete/misleading here.

Fix landed in `trajectory_objective_3d` (`gaussianprocesstraining.py`): computes total path distance
from start through all control waypoints (same style as `build_spline_trajectory_3d`'s own arc-length
calc), divides `variance_reduction` by `max(total_distance, step)`, mirroring Popovic's `gain/cost`
exactly (their cost = time; this codebase has no velocity model anywhere, so distance is the natural
analogue, and is what the grid search's own cost term already uses).

**This fix alone did not fully resolve the underlying pathology** (CMA-ES with more iterations getting
worse) - see §2 for the second fix that was also needed.

## 2. Second fix: boundary penalty was a flat -1000 constant, not Popovic's continuous overshoot ramp

Also found via the real MATLAB source, same investigation. Popovic's boundary penalty
(`compute_objective.m`) is a continuous ramp - `max(overshoot_x, overshoot_y, overshoot_z, 0)`,
summed over every *sampled measurement point along the flown path* (`points_meas`), not just the
waypoint vertices, and accounts for the sensor footprint extent (`submap_edge_size_env`), not just
vehicle position. This codebase's `trajectory_objective_3d` had a flat `-1000` penalty, applied only to
raw (pre-clip) waypoint vertices, giving CMA-ES zero gradient once a candidate was flagged infeasible -
any violation, tiny or huge, looked equally bad.

Fixed: replaced the flat constant with a `_boundary_overshoot()` helper computing the same
continuous max-overshoot-or-zero ramp, applied to *both* the raw control waypoints (catches wildly
out-of-bounds CMA-ES proposals before the hard clip absorbs them) and the sampled spline path (catches
a spline bowing outside the domain between two already-in-bounds waypoints - the clip can't see this
since it only touches the waypoint vertices).

## 3. Densified the low-altitude lattice tier (5x5 -> 9x9) - real but not sufficient on its own

Separately (before the cost-norm fix was even found), quantified that the grid search's low-altitude
lattice tier was genuinely under-resolved: at z=10m the FOV footprint side (~11.5m) is about half the
tier's inter-candidate spacing (~23m at the old 5x5 density), a real, verified ~2x coverage gap -
verified via the actual tiling-contiguity criterion (square lattice + axis-aligned square footprint
tiles gap-free iff spacing <= footprint side), not guesswork. Middle/top tiers already met or exceeded
this criterion (top tier's footprints actually overlap, by design, matching Popovic's own "sparser at
top due to increasing FoV" - not an oversight). Densified only the low tier, 5x5->9x9 (`build_pyramid_lattice_3d`).

**Important negative result along the way**: densifying the lattice alone, *before* finding the cost-norm
bug, did **not** fix CMA-ES underperforming the lattice-only baseline - this directly falsified the
"sparse grid search causes the pathology" hypothesis and is what motivated actually checking Popovic's
source code instead of continuing to guess. Keep this densification (it's independently justified on
its own tiling-math merits, and is now baked into every run), but don't expect it alone to explain any
future CMA-ES weirdness - the objective/penalty fixes in §1-2 were the load-bearing ones.

## 4. `kalman_update` aligned to Cholesky-first/pinv-fallback (perf-only, no behavior change)

Survey of every matrix solve in `gaussianprocesstraining.py` found 3 of 4 Kalman-style covariance
update functions already used `scipy.linalg.solve(..., assume_a="pos")` first with `np.linalg.pinv`
only as a `LinAlgError` fallback (faster, appropriate for a guaranteed-SPD system) - `kalman_update`
(the function that applies *real* executed measurements, called every timestep) was the one holdout,
always using `pinv` directly. Fixed to match the same pattern. Verified numerically identical output
(~1e-14 diff, floating-point noise) before and after on a synthetic test, and the real pipeline still
runs clean. No re-sweep needed for this one - pure performance, zero behavior change. Also confirmed,
while investigating this, that no C++ Eigen library is used anywhere in the codebase (NumPy/SciPy only,
backed by OpenBLAS 0.3.29) - eigendecomposition genuinely only happens inside `pycma`'s own internals
(`np.linalg.eigh`, verified directly in the installed package source), which is correct/necessary there
since that's literally what "Covariance Matrix Adaptation" requires.

## 5. Multi-map sweep results - what's actually settled vs. still open

**Hard-won methodology lesson, confirmed repeatedly this session, apply to every future hyperparameter
question**: single-map results are unreliable. Beta looked like a clean win for one value on map 91
alone (post-fix: beta=8.0, final=1.92 vs next-best 2.14) and completely evaporated into map-to-map noise
once tested on 5 maps (91-95): the spread between beta *means* was ~0.15 while the per-beta *std across
maps* was ~0.5-0.77 - a 3-5x noise-to-signal ratio. The exact same thing happened independently to the
CMA-ES step size (1.5/1.2 default vs. 10/4 "Popovic-scaled" target, see below). Do not trust any
single-map sweep result in this codebase without at least a 5-map check.

**Settled**: the CMA-ES-vs-lattice-search comparison itself (map 91, post both fixes): both
maxiter=2 (2.37 final) and maxiter=20 (2.50 final) now clearly beat lattice-only (2.81) - this is the
one finding that's actually robust and was the real point of the whole cost-norm/penalty investigation.

**Not settled / actively contradictory, needs more data**: beta and CMA-ES step size both show
"no robust winner across 5 maps" *and* "clean-looking monotonic trend on one map" depending on which
data you look at:
- Beta, 5 maps (91-95), maxiter=20, 3 repeats: no map-independent winner (map 51->0.5, 52->2.0,
  53->0.1, 54->3.0, 55->0.1, all different) - see `results_cmaes_beta_sweep_multimap/`.
- Beta, 5 maps (**51-55**, different map range), maxiter=20, 3 repeats, TIMEALLOTED=200: same story,
  every map has a different winner (51->0.5, 52->2.0, 53->0.1, 54->3.0, 55->0.1 on final variance) -
  see `results_cmaes_beta_sweep_maps51-55/analysis/` for the LaTeX table + combined
  final-variance/AUC bar chart figure (`beta_sweep_maps51-55_combined_final_and_auc.png` - two-panel
  figure, y-axis floors set to 2/4 respectively per user request, red winner-dot correctly centered
  in the bar after fixing a real `matplotlib` bug, see §6).
- CMA-ES step size (1.5/1.2 current default vs. 10/4 "Popovic-scaled" target - derived by scaling
  Popovic's own best (3,3,4)m step in their 30x30m domain to this codebase's ~92m/30m workspace), 5
  maps (91-95), maxiter=20: no robust winner either (final_mean 2.58 vs 2.53, effect an order of
  magnitude smaller than the ~0.53 map-to-map std) - see `results_cmaes_stepsize_multimap/`.
- **But then**: single-map (55), single-run, 5-point step-size sweep ((1.5,1.2), (6,4.8), (10,4),
  (20,8), (40,16)), maxiter=20, TIMEALLOTED=200 - see `results_cmaes_stepsize_map55/` and
  `analyze_stepsize_map55.py` - shows a real, clean, roughly-monotonic result: **the current sub-pixel
  default (1.5/1.2) wins on final variance (3.30, best) and wall-clock-integrated AUC (10.24, best),
  and 40/16 is clearly worst on every metric (63.81% reduced vs 79.93% for the default)**. On
  timestep-indexed AUC specifically, though, 10/4 narrowly wins (8.23 vs 8.67) - a genuine 3-way split
  depending on which metric you read, not resolved. This is one map, one run each - **explicitly
  flagged as needing multi-map validation before treating as a real finding**, not yet done.
- Maxiter, 5 maps (51-55), {2,8,20,45}, 3 repeats, TIMEALLOTED=200, beta=1: per-map winner is mostly
  maxiter=45 (3/5 maps) but not universally (51->8, 54->20). **The genuinely decisive result here is
  the quality-vs-wall-clock-time tradeoff**, not per-map final variance: once you integrate the AUC
  over the run's *actual real wall-clock time* (not simulation timestep - see the methodology
  correction below) rather than timestep index, **maxiter=2 wins on every single map and overall**
  (mean AUC 9.02 vs 9.55/10.17/10.88 for 8/20/45) - more than double the compute time (251.0s vs 99.3s)
  buys essentially nothing once time-efficiency is measured correctly. See
  `results_cmaes_maxiter_sweep_maps51-55/analysis/` for both LaTeX tables.

**Methodology correction made mid-session, keep applying it**: "AUC" was initially computed by
integrating over *timestep index* in several of the maxiter/step-size analysis scripts (a
copy-paste inheritance from earlier scripts where every run shared the same wall-clock behavior). This
is wrong whenever the swept parameter changes real per-replan compute cost (maxiter obviously does;
step size does not, since it doesn't change how many CMA-ES fitness evaluations run) - a timestep-axis
AUC silently treats a "tick" as equally weighted regardless of how expensive it was to compute in real
seconds, which defeats the entire purpose of a quality-vs-time comparison. Always check which axis
(`row[0]`=timestep vs `row[1]`=wall_time_seconds in `executed_trajectory.csv`) an AUC calculation
integrates over before trusting it for a time-efficiency claim - this isn't automatically "the earlier
wall-clock-AUC confound" caution from the previous carryover (that one was about *aligning* a shared
clock across runs of different length; a per-run self-normalized wall-clock AUC, as used here, is a
different and legitimate thing once time-efficiency itself is the question being asked).

## 6. Reusable tooling/patterns from this session

- **8-way parallel sweep execution works cleanly on this machine** (10 physical/12 logical cores) -
  `sweep_cmaes_beta_multimap.py`/`sweep_cmaes_maxiter_multimap.py` now use
  `concurrent.futures.ProcessPoolExecutor(max_workers=8)`, submitting `_run_one` calls concurrently.
  Safe because `hpc_sweep_common._run_one` already sets `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` per
  subprocess (a fix from a previous session's "18x slowdown from BLAS oversubscription" lesson) - each
  worker only ever touches one core for its own math, so N workers <= N cores doesn't self-contend the
  way naive parallelism would. Observed ~2.5-3x slower per-run wall time under 8-way contention vs. the
  sequential calibration baseline, but still a large net win overall (75-run beta sweep: ~45min
  sequential estimate vs. actual ~15min wall-clock).
- **Matplotlib gotcha, real bug, cost real time**: `ax.bar(x, values, width=w)` defaults to
  `align="center"` (bar centered at `x`), not `align="edge"` (bar starts at `x`). Manually computing a
  marker's x-position as `x + i*bar_width + bar_width/2` assumes edge-alignment and silently produces a
  marker on the bar's right edge instead of its center if you forgot to pass `align="edge"` explicitly.
  Cost two iterations to find/fix in `analyze_beta_sweep_maps51-55*.py` - default to always passing
  `align="edge"` explicitly whenever computing bar-relative marker positions by hand.
- `important_region_variance_from_trajectories.py`'s coarse-grid replay helpers
  (`build_coarse_grid`, `_important_mask_for_map`, `fov_grid_points`, `build_sensor_matrix`,
  `noise_model`, `compress_shared_sensor_rows`, `_covariance_step`) remain the standard toolkit for
  every post-hoc ground-truth-masked analysis this session - reused directly (via
  `import important_region_variance_from_trajectories as m`) rather than reimplemented, in every
  `analyze_*.py` script listed in §0. Keep doing this rather than rewriting the replay logic.
- `consolidate_first_40_maps.py` (in `scripts/`, not the chunks subfolder) - standalone, only needs
  `torch`, concatenates `DataCollector_3D_randomstart_CMAESregularized.py`'s per-map chunk files for a
  configurable map range (`START_MAP`/`MAP_COUNT` constants at top). Reports missing maps rather than
  erroring, safe to run against a still-in-progress collection job. `DELETE_CHUNKS_AFTER=False` by
  default (safety).

## 7. DataCollector changes - four separate edits this session, still uncommitted (see §0)

All in `DataCollector_3D_randomstart_CMAESregularized.py`. This resolves the "dangling, unresolved
request" flagged at the end of the previous carryover (§7 there) - the user did eventually specify what
changes were wanted:

1. **`execution_chunk` split into `EXECUTION_CHUNK_SCORING=40` / `EXECUTION_CHUNK_UPDATING=10`**.
   Previously a single `execution_chunk=40` value was used both to decide how far to simulate forward
   for scoring branches AND how far the chain actually advances - and since
   `planning_horizon*samples_per_segment+1=41` and `execution_chunk` was 40, the collector was executing
   ~98% of every planned trajectory before replanning (not meaningfully receding-horizon at all).
   Now `execute_refined_candidate` checkpoints the belief state at 10 steps into the rollout (same rng
   draws, same path - causally consistent, not two independent simulations) while continuing to
   simulate to 40 for scoring purposes. Returns both `final_mu`/`final_P` (scoring, unchanged keys) and
   new `update_cx/cy/cz`/`update_mu`/`update_P` (what the chain actually commits to).
   `run_chain_and_record` now advances using `winner["update_*"]` instead of `winner["final_*"]`.
   Verified via a direct numeric check: round 1's baseline masked variance jumped from 23.86 (old,
   ~full-rollout advance) to 84.99 (new, 10-step advance) on an identical smoke test, confirming the
   shorter commitment is genuinely taking effect, not just structurally present.
2. **`RANKLIM` 5 -> 36** (more rounds per chain, to compensate for the now-shorter per-round advance).
3. **`gaussian_grid_start` -> `uniform_grid_start`**: start positions now drawn uniformly across the
   full margin-inset domain instead of `Normal(center=(50,50), std=20)`, which under-sampled
   edge-adjacent starts. Verified via a 2000-sample synthetic check: 35% of draws now land within 10
   units of an edge, vs. roughly half that under the old Gaussian sampler. `START_POSITION_CENTER`/
   `START_POSITION_STD` constants removed (no longer meaningful).
4. **`trajectory_xy_mse` -> `trajectory_xyz_mse`**: the CMA-ES diversity-regularization distance metric
   (used to penalize a regularized CMA-ES variant for landing too close to an earlier solution from the
   same round) now includes altitude, not just xy footprint - per explicit instruction ("altitude
   diversity certainly counts"). `CMA_DIVERSITY_DISTANCE_THRESHOLD=20` is unchanged in value, but its
   *meaning* shifted slightly now that z is folded into the same mean-squared-error computation - **not
   re-measured against the xyz metric** (the calibration numbers below predate this specific change,
   were measured under the xy-only version).

**Diversity-threshold calibration findings** (measured before the xyz change above, still informative,
not yet re-measured after it): `CMA_DIVERSITY_DISTANCE_THRESHOLD=20` does not literally enforce a 20m
separation - because `trajectory_xy_mse` pools per-axis squared differences via a single `np.mean` over
both the point and coordinate axes rather than computing per-point squared-distance-then-averaging, the
actual enforced RMS xy separation is `sqrt(2*20^2) ≈ 28.3m`, a real, non-obvious factor-of-sqrt(2)
discrepancy between the named constant and what it does. Measured real achieved separations across 5
rounds/15 pairwise comparisons (map 91, default step size 1.5/1.2, maxiter=45 - the DataCollector's own
unset default): all 15 cleared the ~28.3m threshold, but 4 of 5 rounds had their tightest pair within
~1m of the wall - consistent with a hard-cutoff penalty whose pressure vanishes exactly at the boundary,
so the search finds "just enough" separation and stops, rather than a step-size limitation (step size
was confirmed not to be the bottleneck here, since escape happened reliably; CMA-ES's own covariance
adaptation over the 45-iteration budget was sufficient starting from the same tiny 1.5m sigma0).

**Genuinely important, unresolved finding from this thread**: checked whether a diversity-regularized
CMA-ES variant ever actually *wins* (achieves higher realized masked-variance-reduction than the plain
"original" unregularized pass) rather than just being kept as a worse-but-tolerated alternative. Answer:
**yes, decisively** - across 15 rounds (map 91), win counts were a perfectly even 5/5/5 split between
`cma_original`/`cma_regularized_1`/`cma_regularized_2` (regularized variants win 67% of the time
combined). This substantially updates the earlier-session skepticism that the diversity mechanism was
"just manufacturing artificially-diversified worse variants" - it isn't; it's functioning as a genuine,
effective multi-seed ensemble that frequently finds a *better* solution than a single CMA-ES pass, not
just a different one. **Open, unresolved question flagged but never tested**: is this benefit coming
specifically from the closeness penalty pushing the search somewhere better, or simply from running 3
independently-seeded CMA-ES passes instead of 1 (an ensemble/multi-restart effect any 3 seeds would
give, penalty or not)? Would need a comparison against 3 *plain* (non-regularized) re-seeded CMA-ES
passes to disentangle. This also surfaces a broader implication never acted on: if CMA-ES is this
seed-sensitive at the current maxiter/popsize budget, every *other* planner script in this codebase
(`CMAES_classic_singlemap.py`, `realgreedy_singlemap.py`) that runs only a single CMA-ES pass per replan
may be regularly settling for a mediocre local optimum that a second differently-seeded attempt would
beat - not yet investigated outside the DataCollector's specific 3-variant structure.

`smoketest_cmaes_datacollector_multimodal.py`, `smoketest_diversity_distance_calibration.py`, and
`smoketest_variant_winner_check.py` are the three standalone diagnostic drivers built for this thread -
none of them import the DataCollector module directly (it executes real dataset-loading code at import
time, needs a training `.pt` file not present in this environment) - instead they call
`dc.run_chain_and_record`/`dc.simulate_candidate` etc. directly after manually replicating the small
slice of setup (`initialize_gp`, `load_map`, `warmup_rollout`) needed. Reuse this pattern for any future
DataCollector-behavior diagnostic rather than trying to import the whole module.

## 8. Thesis writing done this session (not code, but substantial)

Drafted and iteratively corrected, in conversation, not as saved files - **none of this is saved
anywhere except the chat transcript, worth copying into the actual thesis document if not already
done**:
- "Expert planner framework" section: Sequential Grid Search subsubsection (Algorithm 1 pseudocode,
  utility/cost equations, submodular-greedy comparison) and Trajectory refinement via evolutionary
  optimization subsubsection (CMA-ES description, warm-starting, boundary handling, step-size
  hyperparameter discussion).
- Correction to that draft: the CMA-ES subsection initially omitted the cost-normalization term
  (ironic, given §1 above), fixed with an explicit sentence once the omission was caught.
- Why the framework's lack of an optimality guarantee is acceptable: warm-starting reduces the risk of
  the missing guarantee mattering, receding-horizon replanning means a bad round isn't locked in, the
  underlying problem is intractable regardless of method so no alternative could offer a guarantee
  either, and the right validation standard is empirical (beats simpler baselines within the same time
  budget) rather than theoretical.
- Concrete example of why Popovic's objective isn't submodular (built around the mask-eligibility-shift
  mechanism already established in the previous carryover, reused here as the clearest illustration -
  a cell's marginal value can *increase* rather than diminish if an earlier measurement pushes its mean
  estimate across the interest threshold).
- Two paragraphs specifically justifying the beta=1 and step-size (1.5/1.2) defaults, built directly
  around the multi-map sweep findings in §5 above (no robust winner across maps -> no principled reason
  to deviate from the conventional/existing default). **Note**: the step-size paragraph was drafted
  *before* the single-map map-55 5-point sweep (§5) complicated the picture - if this paragraph is used
  in the thesis, it should probably be revisited in light of that map-55 result once/if it's validated
  across more maps.
- Verified and flagged a small draft error: an early paragraph said "averaged over five runs" per
  (map, beta) configuration when the actual sweep used `REPEATS_PER_BETA=3` - not yet confirmed whether
  the user's actual thesis draft still has this error.

## 9. Other findings, smaller

- **One-hot position-marker encoding verified correct**: `threeDSparseTransDiffusion.py`'s
  `make_position_marker_maps` (and 5+ byte-identical duplicate copies across `SparseDiffusion.py`,
  `SparseTransDiffusion.py`, `ImitateTrans*.py`, `threeDSparseTransDiffusion_periodic_eval.py`) is
  correct: scale (`grid_pos = (pos/100)*50`) and orientation (`marker[b,0,y_idx,x_idx]`, matching
  `initialize_gp`'s own `meshgrid(xs,ys)` row=y/col=x convention) both checked via a 6-point round-trip
  test, all exact. `marker_radius` parameter is accepted but never used in the function body anywhere -
  every copy produces a literal single-pixel one-hot despite the name implying a small marked region -
  consistent dead parameter across the whole codebase, not a one-off bug.
- Periodic eval scripts (§0) will not retroactively affect an already-running training process (Python
  doesn't hot-reload); takes effect on next restart of either script.

## 10. Priority-ordered next steps, if picking this thread back up

1. **Commit `DataCollector_3D_randomstart_CMAESregularized.py`** (and ideally the other new sweep/
   analysis scripts) - it's held four sessions' worth of real edits while remaining completely untracked
   in git. This is the single highest-risk item in this carryover.
2. Resolve the beta/step-size 3-way split described in §5: the map-55 single-run sigma sweep and the
   5-map (51-55) beta sweep both show real, non-trivial per-map/per-metric variation that hasn't been
   fully reconciled - either commit to "no single value wins, use the conventional default" (matches
   what's already drafted for the thesis, §8) or run a proper multi-map version of the 5-point step-size
   sweep to see if the map-55 trend (small step size wins on final/wall-clock, mid-range wins on
   timestep-AUC) replicates.
3. Disentangle the diversity-regularization ensemble-vs-penalty question from §7 (3 plain re-seeded
   CMA-ES passes vs. 3 diversity-regularized ones, same belief states) - directly relevant to whether
   every other single-pass CMA-ES planner in this codebase is leaving easy performance on the table.
4. Existing collected DataCollector dataset (`CMAES_beamsearch_dataset_3d_synthetic_final.pt`, dated
   before this session) reflects the pre-fix CMA-ES objective entirely - still not decided whether to
   re-collect from scratch under the now-fixed pipeline, flagged repeatedly, never actioned.
5. `git stash pop` the autoregressive-mask work remains available but now looks *less* necessary than
   before this session - the cost-normalization fix (§1) resolved the maxiter-getting-worse pathology
   that motivated it, without needing progressive mask recomputation. Worth explicitly deciding to drop
   this stash rather than continuing to carry it forward as a live option, unless a new symptom
   specifically pointing at mask staleness (rather than cost/penalty issues) turns up.
6. Copy the thesis paragraphs drafted in conversation (§8) into the actual thesis document if not
   already done - none of it is saved as a file anywhere.
