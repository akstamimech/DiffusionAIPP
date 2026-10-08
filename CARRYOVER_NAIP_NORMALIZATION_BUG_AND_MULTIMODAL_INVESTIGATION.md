# Carryover — NAIP switch completion, the DIFFUSION_DATASET_PATH normalization bug, execution_chunk
# fix, and the multimodal-vs-unimodal training data investigation

Written for a future agent with no memory of this session. This was one continuous, very long session
covering (1) finishing the GRF->NAIP switch across the remaining planner scripts, (2) discovering and
fixing a severe, session-spanning normalization bug that was silently corrupting every diffusion/
ImitateTrans inference run, (3) a real but smaller execution-horizon mismatch fix, and (4) an
in-progress investigation into whether the multimodal branch-collection scheme in
`DataCollector_3D_randomstart_CMAESregularized.py` is hurting diffusion's performance on NAIP. Related,
not superseded: `CARRYOVER_GRF_SWITCH_DPPO_REWARD_AND_INFOGRAPHICS.md` (earlier GRF-era work, different
topic), `CARRYOVER_NAIP_SWITCH_AND_BASELINES.md` and `CARRYOVER_CHECKPOINT_SWEEPS_DIFFUSION_VS_IMITATETRANS.md`
(earlier NAIP-switch sessions — the file-edited-outside-session warning in both is doubly true now;
re-verify every "current default" claim below before trusting it).

## 0. The single most important thing in this file

**Section 5 (the normalization bug) is the highest-value finding of this entire session.** Every
diffusion/ImitateTrans inference run performed anywhere in this project, across every prior session,
was silently using the wrong normalization scale for its belief-map conditioning inputs, because
`DIFFUSION_DATASET_PATH`/`IMITATE_DATASET_PATH` was never set and defaulted to a different, non-NAIP
dataset. This was fixed by changing the code defaults (not just an env var workaround), and the fix is
now proven correct via direct evidence (loss-recomputation test in §5.3), not just inferred from
downstream metrics. **If you are about to run any diffusion or ImitateTrans inference and get
suspiciously bad results, check this first** — verify `threeDSparseTransDiffusion.py`'s and
`ImitateTrans.py`'s `DATASET_PATH` default actually still points at `FINAL_NAIP_DATASET.pt`, since
these files get edited outside sessions repeatedly.

## 1. NAIP switch — now believed complete across the single-map / periodic-eval scripts

Confirmed edited and NAIP-configured this session (LCB=True convention: `MAPTYPE="NAIP"`,
`UTILITY_THRESHOLD` default `"0.3"`, prior mean `utility_threshold - 0.1` active):

- `gaussianprocesstraining.py` — `LCB = True` (module-level, shared by everything that imports
  `importance_filter`/`compute_reconstruction_rmse`-adjacent code); `initialize_gp()`'s defaults are
  `sigma2=0.0079, lengthscale=4.78` (see §2).
- `evalmetrics.py` — its own separate, unsynced `LCB = True` (must move in lockstep with the one above;
  no shared import, this has bitten the project before).
- `CMAES_classic_singlemap.py`, `Diffusionplanner_singlemap.py`, `ImitateTrans_singlemap.py`,
  `lawnmower_singlemap.py` — all confirmed `MAPTYPE="NAIP"`, `UTILITY_THRESHOLD` default `"0.3"`, prior
  mean `-0.1` active where applicable.
- `threeDSparseTransDiffusion_periodic_eval.py`, `ImitateTrans_periodic_eval.py` — `EVAL_MAPTYPE="NAIP"`,
  `EVAL_UTILITY_THRESHOLD=0.3`, and a **hardcoded ground-truth mask direction** (`important_mask =
  true_map_flat > EVAL_UTILITY_THRESHOLD`, which does NOT read the shared `LCB` flag at all — a
  duplicated-logic bug pattern, same class as `important_region_variance_from_trajectories.py` had
  before an earlier fix) was flipped to `<=` in both files. Y-axis plot labels ("cells > threshold")
  were also corrected to "cells <= threshold" in both.
- `DataCollector_3D_randomstart_CMAESregularized.py` — was already NAIP-configured going into this
  session (explicitly excluded from an earlier GRF switch-back); confirmed still `MAPTYPE="NAIP"`,
  `utility_threshold=0.3`, prior mean `-0.1` active. This session added `COST_EXPONENT` support to it
  (§4) and re-verified its GP kernel constants match live data (§2).

**Not verified this session, flagged as unresolved**: `realgreedy_singlemap.py` was named alongside
`lawnmower_singlemap.py` in an early request to switch to NAIP, but there is no confirmed record in this
session of that edit actually happening (unlike lawnmower, which was independently confirmed NAIP-
configured via a system reminder). **Check `realgreedy_singlemap.py`'s `MAPTYPE`/`UTILITY_THRESHOLD`/
prior-mean-sign fresh before trusting it's NAIP-ready.** Also not touched, by explicit user choice
earlier this session ("staged to CMA-ES only for now" — though a lot has changed since that was said,
worth re-confirming it's still the intended scope): `Diffusionplanner_PPO.py`, `greedygradient_singlemap.py`.

## 2. GP kernel scale — empirically fit for NAIP, and confirmed to match live data collection

`gaussianprocesstraining.py`'s `initialize_gp()` defaults are `sigma2=0.0079, lengthscale=4.78`
(previously `0.05`/`6.0`, GRF-era values). These came from fitting a `ConstantKernel * Matern(nu=1.5)`
against all 62 non-degenerate NAIP maps (median signal variance and median GP-fitted lengthscale across
maps — see the fitting script pattern in `Kerneltraining/kerneltrainingexperiments.py`, adapted for
NAIP). `DataCollector_3D_randomstart_CMAESregularized.py`'s own `GP_KERNEL_SIGMA2`/`GP_KERNEL_LENGTHSCALE`
constants were updated to match.

**Directly verified this session** (not just assumed): `FINAL_NAIP_DATASET.pt`'s own timestep=0 (pre-
sensing) samples have a mean per-cell prior variance of 0.007749, essentially exactly matching the
current `GP_KERNEL_SIGMA2=0.0079` constant — confirming the dataset was collected with the current,
already-correct kernel parameters, not stale GRF ones.

## 3. NAIP map renumbering (shuffle) — still in effect, still a live footgun

Earlier in this session, the first 60 NAIP maps (originally ids 0-59) were shuffled and relabeled 1-60,
touching only `scripts/csv/` (the operational copy every script reads) — **not** the mirror at
`Datasets/NAIP_dataset/selected_tiles_csv/`, which still holds the original, pre-shuffle numbering.
Confirmed via direct correlation test: `FINAL_NAIP_DATASET.pt` (dataset collection) was run against the
**post-shuffle** numbering — its belief-mean values correlate strongly (r≈0.94-0.95) with today's
`scripts/csv/` content and not at all with the untouched mirror. So `scripts/csv/` is the right source
to trust for anything cross-referencing this dataset.

**Consequence, not yet reconciled**: every numbered map reference anywhere in this project's memory,
prior carryover docs, or informal notes ("map 51 is X%", "maps 1/9/49 are degenerate") predates the
shuffle and no longer describes the same underlying content. Section 3.1 below is the **only**
trustworthy, current map-quality survey.

### 3.1 Current (post-shuffle) map importance-fraction survey, ids 1-60, threshold 0.3/LCB

- **Degenerate low** (avoid): maps 4, 8, 42 (3.3-3.9%), 13, 45 (11-12%), 51 (18.4%, borderline), 43 (18.7%).
- **Degenerate high** (avoid): maps 49, 11, 32, 40, 60, 6, 31, 54 (93.6-99.2% important — almost the
  whole map, nothing to discriminate).
- **Moderate band (20-50%, the trustworthy test set)**: 20, 39, 17, 59, 3, 19, 55, 5, 41, 35, 23, 1, 12,
  15, 22, 14, 30, 34, 48, 56, 37, 21, 52, 27.
- Maps actually used for comparisons this session: **41 (29%), 55 (28.5%), 56 (37.4%)** — all solidly
  moderate. **Map 51 (18.4%)** was used once (CMA-ES-vs-Lawnmower 300-step test) and sits just outside
  the moderate band — not degenerate, but closer to the edge than the others.
- **GRF's own clustering structure is structurally different from NAIP's**, confirmed via connected-
  component analysis at matched grid resolution: GRF map 51's important region is ~85% one single
  connected blob (max gap between components 22.7 units); NAIP maps 55/56 have genuinely disjoint
  clusters with max gaps of 61.6/67.7 units (~3x larger). This matters for §6 and §7.

## 4. COST_EXPONENT — softened the distance-cost bias in CMA-ES's own objective

`gaussianprocesstraining.py`'s `_grid_search_3d_impl` (`score = gain / distance`) and
`trajectory_objective_3d` (`variance_reduction / cost`) both structurally discount distant high-value
regions relative to travel cost. This is a deliberate design choice (mirrors Popovic et al. 2020's
reference implementation) but interacts badly with NAIP's genuinely disjoint important-region clusters
(§3.1) — it can make "milk the cluster I'm already at" locally rational even when a much richer cluster
sits far away.

**Fix**: added `COST_EXPONENT = float(os.environ.get("COST_EXPONENT", "0.5"))` to
`gaussianprocesstraining.py` (near the `LCB` flag), and changed both scoring formulas to use
`gain / (distance ** COST_EXPONENT)` / `cost ** COST_EXPONENT` instead of linear division.
`DataCollector_3D_randomstart_CMAESregularized.py`'s own **duplicated** copies of the grid-search
scoring (`top_k_first_candidates`, `branch_grid_search_3d` — these reimplement the loop rather than
calling the shared function) were updated to import and use the same `COST_EXPONENT`. 1.0 recovers the
old linear behavior exactly; the new default of 0.5 flattens the distance penalty (e.g. at distance 60,
the penalty is ~8x weaker than linear; at distance 5, only ~2x weaker).

**Tested on live CMA-ES** (300-simulation-step comparison, map 51, wall-clock unconstrained): both
`COST_EXPONENT=0.5` and the old `1.0` clearly beat Lawnmower on variance/RMSE metrics, but the 0.5-vs-1.0
comparison itself was a **mixed, modest result** — 0.5 wins on final variance/global RMSE, 1.0 was
slightly better on occupied RMSE and global RMSE AUC. Not a dramatic win in isolation; its main value
turned out to be as raw material for future re-collection (not yet done — `FINAL_NAIP_DATASET.pt` and
every checkpoint trained on it predate this change).

## 5. THE NORMALIZATION BUG — read this before touching any diffusion/ImitateTrans inference

### 5.1 The mechanism

`Diffusionplanner_singlemap.py` imports `diffusion` from `sample_3d_sparse_trans_diffusion.py`, which
does **not** do a normal Python import — it uses `importlib` to dynamically load and **execute**
`Diffusion/threeDSparseTransDiffusion.py` fresh, every single inference run. That script's top-level
code:

```python
DATASET_PATH = Path(os.environ.get("DIFFUSION_DATASET_PATH", str(SCRIPT_DIR.parent / "FINAL_NAIP_DATASET.pt")))
data_dict = torch.load(DATASET_PATH, ...)
...
mean_center = means.mean(); mean_scale = means.std().clamp_min(1e-6)
var_center = vars.mean();  var_scale = vars.std().clamp_min(1e-6)
total_variance_center = total_variance_log.mean(); total_variance_scale = total_variance_log.std().clamp_min(1e-6)
```

These constants (**not saved in any checkpoint** — recomputed fresh every run) are what
`sample_diffusion_trajectory()` uses to normalize the *live* belief-mean/variance maps at every planning
step: `mean_map = (current_mean - diffusion.mean_center) / diffusion.mean_scale`.

**Before this session's fix**, `DIFFUSION_DATASET_PATH` was never set by any inference invocation
anywhere in this project's history, and the code default silently pointed at a **different** dataset
(`CMAES_beamsearch_dataset_3d_synthetic_final.pt`) — meaning every diffusion inference run ever
performed used that dataset's normalization statistics, not NAIP's, regardless of what the model was
actually trained on. `ImitateTrans.py` had the identical pattern but **worse**: its `DATASET_PATH` was
fully hardcoded with **no env-var override at all**.

### 5.2 The fix

- `Diffusion/threeDSparseTransDiffusion.py`: default changed to `SCRIPT_DIR.parent / "FINAL_NAIP_DATASET.pt"`
  (still overridable via `DIFFUSION_DATASET_PATH`).
- `ImitateTrans.py`: added a proper env-var mechanism, `IMITATE_DATASET_PATH`, defaulting to
  `SCRIPT_DIR / "FINAL_NAIP_DATASET.pt"`.
- Both fixes cover **every** dataset-derived constant in one edit (`mean_center/scale`, `var_center/scale`,
  `total_variance_center/scale` all derive from the same loaded `data_dict`) — verified by direct grep,
  nothing else needed a separate patch.

### 5.3 Proof this is real, and proof no retraining is needed

Do not trust an "it got better" result alone as proof of the mechanism (a red herring is possible: fixing
input scale to match the live NAIP data's own natural range could plausibly help even a network trained
on the wrong scale, without proving there's no residual mismatch). The rigorous test performed:
recompute the actual training loss of the epoch-2000 checkpoint (`NAIP_FINAL_DIFF.pth`), using its real
saved weights, under both normalization choices, and compare against the checkpoint's own recorded
training loss (`0.0917`):

| Normalization applied | Recomputed loss |
|---|---|
| Correct (NAIP) | 0.115 (within noise of 0.0917) |
| Wrong (synthetic, the historical bug) | **0.850** (7.4x higher, far beyond noise) |

This is decisive: the network's actual weights are calibrated to NAIP-correct normalization. Training
was fine; only inference was broken. **No retraining is needed** — this was purely a deployment-side bug.
The diagnostic script for this (`verify_checkpoint_normalization.py`, in the session's scratchpad, not
committed anywhere permanent — worth recreating if needed) is a cheap, reusable few-minute sanity check
worth rerunning on any checkpoint whose behavior looks suspicious in the future.

### 5.4 Magnitude and measured impact

| | mean_center | mean_scale | var_center | var_scale |
|---|---|---|---|---|
| Wrong (synthetic, historical default) | 0.290 | 0.223 | 0.00235 | 0.00409 |
| Correct (NAIP) | 0.282 | 0.123 | 0.00200 | 0.00256 |

Scales were off by ~1.6-1.8x. Fixing it produced large, consistent improvements across every checkpoint
tested on map 55 (occupied variance, epoch 2000: 0.777→0.544; global variance: 3.144→1.648; similar
magnitude improvements for epochs 400/750/1250/1600). It also mostly (not entirely) explains the earlier
apparent "more training makes the model worse" finding — that finding was measured entirely under the
bug, and a well-converged network suffers more from wrong-scale inputs than an undertrained one, which
produced a misleading epoch-vs-performance trend. Post-fix, checkpoints 400-1600 cluster much more
tightly together; only epoch 2000 still shows a smaller, real residual degradation (see §8).

## 6. execution_chunk mismatch — real, smaller, tested, not yet made the default

`DataCollector_3D_randomstart_CMAESregularized.py` has two separate constants:
`EXECUTION_CHUNK_SCORING = 40` (how far ahead branches get planned/scored — this is what's saved as the
`trajectories` field in the dataset) and `EXECUTION_CHUNK_UPDATING = 20` (how far the chain **actually**
physically commits before the next replanning round — confirmed by direct measurement: index ~19-20 out
of each round's 41-point dense trajectory is exactly where the next round's `current_position` starts).
`Diffusionplanner_singlemap.py`'s own `execution_chunk` default is **30** — a real, previously-unnoticed
mismatch between how far training commits to a plan (20) and how far inference trusts one (30).

**Tested** (map 55, epoch 400 and epoch 2000, both post-normalization-fix, `EXECUTION_CHUNK=20` via env
override): epoch 400's occupied variance improved 0.371→0.313 (this is what first let it *decisively*
beat Lawnmower rather than merely tie it); epoch 2000 improved 0.544→0.498 (still behind Lawnmower, but
closer). **This has not been made the code default** — `Diffusionplanner_singlemap.py:44` still reads
`execution_chunk = int(os.environ.get("EXECUTION_CHUNK", 30))`. Recommended but not yet done; would need
the same check on `ImitateTrans_singlemap.py` too (not yet tested there).

## 7. Lawnmower flight speed changed (3→2), fairness question still open

`lawnmower_singlemap.py`'s `FLIGHT_SPEED_MPS` was changed by the user (outside this session) from `3` to
`2`. This makes Lawnmower cover less ground in the same 150s wall-clock budget (301 points vs. 401
before, now hits the wall-clock cutoff rather than finishing its sweep) — its performance dropped
accordingly (occupied variance 0.375→0.450 on map 55; global variance 1.251→2.224). **Not verified**:
whether `Diffusionplanner_singlemap.py`/`CMAES_classic_singlemap.py`/etc. use the same
`FLIGHT_SPEED_MPS=2` or still default to 3 — if they differ, comparisons are no longer apples-to-apples
on a shared physical constraint. Check this before trusting any further Lawnmower-vs-X comparison.

## 8. Where diffusion vs. CMA-ES vs. Lawnmower actually stands, post-fixes

Established this session, in order of investigation:

1. **CMA-ES (online, live optimization) clearly beats Lawnmower** on variance/RMSE metrics — this was
   never in question and holds under both `COST_EXPONENT` values.
2. **The raw CMA-ES training demonstrations themselves also clearly beat Lawnmower** — replayed the
   actually-executed (first-20-of-41-points-per-round) portion of real training chains for map 1 (which
   *is* in the training set, unlike 41/55/56): averaged 0.353 occupied variance / 1.182 global variance
   in 280 steps vs. Lawnmower's 0.403/1.251 in 401 steps. **This proves the training data is genuinely
   high quality** — the imitation-learning gap is not a data-collection-quality problem.
3. **Ruled out as explanations for the remaining diffusion-vs-CMA-ES gap**: CMA-ES budget mismatch
   between data collection and online comparison (both use identical `maxiter=20, popsize=12,
   maxfevals=1000` — no override at either call site); the kernel-lengthscale-structurally-favors-
   Lawnmower hypothesis (directly tested by re-scoring the same recorded trajectories under
   lengthscale=4.78 vs. 6.1 — the gap moved the **opposite** direction from predicted, hypothesis
   falsified, do not resurrect without new evidence).
4. **Best result achieved this session** (pre-multimodal-filtering experiment): epoch 400 checkpoint
   (`NAIP_FINAL_DIFF_400.pth`), fixed normalization (§5), `EXECUTION_CHUNK=20` (§6): occupied variance
   0.313 vs. Lawnmower's 0.371 at the time (speed=3) — a clean, decisive win. This is the best-evidenced
   checkpoint+settings combination as of this writing.
5. **Remaining honest explanation for the residual gap**: behavior cloning compresses an expensive,
   per-decision iterative optimizer into one cheap feedforward pass, replayed open-loop over a long
   compounding rollout — a real, known limitation, not (as far as has been found) a remaining bug.

## 9. Checkpoint reference table

| File | Epoch | Recorded loss | Trained on | Notes |
|---|---|---|---|---|
| `NAIP_FINAL_DIFF_400.pth` | 400 | 0.228 | `FINAL_NAIP_DATASET.pt` (multimodal) | Best-tested checkpoint (§8.4) |
| `NAIP_FINAL_DIFF_750.pth` | 750 | 0.182 | same | |
| `NAIP_FINAL_DIFF_TRY.pth` | 1250 | 0.140 | same | |
| `naip_diffusion.pth` | 1600 | 0.101 | same (probably — different file lineage, dated Jul 30, not confirmed same training run as the others) | |
| `NAIP_FINAL_DIFF.pth` | 2000 | 0.092 | same | Currently the most-converged; still shows a real (smaller, post-fix) residual underperformance — see §8.5 |
| `UNIMODAL_NAIP_FINAL_DIFF_1000.pth` | 1000 | 0.197 | `FINAL_NAIP_DATASET_FILTERED.pt` (winner-only, see §10) | New, see §10.3 for the one test run so far |
| `NAIP_FINAL_IMIT.pth` | — | — | ImitateTrans NAIP checkpoint | Not yet re-tested with the normalization fix beyond one map-55 before/after (0.769→0.533 occupied variance) |

All diffusion checkpoints live in `Diffusion/checkpoints/`. Current
`Diffusionplanner_singlemap.py` default is `UNIMODAL_NAIP_FINAL_DIFF_1000.pth` (changed by the user
outside this session, most recently) — **re-check this fresh**, it has changed multiple times per
session already.

## 10. Multimodal-vs-unimodal training data investigation — IN PROGRESS, open thread

### 10.1 The hypothesis

`DataCollector_3D_randomstart_CMAESregularized.py`'s branching mechanism records near-tied alternate
CMA-ES branches (`parent_beam_index` 1, 2, ...) at ~22% of decision points (608 of 2800 winning-branch
samples in `FINAL_NAIP_DATASET.pt` have at least one recorded alternate), but **only the winning branch
(`parent_beam_index==0`) ever continues the chain** — `cx, cy, cz = winner["update_cx"], ...]`. A
non-winning branch's resulting belief state is never explored further; no chain in the dataset ever
starts from having-just-flown the losing option. Hypothesis: if a diffusion model trained to preserve
this multimodality ever samples a losing-branch-like output during a real rollout, it lands in a belief
state with zero relevant training data for what comes next — a covariate-shift problem specifically
induced by preserving multimodality on orphaned branches.

### 10.2 What was checked, and what's still just reasoning

**Directly measured**: winner-vs-runner-up endpoint distance, same decision points, both `FINAL_NAIP_DATASET.pt`
(mean 31.60, median 28.90) and `CMAES_beamsearch_dataset_3d_synthetic_final.pt`/GRF (mean 28.68, median
24.89). **Only a modest ~10-16% difference** — does not strongly support "NAIP's alternate branches are
much more spatially divergent than GRF's" as a clean explanation. This was a real test that came back
weaker than expected; don't oversell it.

**Not directly tested, better-supported reasoning only**: the *consequence* of orphaned-branch-induced
confusion likely differs between map types because of the already-confirmed geography difference (§3.1)
— GRF's single connected blob absorbs indecision cheaply (almost anywhere is "on the way" to something
useful); NAIP's disjoint clusters make the same confusion costly (heading toward neither cluster wastes
real distance). This would make orphaned branches a plausible *second contributor* to the circling
behavior already root-caused to the `gain/cost` distance-discounting bias (§4), not a competing
explanation. **This link has not been empirically tested** — a real next step would be checking whether
diffusion's actual observed indecision/circling moments spatially coincide with known multi-branch
decision points in training, which hasn't been done.

**Also worth remembering**: the premise "multimodality worked fine on GRF" is not well-established
either — an earlier session's investigation (see memory: "Diffusion multimodality investigation") found
real, severe mode collapse on GRF checkpoints too (2 of 3 tested conditions showed *total* collapse onto
one branch). It just was never measured against a task-performance baseline the way NAIP has been this
session, so it's possible the same orphaned-branch problem was always there and just wasn't visible.

### 10.3 What's actually been done, and the one result so far

- `FINAL_NAIP_DATASET_FILTERED.pt` was created (in `scripts/`, confirmed exists, ~89MB vs. original
  ~110MB) — a straightforward filter of `FINAL_NAIP_DATASET.pt` to `parent_beam_index == 0` only,
  applied uniformly across all 17 dataset keys. This preserves all 4 `start_index` parallel chains per
  map — **it removes within-round alternate branches, not the parallel-chain structure** (this was
  confirmed explicitly with the user mid-session; don't confuse the two kinds of "multiple paths").
- The user trained `UNIMODAL_NAIP_FINAL_DIFF_1000.pth` (epoch 1000, loss 0.197) on this filtered dataset,
  outside this session.
- **One test run so far** (map 55, `EXECUTION_CHUNK=20`, fixed normalization, Lawnmower at the current
  speed=2 setting): **mixed, somewhat disappointing result**:
  - Occupied variance: unimodal epoch 1000 = **0.636**, worse than Lawnmower's 0.450 (unimodal loses,
    where the original multimodal epoch-400 checkpoint had decisively won under comparable settings).
  - Global variance: unimodal epoch 1000 = **1.750**, better than Lawnmower's 2.224 (unimodal wins here).
  - This was the very last thing running when this session ended — **interpretation has not happened
    yet**. Important confound before drawing any conclusion: this compares epoch **1000** of the
    unimodal run against epoch **400** of the original/multimodal run — different training extents,
    different datasets, not a matched comparison. It is entirely possible the unimodal dataset just
    needs a different (probably earlier, given the whole epoch-vs-performance story in §5.4/§8) epoch to
    show its best result, not that filtering itself was a bad idea. **Do not conclude "filtering
    multimodality hurts" from this one data point.**

### 10.4 Suggested next steps for this thread

1. Repeat the same epoch-sweep methodology used in §8 (multiple checkpoints at different training
   extents, same map, same seed, same fixed settings) but for checkpoints trained on
   `FINAL_NAIP_DATASET_FILTERED.pt`, to get a fair, matched comparison against the original multimodal
   epoch-sweep curve rather than a single mismatched epoch.
2. If time allows, the spatial-coincidence check described in §10.2 (do observed circling moments align
   with known branch points) would directly test the mechanism rather than just its downstream symptom.
3. Keep the targeting-ratio metric (occupied-%-drop ÷ global-%-drop; ~1 = indifferent to importance, >1
   = genuinely targets it) in the comparison — it was diagnostically useful throughout §8 and would help
   distinguish "unimodal explores less" from "unimodal targets worse" as the reason for any gap found.

## 11. Standing evaluation methodology (reusable, unchanged from earlier sessions)

- **Kalman-replay reconstruction**: `important_region_variance_from_trajectories.py`'s `reconstruct_one()`
  — set `MAPTYPE=NAIP`, `UTILITY_THRESHOLD=0.3` as env vars **before** importing the module (module-level
  constants read once at import), then call with `{"map_id":.., "traj_path":..}`. Returns
  `wall_time_seconds`/`important_variance`/`global_variance` arrays. This is how every occupied/global
  variance number in this whole document was computed, post-hoc, from saved `executed_trajectory.csv`
  files — no re-simulation needed, since the covariance update depends only on sensor pose, never on
  measured values.
- **Standard run settings**: `MAPTYPE=NAIP UTILITY_THRESHOLD=0.3 SKIP_VIZ=1`, BLAS threads capped to 1
  each (`OMP_NUM_THREADS` etc.) for safe parallelism, `RUN_SEED=90001` for diffusion (repeat-seed
  convention), `WALLCLOCK_SECONDS=150` for the standard wall-clock comparisons, or
  `WALLCLOCK_SECONDS=0 TIMEALLOTED=300 ENFORCE_MIN_STEP_TIME=0` for the "simulation steps, not wall
  clock" variant.
- **Parallel run pattern**: `ThreadPoolExecutor` + `subprocess.Popen(...).wait()` per job. **Give every
  parallel job a distinct `RUN_OUTPUT_TAG`** — this session hit a real bug where several diffusion
  checkpoint runs sharing one tag wrote to the same output directory simultaneously and clobbered each
  other's trajectory CSV; always include the checkpoint/variant name in the tag.
- **Targeting ratio** = (% occupied-variance drop) ÷ (% global-variance drop). ~1.0 = the planner
  reduces occupied variance in exact proportion to how much of the map it explores (indifferent to
  importance); >1 = genuinely prioritizing important cells beyond what raw exploration would predict;
  <1 = worse than proportional. Useful for separating "explores less" from "targets worse" as failure
  modes — these are not the same thing and can point in opposite directions across checkpoints (see §8.4
  vs. the pre-fix epoch sweep).

## 12. Explicitly declined / do not re-propose

- **Sensor noise scaling** (`noise_model(altitude)` in `gaussianprocesstraining.py` returns a fixed
  absolute measurement-noise variance regardless of map type, proportionally larger relative to NAIP's
  narrower value range than it was for GRF). Raised again this session for completeness only; the user
  declined this in an earlier session and nothing new has changed that calculus.
