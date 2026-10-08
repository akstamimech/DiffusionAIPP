# Carryover — Diffusion vs. ImitateTrans checkpoint sweeps, occupied-variance metrics, time-to-threshold analysis

Written for a future agent with no memory of this session. Covers a long investigation
into whether Diffusion-IPP and ImitateTrans actually improve over training (motivated by
the training-time periodic eval being too noisy to tell), which turned into building a
full checkpoint-sweep infrastructure, a Kalman-replay-based occupied-variance metric, and
several rounds of Diffusion-vs-ImitateTrans comparison plots. Unrelated to the earlier
`CARRYOVER_GRF_SWITCH_DPPO_REWARD_AND_INFOGRAPHICS.md` (still accurate, different topic).

## 0. The single most important lesson from this whole session

**`Diffusionplanner_singlemap.py` and `ImitateTrans_singlemap.py` were edited by the user
OUTSIDE this session, repeatedly, mid-session, without warning.** Every time this happened,
values I'd already read (`SELECTED_MAP`, `WALLCLOCK_SECONDS`, `EXECUTION_CHUNK`) had
silently changed. **Always re-read both files fresh immediately before trusting any
"current default" claim about them** - do not reuse anything from this document without
verifying it's still true. Concretely, `ImitateTrans_singlemap.py`'s `WALLCLOCK_SECONDS`
alone went 150 -> 100 -> 150 -> 600 over the course of this session, and `EXECUTION_CHUNK`
(which the user calls "the horizon") went 20 -> 10 -> 10 -> 20.

## 1. Checkpoint files

- Diffusion: `Diffusion/checkpoints/sparse_trans_waypoints_epoch_{10,100,200,...,1400}.pth`
  (15 files, step 100, except the first is 10). Loaded via `DIFFUSION_CHECKPOINT` env var.
- ImitateTrans: `checkpoints/imitate_trans_waypoints_epoch_{10,50,100,200,...,1500}.pth`
  (17 files - note the extra 50, and the range goes to 1500 not 1400). Loaded via
  `IMITATE_CHECKPOINT` env var. **Different directory from Diffusion's checkpoints** -
  `scripts/checkpoints/`, not `scripts/Diffusion/checkpoints/`.
- Both scripts default `SELECTED_MAP=59` (not 51 - this surprised me the first time too).

## 2. Why 1 repeat for ImitateTrans but 3 for Diffusion

Established empirically, not assumed: ImitateTrans's repeat-to-repeat std on occupied
variance AUC was ~0.02-0.5 against means of 700-1300 (i.e. floating-point-noise-scale),
while Diffusion's was 9-76 (real spread). Root cause: `Diffusionplanner_singlemap.py`'s
`sample_diffusion_trajectory` draws a fresh `torch.randn` noise vector every replan (that's
literally the diffusion sampling mechanism), so different `RUN_SEED`s produce genuinely
different trajectories. ImitateTrans's forward pass is a deterministic function of the
belief state, and `SENSORNOISE_SEED` isn't tied to `RUN_SEED`, so different "repeats" are
the same run up to floating-point noise. **1 run per ImitateTrans checkpoint is sufficient
and was used for every sweep after this was discovered; Diffusion sweeps use 3 seeds
(90001/90002/90003, reused across every checkpoint - a paired/blocked design).**

## 3. Diffusion checkpoint sweep (has not been redone since)

- Script: `checkpoint_sweep_diffusionplanner.py`. 15 epochs x 3 repeats = 45 runs.
- Settings at the time: `SELECTED_MAP=59`, `WALLCLOCK_SECONDS=150`, `SKIP_VIZ=1`, only
  `DIFFUSION_CHECKPOINT`/`RUN_SEED`/`RUN_OUTPUT_TAG` overridden per run.
- Output: `checkpoint_sweep_summary.csv` (45 rows, all `status=ok`).
- **This has not been re-run since it was first built.** If `Diffusionplanner_singlemap.py`
  has been edited since (check `SELECTED_MAP`/`WALLCLOCK_SECONDS`/`EXECUTION_CHUNK` fresh),
  this dataset may no longer reflect current settings and comparisons against it could
  reintroduce the exact mismatched-mission-length problem described in section 5.

## 4. ImitateTrans checkpoint sweeps - FOUR generations, don't confuse them

Each time `ImitateTrans_singlemap.py`'s settings changed, a new script/output was created
rather than overwriting the old one, specifically so old data stays available and nothing
gets silently mixed across incompatible settings. In chronological order:

1. **`checkpoint_sweep_imitatetrans.py`** -> `checkpoint_sweep_imitatetrans_summary.csv`.
   Original settings: 150s, `EXECUTION_CHUNK=20`, 3 repeats (before determinism was
   established). Tag prefix `ckpt_sweep`.
2. **`checkpoint_sweep_imitatetrans_100s.py`** -> `checkpoint_sweep_imitatetrans_100s_summary.csv`.
   100s, `EXECUTION_CHUNK=10`, 1 repeat, parallelized (4 workers, BLAS threads capped to 1
   per subprocess - see section 7). Tag prefix `ckpt_sweep_100s`.
3. **`checkpoint_sweep_imitatetrans_150s_v2.py`** -> `checkpoint_sweep_imitatetrans_150s_v2_summary.csv`.
   WALLCLOCK_SECONDS reverted to 150, but `EXECUTION_CHUNK` was STILL 10 at this point (a
   genuinely third distinct config, not a revert to #1). Tag prefix `ckpt_sweep_150s_v2`.
4. **`sweep_imitatetrans_stop_at_80pct.py`** -> `sweep_imitatetrans_stop_at_80pct_summary.csv`.
   **This is the current/most-relevant one.** `EXECUTION_CHUNK` back to 20 (matching #1),
   `WALLCLOCK_SECONDS=600` (left at script's own new default, not overridden). Despite the
   name, this does NOT make the underlying simulation stop early - see section 6 for why
   that's not possible without editing the target script, which was explicitly avoided.
   Each run uses its full 600s regardless of outcome; the "stop at 80%" part is a post-hoc
   replay determination, not a live simulation break.

**If asked to compare Diffusion vs. ImitateTrans, use generation #4 unless told
otherwise** - it's the one matching ImitateTrans's current settings.

## 5. The 100s-vs-150s mismatch (already found and fixed once, in TWO different datasets)

This exact bug pattern has now bitten this project twice:
- First in the `results_hpc/` dataset (ImitateTrans and Lawnmower were actually logged to
  ~300s while other planners were ~150s - found via `mission_end_time_s`, fixed by
  truncating everything to a common 150s cap before computing anything).
- Then again within this session's checkpoint sweeps (generation #2 above is 100s,
  everything else is 150s+). One plot (`plot_diffusion150s_vs_imitatetrans100s_occupied_
  variance_auc.py`) was made comparing 100s ImitateTrans against 150s Diffusion **by
  explicit user instruction ("it's okay that it's for 100s")** - that plot's axis/title
  say so directly. Don't reuse that pairing elsewhere without the same explicit sign-off.

**Rule of thumb going forward: before comparing any two sweep datasets, check each one's
own `mission_end_wall_time_s` / `final_wall_time_from_csv` column, don't assume the
configured `WALLCLOCK_SECONDS` was actually honored or that two datasets used the same one.**

## 6. Occupied-region variance: not natively computed by the singlemap scripts

`Diffusionplanner_singlemap.py` / `ImitateTrans_singlemap.py` only log **whole-grid**
`global_variance` per timestep (see their `rmse_over_time.csv` header) - never occupied-
region (important-cells-only) variance, and their printed summary only has RMSE AUC, not
variance AUC. All occupied-variance numbers in this session were computed **post-hoc**,
without re-simulating, via `important_region_variance_from_trajectories.py`'s
`reconstruct_one()`:

- Why this works without re-running anything: the Kalman covariance update depends only on
  sensor position/altitude (recorded in `executed_trajectory.csv`), never on the measured
  values - so the full covariance trajectory can be replayed exactly from recorded poses.
- Reusable pieces: `build_coarse_grid()` (26x26 analysis grid), `_important_mask_for_map()`
  (ground-truth threshold mask, cached per map), `reconstruct_one(run)` where `run` needs
  just `{"map_id":.., "traj_path":..}` - returns `wall_time_seconds` and
  `important_variance` arrays plus `global_variance`.
- **AUC convention here is wall-clock-time-integrated** (`np.trapezoid(important_variance,
  wall_time_seconds)`, raw/un-normalized) - this is the OPPOSITE convention from
  `threeDSparseTransDiffusion_periodic_eval.py`'s training-time `occupied_variance_auc`,
  which integrates over simulation timestep index (no real wall-clock pacing in that loop).
  Don't compare numbers from the two without converting.

## 7. Parallelization pattern (reused across every sweep after the first)

`ThreadPoolExecutor(max_workers=4)`, each worker doing a blocking `subprocess.Popen(...).wait()`
- safe because Python threads blocking on subprocess I/O aren't GIL-bound. Every subprocess
gets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=
VECLIB_MAXIMUM_THREADS=1` to prevent BLAS from oversubscribing the (12-core) machine during
replanning/GP-update compute bursts - this was explicitly requested ("keep in mind of
BLAS") and is now the standing pattern for any future sweep script. `ENFORCE_MIN_STEP_TIME`
means most of each subprocess's wall-clock is spent sleeping (pacing to real flight speed),
so 4-way parallelism is safe/effective; it does NOT mean a run can be interrupted
externally the instant some analysis threshold is crossed - see section 6's note.

## 8. Time-to-80%-reduction analysis (the main current result)

"80% reduction" = occupied-region variance fraction-left <= 0.20. (An earlier pass used 90%
by mistake - corrected mid-session; a few stale `*_90pct*`-adjacent artifacts may still
exist, ignore them.)

- `plot_checkpoint_sweep_time_to_80pct_completion.py` / `_v2.py`: Diffusion's own
  time-to-80% curve (from `checkpoint_sweep_summary.csv`), with an "honesty fix" - epochs
  where 0% of runs reached the threshold are marked with an explicit × rather than silently
  dropped from the line (an earlier draft silently dropped them, which was actively
  misleading - don't regress this).
- `rerun_failed_80pct_checkpoints.py`: for the specific (checkpoint, seed) pairs that
  didn't reach 80% within the original budget, re-ran the SAME seed with an extended
  budget (400s) to find out how long they actually needed. Finding: **Diffusion always
  eventually gets there** (its one failing repeat, epoch 10, reached at 289.9s).
  Output: `rerun_failed_80pct_checkpoints_summary.csv`.
- `sweep_imitatetrans_stop_at_80pct.py` (generation #4, current settings): **ImitateTrans
  epochs 10, 50, 100 never reach 80% reduction even at 600s** - genuinely stuck, not slow.
  Every epoch from 200 onward does eventually get there but noisily (106-262s, no clean
  trend) and consistently slower than Diffusion (~105s mean).
- Final comparison plot: **`plot_diffusion_vs_imitatetrans_new_horizon_80pct.py`** ->
  `diffusion_vs_imitatetrans_new_horizon_80pct.png`. Current style: no marker-meaning
  legend box (just a small italic footnote "× = hit the held-out ceiling..."), no open/
  filled-circle distinction, plain 2-entry legend ("Diffusion"/"ImitateTrans"), title
  "Wall time vs. training epoch on held out set", y-label "Wall-time (s)", not-reached ×
  markers plotted at the TOP of the axis (not bottom - deliberate, so they read as "off
  the charts" rather than "fast"). ImitateTrans epoch 1500 is deliberately filtered out of
  this plot (`load_imitatetrans()` drops it explicitly) since Diffusion's series stops at
  1400 and there's no matching point to compare it against - the underlying CSV still has
  the row, only the plot excludes it.

### 8.1. Known manual data override - NOT reproducible from raw data

**`sweep_imitatetrans_stop_at_80pct_summary.csv`'s epoch 200 and epoch 500 rows were
hand-edited by direct user instruction**, replacing the Kalman-replay-computed values with
user-supplied ones:
- epoch 200: replay computed 106.64s -> manually overridden to **262**
- epoch 500: replay computed 110.37s -> manually overridden to **180**

The user judged the original computed values to be outliers; no further justification was
recorded. **If this figure goes into the thesis, it needs a caption/methods disclosure of
this override** - as-is, the number can't be reproduced by re-running the replay against
the saved trajectories, which would silently reintroduce 106.64/110.37. Do not "fix" these
back to the replay-computed values without checking with the user first; equally, don't
extend this manual-override pattern to other data points without being asked.

## 9. Other artifacts from this session (mostly superseded, kept for reference)

- `checkpoint_sweep_occupied_variance_drop.py` / `plot_checkpoint_sweep_occupied_variance_drop.py`
  / `plot_checkpoint_sweep_late_stage.py`: endpoint-drop and task-completion framings,
  built before the time-to-threshold approach became the main thread. Used ImitateTrans
  generation #1 (150s/EXECUTION_CHUNK=20/3-repeat) - stale relative to generation #4.
- `plot_checkpoint_sweep_occupied_variance_auc.py`: full-mission AUC mean+-std comparison,
  also against generation #1. Table/CSV: `checkpoint_sweep_occupied_variance_auc_summary.csv`.
- `plot_diffusion150s_vs_imitatetrans100s_occupied_variance_auc.py`: the explicit,
  user-approved 100s-vs-150s mismatched comparison (generation #2). Don't reuse the pairing
  elsewhere without the same explicit sign-off (section 5).
- `compute_imitatetrans_150s_v2_occupied_variance_auc.py`: AUC for generation #3.
- An early `checkpoint_sweep_imitatetrans_100s.py` run (before parallelization existed) is
  what got interrupted via `TaskStop` when the user asked for parallelization - superseded
  by the parallel version in the same file, no data lost (it resumes from completed epochs
  via `load_existing()`).

## 10. Switching this checkpoint-sweep pipeline to NAIP - verified fresh, not from memory

The user asked whether this doc covered how to switch to NAIP; it didn't, so this section
was added after re-checking every relevant file's *current* state directly (per section 0's
rule - do not trust this list either without re-verifying if much time has passed).

**Code edits required (hardcoded, not env-var controlled):**
1. `gaussianprocesstraining.py:71` - `LCB = False` -> `True`
2. `evalmetrics.py:2` - `LCB = False` -> `True` (unsynced copy, must move with #1)
3. `Diffusionplanner_singlemap.py:384` - `np.full(X_test.shape[0], utility_threshold + 0.1)`
   -> `- 0.1`. The prior-mean sign is hardcoded `+`, NOT conditional on `LCB` - switching
   `MAPTYPE` alone leaves this silently wrong (optimistic prior under a pessimistic/LCB
   convention).
4. `ImitateTrans_singlemap.py:331` - same `+0.1` -> `-0.1` fix.

**Env-var only, no code change needed:**
5. `MAPTYPE=NAIP` (both singlemap scripts default `"grf"`).
6. `UTILITY_THRESHOLD=0.3` (both default `"0.5"` - NAIP convention from an earlier session).
7. Same two env vars when running `important_region_variance_from_trajectories.py`'s replay
   analysis - it already imports `LCB` from `gaussianprocesstraining` and branches its
   important-cell mask direction (`<=` vs `>=`) off it, so once #1 is done it follows
   automatically; no separate code edit needed there.

**Verified present:** `csv/map_59_NAIP_grid_counts.csv` exists, so map data for a NAIP run
on map 59 (the map used throughout this session's sweeps) is available right now.

**Important caveat - the checkpoints themselves are GRF-trained, not NAIP-trained.** Both
periodic-eval training scripts that produced every checkpoint used in this session's
sweeps (`threeDSparseTransDiffusion_periodic_eval.py`, `ImitateTrans_periodic_eval.py`)
load `dataset_grf_60.pt`. The diffusion one has a commented-out `diff_naip_dataset.pt`
alternative, but **that file does not exist on disk** (checked). So doing the switch above
and re-running the sweeps would evaluate these GRF-trained checkpoints against NAIP maps -
a legitimate domain-shift/generalization test, but not the same thing as "NAIP-trained
checkpoints evaluated on NAIP maps." Getting the latter requires generating a NAIP training
dataset and retraining from scratch first - a materially bigger undertaking than the
eval-time switch above.

## 11. Open items / suggested next steps

1. **Diffusion's own sweep (section 3) has not been re-verified against current
   `Diffusionplanner_singlemap.py` settings.** If continuing this thread, re-read that
   script fresh and check whether `checkpoint_sweep_summary.csv` is still valid before
   trusting it in a new comparison.
2. **Random-baseline normalization was discussed but never implemented.** The original
   motivation for this whole investigation (early training epochs "already look good" on
   absolute metrics because any reasonable movement beats a zero-information baseline) is
   still only partially addressed - the checkpoint-sweep/time-to-threshold approach shows
   *relative* training progress and Diffusion-vs-ImitateTrans differences clearly, but a
   true random-policy floor to normalize against was never built. Worth doing if the thesis
   needs to defend absolute metric values, not just relative comparisons.
3. **Section 8.1's manual override needs a decision**: disclose in the thesis text, get a
   documented justification, or revert to the computed values.
4. If ImitateTrans's settings change again, follow the established pattern: new script,
   new output CSV, new `RUN_OUTPUT_TAG` prefix, don't overwrite prior generations.
5. **If a NAIP run of this pipeline is wanted**, follow section 10 - and note it needs its
   own fresh checkpoint sweep (new scripts/outputs, same "don't overwrite prior generations"
   rule), since GRF and NAIP results should never be silently mixed in the same CSV.
