# Carryover — GRF/LCB switch-back, DPPO reward redesign, diffusion-math Q&A, and a batch of new figures

Written for a future agent with no memory of this session. Unrelated in topic to the earlier
`CARRYOVER_CMAES_REWARD_2.md`/`_3.md` (CMA-ES objective/hyperparameter work, still accurate) and to
`CARRYOVER_NAIP_SWITCH_AND_BASELINES.md` (the **previous** session's NAIP switch — this session did
**the mirror-image switch back to GRF**, so treat everything in that older file as superseded/reversed
for anything about MAPTYPE/LCB/threshold defaults; its other content — occupied-variance-AUC
methodology, map-200 four-corner findings, multimodality investigation — is untouched and still valid).

## 0. Repo/git state

**Important directory: `C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\`** — this is the repo root
(confirmed via `git rev-parse --show-toplevel`). The working `Diffusion/` subfolder
(`scripts/Diffusion/`) holds the diffusion/DPPO stack specifically. Branch `prototype`. **Still zero
commits made this entire project** — everything below is uncommitted working-tree state, consistent
with every prior session. `scripts/CARRYOVER_*.md` files (this one included) are the established
cross-session handoff mechanism in lieu of commit history.

## 1. THE BIG ONE — GRF/LCB switch-back (read this first)

The user switched the whole codebase from NAIP (real aerial imagery, LCB/UCB=lower-confidence-bound)
back to **GRF (synthetic Gaussian random fields), LCB=False (UCB/upper-confidence-bound)** — a full
mirror-image reversal of the previous session's NAIP switch. Requested range was GRF maps **0–150**;
verified all of `map_0..map_150_grf_grid_counts.csv` already exist in `csv/` (in fact up to map_200),
so this was a pure config/code change, no new data generation needed.

**Explicitly excluded from the switch, per direct user instruction:** all DataCollector scripts —
`DataCollector_3D_randomstart_CMAESregularized.py` (the canonical one) and everything under
`DataCollector+datasets/`. These still have their pre-existing (mixed NAIP/multiblob/grf) hardcoded
`MAPTYPE` values from before this session — do not touch them without a fresh explicit request.

### 1.1 What actually changed (verify with the greps below before trusting this list blindly)

- **`LCB = False`** in both `gaussianprocesstraining.py` (line ~71) and `evalmetrics.py` (line ~2,
  the separate unsynced copy — must always move in lockstep with the first).
- **Prior-mean sign flipped `utility_threshold - 0.1` → `+ 0.1`** (UCB/GRF: cells start
  optimistically-*unimportant*, vs. the old LCB/NAIP pessimistic-*important* prior) in:
  `CMAES_classic_singlemap.py`, `greedygradient_singlemap.py`, `realgreedy_singlemap.py` (these three
  already had a commented-out alternate line from a previous session — just swapped which is active),
  `Diffusionplanner_singlemap.py`, `ImitateTrans_singlemap.py`, `lawnmower_singlemap.py`,
  `Diffusionplanner_PPO.py`, `Diffusion/DPPOTRAINING_batched.py`,
  `Diffusion/threeDSparseTransDiffusion_periodic_eval.py`, `ImitateTrans_periodic_eval.py` (all direct
  edits), plus `random_singlemap.py`'s own `-1.0`→`+1.0` variant (different magnitude, same sign flip).
- **`MAPTYPE` default `"NAIP"` → `"grf"`** in ~17 files: all of the above, plus
  `PPO_residual_training.py`, `Diffusion/DPPOTRAINING.py`, `Diffusion/DPPOTRAINING_overnight.py`,
  `hpc_sweep_common.py` (both occurrences), `compare_map51_all_planners.py`,
  `compare_multimap_all_planners.py`. `EVAL_MAPTYPE` too, in the two `_periodic_eval.py` files.
- **`UTILITY_THRESHOLD`/`utility_threshold` default `0.3` → `0.5`** across the same file set (GRF
  convention; 0.3 was NAIP-specific).
- **`important_region_variance_from_trajectories.py` parametrized** (was hardcoded `_NAIP_` in two
  places: `DIR_PATTERN` regex and `gt_path`) — now reads `MAPTYPE` env var (default `grf`), threshold
  default `0.5`, and its mask direction is LCB-aware (`<=` vs `>=`) instead of hardcoded.
- **`hpc_sweep_common.py`**: also fixed a real bug found along the way — `_run_one`'s output-directory
  naming hardcoded `map_grf_...` literally regardless of the actual `MAPTYPE` env var, which would have
  broken metric-file discovery for any non-GRF sweep. Now builds the directory name from the actual
  `maptype` value. Also changed `MAP_START`/`MAP_END` defaults from `41`/`50` to `0`/`150` to match the
  user's stated range (can still be overridden per-sweep via env var).
- **Two real bugs found and fixed** (not just config flips): `ImitateTrans_periodic_eval.py` and
  `Diffusion/threeDSparseTransDiffusion_periodic_eval.py` both had `important_mask = true_map_flat >
  EVAL_UTILITY_THRESHOLD` hardcoded regardless of LCB state — now branches on `LCB` like everywhere
  else (verified fixed: `grep -n "important_mask = true_map_flat" ImitateTrans_periodic_eval.py
  Diffusion/threeDSparseTransDiffusion_periodic_eval.py` should show both `<=`/`>=` branches).

### 1.2 Verification commands (re-run these before trusting the switch is intact)

```
grep -rn "LCB = True" gaussianprocesstraining.py evalmetrics.py          # should be EMPTY
grep -rln '"MAPTYPE", "NAIP"' --include="*.py" .                          # should only hit DataCollector files
grep -rln "UTILITY_THRESHOLD.*\"0\.3\"\|utility_threshold.*0\.3" --include="*.py" .   # should be EMPTY (excl. DataCollector)
```

### 1.3 DPPOTRAINING_batched.py's map pool was also updated for GRF

`TRAIN_MAP_IDS = list(range(0, 50))`, `VAL_MAP_IDS = list(range(51, 61))` — matches the thesis's own
stated convention ("trajectory executions on 50 GRF maps... maps 51-60 form a held-out set"). This
replaced the NAIP-era `[0..44]`/`[45..49]` split (which existed specifically to work around three
near-empty NAIP maps 1/9/49 — that concern doesn't apply to GRF maps at all, no need to re-derive it).

## 2. DPPO reward was redesigned this session — belief-masked, not whole-grid

`Diffusion/DPPOTRAINING_batched.py`'s `compute_step_reward` used to score every replan by **whole-grid**
posterior variance (no interest mask at all). Per direct discussion with the user, it now scores only
variance in cells the **current belief** (not ground truth) flags as important, via
`importance_filter(mu_after, P_after, beta=UCB_BETA, threshold=UTILITY_THRESHOLD) > 0` — reusing the
shared `gaussianprocesstraining.importance_filter` function directly (newly imported into this file)
rather than reimplementing the mu±beta*sigma comparison inline, which is what the file's own dead-code
docstring sketch had done (and which was still written in the old LCB direction despite being named
`UCB_BETA` — that stale sketch is now gone, replaced with accurate documentation).

**Why this is principled, not arbitrary** (worth restating if asked to justify it again): it aligns
DPPO's fine-tuning objective with what the BC-cloned expert's own trajectory objective already
optimizes (`masked_expected_variance_reduction_from_sensor`, itself belief-masked). The denominator
(`baseline_total_variance`, fixed at mission start) stays unchanged — under the current GRF/UCB prior
(`mu0 = threshold + 0.1`, uniformly above threshold), the *initial* belief mask is provably the whole
grid on every map, so that fixed denominator was already exactly right; only the numerator needed
masking. `variance_global`/`variance_occupied` (ground-truth-masked) are still computed and returned
for logging/`evaluate_generalization`/checkpoint-selection — verified `_save_best_checkpoint` still
deliberately selects on whole-grid fraction for cross-iteration comparability, unaffected by this change.

Two doc-only fixes made in the same file while in there: `_sample_episode_envs`'s docstring described
stale constants (`num_episodes=16`, pool size 10) that no longer match reality (`32` episodes, pool of
`50`) — corrected to describe the actual current behavior (32-of-50 shuffled per iteration, no repeats
within a round, no cross-iteration persistent cursor). And a comment said `diffusion.T=30` — actual
value is `20` — fixed.

**Notation correction, told to the user directly, not written anywhere else**: earlier in this session
I incorrectly said the live planner samples with `eta=1.0`. That's wrong — `Diffusionplanner_singlemap.py`'s
`sample_diffusion_trajectory` uses **its own** `ETA = float(os.environ.get("ETA", "0.0"))` constant
(default **0.0**, deterministic DDIM), not `sample_3d_sparse_trans_diffusion.py`'s unrelated
`ETA=1.0` (a different file's constant, coincidentally same name, never imported into the planner).

## 3. Still-open, NOT fixed this session — DPPO training scripts remain broken

Two bugs identified (and reported to the user in detail) in a prior part of this session, offered to
fix, but the user never confirmed and the conversation moved elsewhere — **verified still present as
of the end of this session**:

1. **`REPO_DIR = SCRIPT_DIR` (missing `.parent`)** in all three of `Diffusion/DPPOTRAINING.py`,
   `Diffusion/DPPOTRAINING_batched.py`, `Diffusion/DPPOTRAINING_overnight.py` — this means
   `sys.path.insert(0, str(REPO_DIR))` never actually adds `scripts/` to the path, so running any of
   these directly (`python DPPOTRAINING_batched.py` from inside `Diffusion/`) still dies with
   `ModuleNotFoundError: No module named 'gaussianprocesstraining'`. Only works today via
   `python -m Diffusion.DPPOTRAINING_batched` invoked from `scripts/` (confirmed this workaround during
   this session's own reward-editing verification).
2. **`DPPOTRAINING_batched.py`'s `DEFAULT_CHECKPOINT = checkpoints/current_best.pth` still doesn't
   exist** — no env-var override, will crash at `torch.load` the moment training actually starts.
   Real candidates on disk: `current_best_bc.pth`, `current_best_updated.pth`, `naip_diffusion.pth` (all
   verified this session to load cleanly into the current `NoisePredictor` architecture) — **but now
   that the whole codebase is back on GRF, `naip_diffusion.pth` is almost certainly the WRONG default**
   (it's presumably NAIP-trained, per its filename). `current_best_updated.pth` is what
   `Diffusionplanner_singlemap.py` itself currently defaults to (the user manually set this — it was
   briefly `dppo_best.pth` mid-session, then changed back) and is the more defensible GRF-era choice,
   but this hasn't been confirmed against actual training provenance.
3. Also still true from before: old `dppo_best.pth`/`dppo_iteration_*.pth` (July-dated) **cannot** be
   loaded into the current architecture at all — missing `wp_tokenization.total_variance_mlp.*` keys
   (predates the "total variance conditioning" architecture addition). Any new DPPO run has to start
   fresh from a BC checkpoint, not resume.

None of this blocks anything else — DPPOTRAINING_batched.py's *code* (reward, map pool, imports) is
fully updated and correct; it just can't literally be executed yet without one of the two path/checkpoint
fixes above.

## 4. Also still-open: threeDSparseTransDiffusion.py vs. its `_periodic_eval.py` variant disagree

Found via direct comparison (user asked "do they have the same settings"), reported, **not yet fixed**:
- `BATCH_SIZE`: `256` in the main file vs. `512` in `_periodic_eval.py`.
- `DATASET_PATH`: main file defaults to `scripts/CMAES_beamsearch_dataset_3d_synthetic_final.pt`
  (overridable via `DIFFUSION_DATASET_PATH` env var); `_periodic_eval.py` hardcodes
  `Diffusion/dataset_grf_60.pt` with **no override** — a different file in a different directory. This
  means their `build_map_id_split`-derived train/val splits (and which map `_periodic_eval.py`'s own
  periodic single-map eval runs against) don't line up with the main script's, by default.
Everything else checked matches exactly (architecture, LR/WEIGHT_DECAY/T/etc., optimizer/scheduler,
DataLoader setup, and the `train()` loop body itself is byte-identical apart from the added eval hook).

## 5. Abandoned/interrupted task from early this session — reconsider before resuming

The user asked to sweep `ImitateTrans_singlemap.py` over NAIP maps 41–50 (one run each, no repeats,
parallelized, results into a **copy** of the `results_hpc` folder structure since NAIP results
shouldn't mix with the existing GRF ones there). I fixed a real bug in `hpc_sweep_common.py` in prep
(the output-dir-naming bug described in §1.1), then launched a first serial attempt, got interrupted
("why is it 50 mins, I need ~20") mid-parallelization-redesign, and **the conversation permanently
pivoted to the GRF/LCB switch before this sweep was ever actually completed or even successfully
re-launched**. No `results_hpc_naip`-style output exists from this attempt.

**Complication if resuming**: this whole session's subsequent work flipped every default back to GRF.
Resuming the original NAIP sweep now requires explicit `MAPTYPE=NAIP` (and `UTILITY_THRESHOLD=0.3`,
etc.) env-var overrides per run, since the script defaults no longer point at NAIP. Worth confirming
with the user whether they still want this NAIP sweep at all, given the GRF pivot.

## 6. Diffusion-math Q&A — corrections/clarifications given directly in chat, not saved anywhere

A long thread walked through: the BC-vs-diffusion mode-averaging math (conditional-mean-of-BC vs.
score/responsibility-weighted-mixture for diffusion, with a worked two-mode toy derivation), where the
"log" in the score function ∇log p comes from (the ∇f/f identity, applied to the Gaussian forward
kernel — reproduces the paper's own eq. 30), and a detailed line-by-line walkthrough of
`sample_diffusion_trajectory`/`ddim_sample_timestep`'s actual DDIM math in `Diffusionplanner_singlemap.py`.
None of this was written into any file — if picking this thread back up, it's not recoverable from code
comments alone, only from this session's transcript.

One correction worth remembering: the "AUC" metric name used throughout `analyze_beta_sweep_maps51-55*.py`
and `important_region_variance_from_trajectories.py`'s downstream analysis scripts is a misnomer — it's
actually `trapz(variance_curve, timesteps) / (timesteps[-1] - timesteps[0])`, i.e. **step-averaged**
variance (normalized by elapsed *timestep count*, not wall-clock seconds — the code uses the trajectory
CSV's `timestep` column, not `wall_time_seconds`), not a raw area-under-curve. A new script
(`analyze_beta_sweep_maps51-55_auc_only.py`, see §7) renames this correctly.

## 7. New scripts created this session (all in `scripts/` unless noted)

**Figure/animation scripts** (all reuse real functions from `gaussianprocesstraining.py`,
`CMAES_classic_singlemap.py`, `Diffusionplanner_singlemap.py`, `threeDSparseTransDiffusion.py` directly
— none reimplement simulator/diffusion math):
- `plot_lattice_over_map.py` — 3D pyramid candidate-lattice over NAIP map 11 ground truth.
- `plot_sensor_noise_model.py` + `plot_fov_resolution_altitude.py` + `combine_noise_fov_resolution.py`
  — sensor noise vs. altitude, FOV/resolution vs. altitude (real `build_sensor_matrix` block-averaging),
  and a PIL-stacked combined version. IEEE-print-sized (large fonts, minimal chrome).
- `generate_hero_trajectory.py` / `plot_simulator_hero_shot.py` — a real, longer (early-exits once
  enough replans happen) CMA-ES flight over NAIP map 11 with control waypoints highlighted, LaTeX
  Computer-Modern mathtext styling (no real LaTeX installed on this machine — `mathtext.fontset="cm"`
  is the no-dependency stand-in).
- `plot_diffusion_belief_progression.py` — 3-panel belief-mean progression (`b_5→b_10→b_20`) for a real
  diffusion-planner mission (map 51, GRF now), caches simulation output to `.npz` so re-styling doesn't
  require re-running the ~20s simulation (`REPLOT_ONLY=1`).
- `plot_forward_noise_trajectory.py` — the most-iterated one. Takes a real planner-sampled `x0`,
  applies `threeDSparseTransDiffusion.forward_diffusion_sample` at 5 steps (`K=0,5,10,15,19` — note
  `K=20` doesn't exist, `T=20` means valid range is `0..19`), and renders it TWO ways from the same
  noise draws: real-world-scale spline trajectories (dashed 100×100m box, no axes) and the raw 3×8
  tensor as an actual pixel grid (now transposed to 8×3, waypoint-index × xyz, axis ticks restored,
  diverging blue/red colormap). `run()` takes `(x0_seed, noise_seed_base, run_tag)` and is called twice
  in `main()` for two example draws — outputs suffixed `_run1`/`_run2` for both the trajectory and
  pixel renders (4 files total per execution).
- `animate_reverse_process.py` — GIF of the reverse DDIM process, one frame per denoising step, for one
  real replan (`eta=0`, matching the corrected planner default from §2). Calls
  `diffusion.ddim_sample_timestep` directly in its own loop since `ddim_sample` discards its
  intermediates internally.
- `plot_grf_naip_comparison.py` — pre-existing file (not created this session), modified to take a
  `cmap` parameter; NAIP panel now renders in `"Greys"`, GRF stays on the project's blue sequential ramp.

**Analysis script:**
- `analyze_beta_sweep_maps51-55_auc_only.py` — single-panel extract of
  `analyze_beta_sweep_maps51-55_combined.py`'s right-hand panel, correctly relabeled "step-averaged"
  (see §6), sized narrow/tall (4.2×5.2in) for a single IEEE column with a large bold y-axis label
  (`"Step-avg. variance"`) and a compact 3-column legend, since the original's wide single-row legend
  and long axis label don't survive being shrunk to column width.

## 8. Priority-ordered next steps

1. If continuing DPPO work: fix the two blocking bugs in §3 first (`REPO_DIR`, `DEFAULT_CHECKPOINT`) —
   trivial one-line fixes, just never confirmed/applied. Decide the right default checkpoint given the
   GRF pivot (probably `current_best_updated.pth`, not `naip_diffusion.pth` — verify provenance first).
2. Reconcile §4 (`threeDSparseTransDiffusion.py` vs. `_periodic_eval.py` batch size / dataset path) if
   you want the two to be directly comparable — currently they silently train on different data.
3. Decide whether §5's NAIP `ImitateTrans` sweep (maps 41-50) is still wanted; if so it needs explicit
   `MAPTYPE=NAIP` overrides now that defaults are GRF, and still needs actual parallelization (was
   interrupted before that redesign was finished).
4. If you want the diffusion-math explanations from §6 preserved beyond this transcript, ask for them
   to be written into the thesis document directly — they were given only in chat.
5. A tightened rewrite of the thesis abstract was also produced this session (fixed an MLP-vs-transformer
   terminology inconsistency and a muddled causal claim about *why* replanning is faster) — given only
   in chat, not saved to any file; copy into the actual thesis document if not already done.
