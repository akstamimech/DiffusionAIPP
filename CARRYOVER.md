# Session carryover — DPPO congruence/critic-lag (§0-10) + SYNTHETIC DATA PIPELINE, convention
# flip, noise-schedule fix, multimodal maps (§11, added later — READ THIS FIRST for recent work)

Written for a future agent with no memory of this session. §0-10 cover the DPPO training pipeline
(train/deploy simulation mismatches — now fixed; reward simplification; a slow degradation
diagnosed to an under-trained critic). **§11 covers the second half of the session** — a pivot to
the SYNTHETIC (multiblob) map/data pipeline, an interest-convention flip (UCB for synthetic), the
diffusion noise-schedule cap fix, and map-generation changes for multimodality. **If you're picking
up the current thread, start at §11.**

---

## 0. FILE MAP (all important addresses)

All paths relative to `C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\`. **NOTE the flat-vs-nested
layout:** the repo has a top-level `scripts/` dir and a `scripts/Diffusion/` subdir. The HPC
deployment is often a FLAT dir (everything together); several path constants (`CSV_PATH`,
`DATASET_PATH`) are set for the flat layout and can mismatch locally — see Gotchas (§9).

### DPPO training (the focus)
- **`Diffusion/DPPOTRAINING_batched.py`** — the live DPPO trainer (GPU-batched, multiprocessing
  rollout pool). All constants, `compute_step_reward`, `collect_rollouts_batched`,
  `collect_pooled_episodes_batched`, `compute_advantages`, `update_policy`/`_ppo_step`,
  `update_value`, `evaluate_generalization`, `_save_best_checkpoint`, `train`. **UNTRACKED in git**
  (`git status` shows `??`) — no history, and it repeatedly diverges from the HPC copy.
- **`Diffusion/dppo_rollout_worker.py`** — CUDA-free reimplementation of the per-episode chunk
  rollout + Kalman/dynamics, run in the multiprocessing pool (`rollout_chunk`, `warmup_chunk`,
  `apply_measurement_update_3d`, `pool_task`). **Must be kept congruent with the deployment loop
  in `Diffusionplanner_singlemap.py`** — this session's whole point. UNTRACKED.
- **`Diffusion/DPPOvaluefunction.py`** — `ValueFunction` (critic, class at L24). UNTRACKED.
- `Diffusion/DPPOTRAINING.py`, `Diffusion/DPPOTRAINING_overnight.py` — OLD MPI variants,
  deliberately NOT touched; still have the original RNG-collision bug and none of this session's
  fixes. Do not use.

### Deployment / evaluation (GROUND TRUTH for the simulation)
- **`Diffusionplanner_singlemap.py`** — single-map deployment/eval. **The user declared this the
  correct/authoritative simulation; DPPO training was bent to match it, not vice-versa.** Key spots:
  main flight loop (~L418-590), `sample_diffusion_trajectory` (~L95-175),
  `apply_measurement_update_3d` (~L189), `build_true_map_flat`. Start pose `(4,4,INIT_ALTITUDE)`
  (L387), `execution_chunk` default now **10** (L44, env-overridable via `EXECUTION_CHUNK`),
  `diffusion_path` toggled between BC and `dppo_best.pth` (L67-68, currently both commented — check
  which line is active before running).

### Shared physics / GP / metrics
- **`gaussianprocesstraining.py`** — `initialize_gp` (L1303; **map-independent** GP prior — same
  `P0`/`trace(P0)=26.53` for every map), `kalman_update` (L1283; covariance update depends ONLY on
  measurement geometry, not measured values), `build_spline_trajectory_3d` (L1219; scipy natural
  cubic spline used by training), `importance_filter` (L348; the UCB interest mask CMA-ES uses),
  `build_sensor_matrix`, `noise_model`, deps of `compute_fov`.
- **`evalmetrics.py`** — `compute_reconstruction_rmse` (L85; returns `global_rmse`, `occupied_rmse`,
  `occupied_mask`=`true_map<utility_threshold`), `compute_rmse_time_metrics` (L145;
  `auc_rmse`/`mean_rmse` — the eval is AUC-shaped over time).
- **`CMAES_classic_singlemap.py`** — `compute_fov` (L221), `dynamics_3d` (L202), `waypoint_3d`
  (shared by the training worker AND deployment).

### Diffusion model
- **`Diffusion/threeDSparseTransDiffusion.py`** — the BC-pretrained diffusion policy: model
  (`NoisePredictor`, `MeanVarMarkerCNN`, attention blocks), forward process / **noise schedule**
  (`cosine_beta_schedule` L86, `T=20` L112), samplers (`ddpo_ddim_sample_timestep` L1013 used in
  TRAINING; `ddim_sample_timestep`/`ddim_sample` used in DEPLOYMENT), `get_loss` (ε-prediction,
  L952). `DATASET_PATH` L123 = `scripts/CMAES_beamsearch_dataset_3d_randomstart_multimodal.pt`
  (mmap-loaded at import). Also `Diffusion/threeDSparseTransDiffusion_periodic_eval.py` (T=20 copy).
- Sibling/older diffusion files (share the SAME beta-cap line — see §7): `Diffusion/SparseTransDiffusion.py`
  (T=30), `Diffusion/SparseDiffusion.py` (T=70), `Diffusion/old/UNETdiffusion.py` &
  `Diffusion/old/MLPdiffusion.py` (T=1000).

### Data / checkpoints / literature
- `csv/map_{0..64}_NAIP_grid_counts.csv` — 65 NAIP maps (51×51 grid, step 2.0). **Local maps live
  at `scripts/csv/`, NOT `scripts/Diffusion/csv/`** (but `CSV_PATH=SCRIPT_DIR/"csv"` points at the
  latter for the flat HPC layout — mismatch locally).
- `Diffusion/behaviouralcloninginstance/current_best_bc.pth` = `DEFAULT_CHECKPOINT`, the BC weights
  DPPO fine-tunes from (also copied to `Diffusion/checkpoints/current_best_bc.pth`).
- `Diffusion/checkpoints/` — `dppo_best.pth` (val-gated best), `dppo_iteration_00{65,80,125}.pth`,
  `sparse_trans_waypoints_epoch_1000_multimodaltest.pth` (a BC checkpoint used in this session's tests).
- `scripts/CMAES_beamsearch_dataset_3d_randomstart_multimodal.pt` — BC training dataset. Has a
  stored `initial_heading_velocity` key; its heading magnitude (median ~0.24 normalized) is the
  convention the model was BC-trained on.
- `literature/2512.02535v2.pdf` — **AID paper** (Lew et al., "AID: Agent Intent from Diffusion for
  MAIPP"), the closest published match to this thesis. AID code: `github.com/marmotlab/AID`
  (public; key files `model/diffusion/diffusion_ppo.py`, `execution/worker/ft_diffusion_worker.py`,
  `classes/env/env.py` step_coord reward, `config/finetune/ft_ppo_diffusion_unet_gpipp_delta_maippdata.yaml`).
- SLURM logs this session: `slurm_10436057_training_metrics.csv` /
  `slurm_10436057_ppo_diagnostics.csv` (in the user's Downloads) — the degrading run analyzed in §6.

---

## 1. CURRENT CONFIG (local `DPPOTRAINING_batched.py`, may drift from HPC — always confirm)

```
EXECUTION_CHUNK = 10        # NOW a SPLINE-INDEX threshold (see §5.2), matches deployment's 10
ENV_HORIZON = 40            # replans per episode  (was 15 earlier this session)
NUM_FINE_TUNE_STEPS = 12    # of T=20 denoising steps trained
NUM_EPISODES_PER_ITERATION = 32
NUM_ROLLOUT_WORKERS = 32    # => --cpus-per-task must ~match this, NOT the map-pool size
TRAIN_MAP_IDS = [0..44]     # 45 maps ; VAL_MAP_IDS = [45..49]
VARIANCE_WEIGHT = 1.0 ; CONCAVITY_FACTOR = 0.5 ; UCB_BETA = 1.0 (UNUSED now)
NUM_PPO_EPOCHS = 7 ; MINIBATCH_SIZE = 1024 ; CLIP_EPSILON = 0.03
CLIP_FRACTION_EARLY_STOP = 0.30 ; TARGET_KL = 1.0 ; MAX_GRAD_NORM = 0.5
NUM_VALUE_EPOCHS = 7 ; VALUE_LR = 1e-3   # <-- reverted from 1e-4 this session, see §6
ENV_GAMMA = 0.999 ; GAE_LAMBDA = 0.95
SIGMA_PROB_MIN = 0.3        # likelihood-variance floor (PPO ratio)
SIGMA_SAMPLE_MIN = 0.05     # sampling-noise floor (exploration) — DIVERGES from deployment, see §8.3
LOGPROB_CLAMP = [-5.0, 2.0] # per-dim log-prob clamp, both old & new sides
N_CRITIC_WARMUP_ITERATIONS = 2
START_XY = 4.0 ; INIT_ALTITUDE = Z_MIN = 10.0 ; WARMUP_GOAL_XY = 80.0 ; WARMUP_STEPS = 2
```
Memory: ~17GB total footprint at 32 episodes (2601×2601 float64 covariance = 54MB × 32 copies in
main + ~400MB/worker, CUDA-free workers). **1GB/CPU is enough IF --cpus-per-task ≈ 32.**

---

## 2. THE CENTRAL PROBLEM (mostly resolved this session)

DPPO improved on training AND held-out VAL maps but **underperformed plain BC at deployment**
(`Diffusionplanner_singlemap.py`). Key realization: `evaluate_generalization` reuses the SAME
`collect_rollouts_batched` that training optimizes, so a good val curve only proves the policy got
better **at the training rollout mechanism** — it CANNOT detect a train/deploy *simulation*
mismatch. We found several such mismatches (§5). The map-generalization angle from the prior
carryover turned out secondary to these sim mismatches.

---

## 3. VERIFIED FACT used repeatedly (keep in mind)

`kalman_update`'s covariance update `P_after = f(sensor geometry)` depends ONLY on WHERE you fly,
not on measured values or the map — so **variance reduction is an exact, deterministic function of
a candidate trajectory** (measurable offline, no model/dataset needed). This enabled cheap
numerical experiments on the real GP prior on this CUDA-less dev machine.

Also measured: whole-grid variance has a **2.14× center bias** — a central measurement reduces ~2×
more raw total variance than a corner one, purely from finite-grid FOV/kernel edge geometry
(interior footprint never clipped by the boundary). Relevant to reward design (§6).

---

## 4. AID PAPER — what it actually does (from its code, not just the paper)

- **Best-of-N is INSIDE their training loop:** every replan samples `num_paths=5` trajectories,
  scores each with an exact cov-trace predictor, executes the winner, stores ONLY the winner's
  chain in the PPO buffer. Every AID number (incl. BC baselines) has this selection baked in.
- **Procedurally-infinite training maps** (8–12 random Gaussians per episode), never a fixed pool.
- **Reward terminal-only, belief-masked:** `if done: reward = -5*(cov_trace/cov_trace0)**0.5`,
  cov_trace over the UCB "high-info area" `{mu+beta*sigma >= threshold}`.
- **Their own gains are modest:** ~17% vs a weak expert, PARITY vs a strong expert; headline is a
  4× planning-time SPEEDUP, not a quality jump. → "mild improvement over BC" is the field's honest
  result, not a failure. Thesis-worthy framing.
- PPO plumbing: clip 0.001, log-prob clamp [-5,2], sampling-std floor 0.05, logprob-std floor 0.1,
  advantage quantile-clip [5%,95%], critic warmup 2, actor lr 1e-5, critic lr 1e-4, gamma 0.999,
  log-probs MEAN-reduced over action dims (this code SUMs over 24 dims — different scale).
- Best-of-N: user VETOED for our pipeline (adds wall-clock; eval is wall-clock-constrained).

---

## 5. TRAIN/DEPLOY SIMULATION FIXES (done + VERIFIED byte-identical this session)

Deployment = ground truth. Live in `dppo_rollout_worker.rollout_chunk`/`warmup_chunk` +
`collect_rollouts_batched`. All verified against the real deployment loop with matching numbers.

**5.1 Heading conditioning.** Training fed the NET displacement over the whole chunk; deployment
feeds the LAST single ~2m step before replan. Normalized magnitudes differed ~18× (0.7 vs 0.04) —
the model's `initial_heading_velocity` input was far OOD at deployment. FIX: `rollout_chunk`
returns `last_step_heading`; `collect_rollouts_batched` uses it (removed old `previous_poses` net
calc). Verified identical.

**5.2 Replan cadence (BIGGEST).** `execution_chunk` meant MEASUREMENT-COUNT in training
(`for _ in range(40)`) but SPLINE-INDEX in deployment (`spline_idx >= execution_chunk`). At
~1.9 measurements/spline-point, training executed only ~HALF of each planned trajectory before
replanning; deployment executes ~all of it. FIX: `rollout_chunk` loops
`while spline_idx < chunk_size`. Verified same final pos + measurement count. **`EXECUTION_CHUNK`
is now a spline-index count and MUST equal deployment's `execution_chunk`** (both 10).

**5.3 Padded bounds.** Deployment clamps the dense path to `[xmin+2*step, xmax-2*step]`; training
didn't. FIX: `rollout_chunk` clamps `dense_path` to the padded region.

**5.4 Initial position + warmup.** Deployment starts corner `(4,4,10)`, flies toward `(80,80)` for
2 measurement steps (`ts<=1` branch) BEFORE the first replan. Training started center `(50,50)`,
arbitrary heading. FIX: `build_environment` starts at `START_XY=4`; new `warmup_chunk` replicates
the 2-step approach (buffer=step*2, one measurement/step). Verified BYTE-IDENTICAL first-replan
state (pos 8,8,10 / heading 2,2,0 / mu,P allclose).

Confirmed already-equivalent (no fix): `apply_measurement_update_3d` (compute_fov `step` arg
vestigial), the two spline fns, samplers for the current checkpoint. Still-open config congruence:
episode length (training 40 replans vs deployment's step budget) differs — flag, user's call.

---

## 6. REWARD EVOLUTION → CURRENT STATE + LATEST DIAGNOSIS

Reward went: dense multi-term (RMSE+variance+step-penalty) → dense global+occupied variance blend
→ terminal-only belief-masked variance → terminal-only pure whole-grid variance → **[CURRENT]
DENSE pure whole-grid variance**. Per user: "reduce variance as much as possible; BC already imbues
map-specific behaviour."

**CURRENT reward (`compute_step_reward`):** every env step returns
`reward = -VARIANCE_WEIGHT * (variance_after_global / baseline_total_variance)**CONCAVITY_FACTOR`.
Whole-grid trace, NO mask, dense. `baseline_total_variance = trace(P0)` is map-independent so the
reward is the SAME function on every map. RMSE terms dropped. `UCB_BETA` retained but UNUSED
(masked/terminal variant is a documented one-branch restore in the docstring). Center-bias caveat
(§3): now DENSE per user request, so **watch `plot_iteration_trajectory` for center-camping**.
`mean_reward` logging is mean-over-terminal-rows (dense).

**LATEST RUN (slurm_10436057, 80 iters, dense pure-variance):**
- Symptom: variance_fraction 0.203 → **0.189 peak @ iter ~34** → DEGRADED to 0.231 by iter 79
  (worse than iter 0). Reward + value-loss followed the same arch.
- **NOT a trust-region blowup** (first hypothesis, refuted by data): `clip_fraction` steady ~0.24
  (early-stop@0.30 barely fires), `approx_kl` ~0.0005 (2000× under target). Actor well-behaved —
  DON'T touch NUM_PPO_EPOCHS/actor-LR/CLIP_EPSILON.
- **Real cause = under-trained/lagging CRITIC.** `value_loss` bottomed ~0.6-1.0 (iter 6-13) then
  climbed to ~3.0. Value batch = 32×ENV_HORIZON terminal rows; with MINIBATCH_SIZE≥that, each iter
  does ~1 full-batch step × NUM_VALUE_EPOCHS=7 = ~7 tiny gradient steps. At the (this session's)
  `VALUE_LR=1e-4` the critic couldn't track drifting returns → biased advantages → policy slowly
  walked off the good BC prior.
- **FIX APPLIED: `VALUE_LR` reverted 1e-4 → 1e-3** (the LR earlier val-improving runs used).
  **Sync to HPC + re-run**; pull the two CSVs. If `value_loss` stays flat and variance_fraction
  stops degrading post-peak, confirmed.
- Next levers if it persists: raise `NUM_VALUE_EPOCHS` (7→15); then, if drift remains with a
  healthy critic, add a BC anchor (AID `use_bc_loss`/KL-to-BC — this code has NONE). Peak gain was
  only ~7% → expect modest gains regardless; val-gated `dppo_best.pth` should capture the ~iter-34
  peak (verify it wasn't overwritten by the later collapse).

**Other suggestion-package changes this session (kept):** per-dim log-prob clamp [-5,2] (both ratio
sides), advantage quantile-clip [5%,95%] after standardization, critic warmup 2, `ENV_GAMMA`
0.99→0.999, `SIGMA_SAMPLE_MIN=0.05`. Best-checkpoint selection fixed to gate on
`evaluate_generalization`'s HELD-OUT val variance-fraction (not the training-batch metric).

---

## 7. NOISE-SCHEDULE β-CAP FINDING (real, inherited, non-fatal — decide before thesis)

`cosine_beta_schedule` ends `torch.clamp(betas, 1e-4, 0.05)`. The `0.05` cap is copy-pasted
verbatim from the original T=1000 files down to the current T=20. At T=1000 harmless (binds 4%,
ᾱ_T≈0.0005 ≈ pure noise). At **T=20 it binds 80% of steps (16/20), ᾱ_T=0.396** → the forward
process NEVER reaches noise; terminal noised action retains √ᾱ_T=0.63 signal. Sampling starts from
pure `randn` → non-zero-terminal-SNR train/inference mismatch (Lin et al. 2024). All checkpoints
trained with it. Milder for a conditioned policy; model copes. Affects BC and DPPO EQUALLY (NOT the
transfer bug). FIX = raise cap so ᾱ_T→0 (cap 0.999 → ᾱ_T≈0) **but requires retraining BC from
scratch** then re-running DPPO. Recommendation: fold into any future BC retrain; don't retrain
solely for it without measuring it costs something.

---

## 8. OPEN ITEMS / PRIORITIES

1. **Re-run with `VALUE_LR=1e-3`** and confirm the critic-lag diagnosis (§6). Highest priority —
   one-line revert of this session's regression.
2. **Keep `EXECUTION_CHUNK` (training) == `execution_chunk` (deployment)** now both mean
   spline-index (both 10). Change one → mirror the other.
3. **`SIGMA_SAMPLE_MIN=0.05` diverges from deployment** (its `ddim_sample_timestep` has no sampling
   floor). Train-noisier-than-deploy is standard RL and only affects noise (not the mean) — likely
   benign — but match/drop if strict congruence wanted.
4. **Center-camping watch** on the dense whole-grid reward (plot trajectories early).
5. **BC anchor** (AID `use_bc_loss`/KL-to-BC) if slow drift persists after the critic fix.
6. Map pool already widened to 45 train / 5 val. Procedural `multiblob` maps (AID-style unbounded
   diversity) still an option, not done.
7. Noise-schedule cap (§7) — decide for thesis.

---

## 9. GOTCHAS

- **`DPPOTRAINING_batched.py` / `dppo_rollout_worker.py` / `DPPOvaluefunction.py` are UNTRACKED in
  git** and repeatedly diverge from the HPC copy. This session found the LOCAL copy already edited
  to `EXECUTION_CHUNK=10`, `ENV_HORIZON=40`, `NUM_FINE_TUNE_STEPS=12`, `MINIBATCH_SIZE=1024` (not my
  edits). **ALWAYS confirm actual HPC constant values before reasoning about a log or proposing a
  change.**
- **Flat-vs-nested layout:** `CSV_PATH=SCRIPT_DIR/"csv"` and `DATASET_PATH=SCRIPT_DIR.parent/"...pt"`
  are set for the flat HPC dir; locally maps are at `scripts/csv/` and the dataset at `scripts/`.
- **`--cpus-per-task` tracks `NUM_ROLLOUT_WORKERS`/`NUM_EPISODES_PER_ITERATION` (=32), NOT map-pool
  size.** Widening maps needs no extra CPUs. 1GB/CPU enough at cpus≈32.
- Dev machine has **no CUDA**; GPU/PPO paths can't run here, but the NumPy Kalman/GP/spline paths
  CAN (used for all verification this session).
- `dppo_rollout_worker.py` must be manually kept in sync with `Diffusionplanner_singlemap.py`.

---

## 10. FOLDED FORWARD from earlier carryovers (not touched this session, still relevant)
- Retour-loop finding: deterministic ImitateTrans gets stuck in repeat-loops stochastic diffusion
  escapes; architectural (no escape hatch), not data-curation. Untested: inference-time jitter.
- `Diffusion/3DSparseTransDiffusion.py` (digit-first, stale draft) still flagged for deletion.
- Eval ideas not executed: gap-closure %, multimodality probe, compute-matched CMA-ES,
  replan-every-step, RIG-tree baseline.
- Multimodal dataset predates the CMA-ES step-size fix and lattice z-snap fix — consider
  re-collecting before more BC training.
- Prior-session bugs already FIXED (don't re-investigate): RNG seed collision (now
  `np.random.SeedSequence`), `DPPOvaluefunction` bad import, RolloutBuffer CPU/GPU device
  mismatches, shared-GP-prior memory optimization in `build_environment_pool`.

---

# 11. SYNTHETIC DATA PIPELINE + CONVENTION FLIP + NOISE-SCHEDULE FIX (second half of session)

The session pivoted from DPPO to preparing a fresh BC retrain on SYNTHETIC (multiblob) maps. The
plan is: fix the maps → fix the noise schedule → collect expert data → **retrain BC from scratch**
→ re-run DPPO. Several fixes were staged to ride along in that retrain. **A BC retrain is now
REQUIRED** (the noise-schedule change below invalidates all existing checkpoints).

## 11.1 Noise-schedule β-cap — FIXED (both files, done this session)
`cosine_beta_schedule`'s `torch.clamp(betas, 1e-4, 0.05)` → **`0.999`** in BOTH
`Diffusion/threeDSparseTransDiffusion.py` (~L92, live model) and
`Diffusion/threeDSparseTransDiffusion_periodic_eval.py` (~L128). The `0.05` cap was written for
T=1000 and was wrong at the current `T=20`: it bound ~80% of steps and held `ᾱ_T=0.396`, so the
forward process never reached noise (63% signal survived at the terminal step) while sampling
starts from pure `randn` — a non-zero-terminal-SNR mismatch (Lin et al. 2024). At `0.999`, only the
last step is capped, cosine shape preserved, `ᾱ_T≈1e-5` (verified). **CONSEQUENCE: invalidates every
existing BC/DPPO checkpoint — must retrain BC then re-run DPPO.** Older siblings
(`SparseTransDiffusion.py` T=30, `SparseDiffusion.py`, `old/*`) still have the 0.05 cap — only
matters if those are trained. If you want the citable version, replace with Lin et al.'s exact
zero-terminal-SNR √ᾱ rescale.

## 11.2 Interest-convention flip: SYNTHETIC = UCB (high-is-interesting), NAIP = LCB (low)
The multiblob task is INVERTED vs NAIP: the interesting cells are the HIGH-value blobs, not low
ones. Controlled by a module-global **`LCB` flag, duplicated in TWO files** (no single source):
`gaussianprocesstraining.py:63 LCB=False` and `evalmetrics.py:2 LCB=False` — both False now = UCB.
- `importance_filter` (gaussianprocesstraining ~L348): `LCB==True` → `mu-β·σ<=thr` (low);
  `LCB==False` → `mu+β·σ>=thr` (high/UCB). **Default threshold changed to 0.2.**
- `compute_reconstruction_rmse` (evalmetrics ~L130): `occupied_mask = true_map<=thr` (LCB) vs
  `>=thr` (UCB). So the RMSE metric flips too.
- The data collector uses these SHARED functions → the flip is auto-reflected in planning AND
  metrics. ✅
- **⚠️ NOT auto-reflected — the prior-mean init `mean = utility_threshold ± 0.1`** is a SEPARATE
  hardcoded convention (NOT behind LCB). Under UCB it must be `+0.1` (cells start plausibly-high →
  interesting). **Flipped to `+0.1`** in `CMAES_classic_singlemap.py:276`,
  `greedygradient_singlemap.py:159`. **STILL `-0.1` (old, INCONSISTENT)** in:
  `DataCollector_3D_randomstart_multimodal.py:819`, `Diffusionplanner_singlemap.py:358`,
  `ImitateTrans_singlemap.py:319`, `lawnmower_singlemap.py:171`, `Diffusionplanner_PPO.py:343`.
  These NEED flipping to `+0.1` for the synthetic task, else the belief states differ between
  expert/collector/deploy (a train/deploy mismatch). **NOT YET DONE** — user to confirm before
  flipping the deployment-side ones.
- **Threshold value**: user wants **0.25**. Collector has `utility_threshold=0.3` (L94), NOT yet
  changed. Threshold is defined per-file (importance_filter default 0.2, DPPO, each single-map
  script) — set it wherever it matters.

## 11.3 Multiblob value normalization to [0,1] — DONE (script written)
Raw multiblob counts span ~[0, 525] (from groundtruthgrid's radius-8 density kernel), ~400-600x the
NAIP scale. `normalize_multiblob_maps.py` (NEW, at `scripts/` root) applies a GLOBAL min-max
(`v/global_max`) → `map_{id}_multiblob_normalized_grid_counts.csv`. Ran on the OLD 1001 maps
already. Result: std≈0.13 (≈NAIP's 0.134) but mean≈0.11 and **~90% of cells < 0.3** (right-skew +
outlier max squashes the bulk low). Under UCB (high-is-interesting) that's fine — the sparse high
blobs are the targets. If wanted more NAIP-like, switch to a 99th-pctile max (option in that convo).
**Re-run this AFTER regenerating maps** (it globs all multiblob_grid_counts; scope to the new set).

## 11.4 Lengthscale decision — use NAIP's ℓ=4.79 on multiblob (NO code change)
Multiblob true ℓ≈17 (kernel fit), NAIP ℓ≈4.79. Decision: DON'T retrain the lengthscale — keep the
NAIP-calibrated belief (misspecified-but-CONSERVATIVE on multiblob) so (a) it stays consistent with
the trained model's conditioning and (b) it's calibrated to the real target (NAIP). `initialize_gp()`
DEFAULTS to ℓ=4.79/σ_f=0.101, so collect→train→deploy all share it with zero edits. Provenance of
those constants: `Kerneltraining/kerneltrainingexperiments.py` (fits Matérn-3/2 to NAIP maps,
reports median ℓ=4.79 — note: only ℓ is actually optimized; σ_f is empirical std, ν fixed 1.5),
`Kerneltraining/initialvariancetest.py` (visualizes the prior variance), `Kerneltraining/globalmean.py`
(arcpy raster global mean).

## 11.5 Map-generation pipeline + MULTIMODALITY (assetplacement.py edited this session)
Chain to regenerate maps: **`assetplacement.py` → `groundtruthgrid.py` → `normalize_multiblob_maps.py`**.
- `assetplacement.py` (`scripts/` root): places asset POINTS → `map_{id}_multiblob.csv`.
  `number_of_maps=200`. **EDITED this session for multimodality**: the generation loop now makes
  **3-5 EQUAL, well-separated blobs** (min-separation 25 units via rejection sampling), equal
  `intensity=200` points each, isotropic `sample_centre` (scale 7). Rationale: multimodal expert
  paths need SEVERAL EQUAL, SEPARATED targets so the visiting ORDER is a free choice → many paths,
  same variance reduction. The prior 20-blob / variable-intensity versions were unimodal (dense
  blanket, or a dominant blob). CAVEAT: multimodality only survives if the CMA-ES expert isn't
  strongly time-preferring (a heavy step-penalty breaks the ties). Verified via preview — produces
  3-5 clean separated equal blobs.
- `groundtruthgrid.py` (`scripts/` root): points → grid via a **radius-8 density kernel** (each
  point adds +1 to every grid cell within radius 8; that's the ~200x count inflation). Writes
  `map_{id}_multiblob_grid_counts.csv`, `dataset_id_range=1000`. **User said DO NOT change this
  file** (it's the "afterward" converter). Its radius-8 kernel is a 2nd smoothing scale that
  dominates the effective ℓ (floors it ~8) — so shrinking blobs alone won't push ℓ below ~8 without
  also lowering this radius, but user chose not to touch it.

## 11.6 Data collector (DataCollector_3D_randomstart_multimodal.py) — mostly ready
- `MAPTYPE="multiblob_normalized"` (L101) → loads `map_{id}_multiblob_normalized_grid_counts.csv`
  (L735). ✅  `initialize_gp()` default ℓ=4.79 (L1017) ✅ (matches §11.4).
- Output: `CMAES_beamsearch_dataset_3d_synthetic_final.pt`, chunks in
  `CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks/`. Uses shared `importance_filter`
  (UCB via LCB) ✅.
- **TO DO before collecting**: (a) `utility_threshold 0.3 → 0.25` (L94); (b) prior-mean init
  `- 0.1 → + 0.1` (L819, see §11.2); (c) set map range `initial_map`/`mapcount` (currently 1..25)
  to cover the new maps (e.g. 0..199); (d) ensure the maps exist (run the §11.5 chain first).

## 11.7 Thesis-writing artifacts produced (reference, not code)
- A paper-level SIMULATOR description (GP belief + altitude-dependent sensor + Kalman). Key facts:
  51×51 grid @2m, Matérn-3/2 (ℓ=4.79, σ_f=0.101), z∈[10,40]; FOV square half-width `z·tan(30°)`;
  resolution block `b(z)=1/2/4`; noise `R(z)=0.01→0.04` saturating + BLOCK-CORRELATED; Kalman =
  recursive GP regression (equiv. to the batch posterior-cov Eq.3, generalized to block-averaging
  H and structured R). **Covariance update depends only on sensor geometry, not measured values**
  → variance reduction of any path is deterministic/computable offline (used for all this session's
  numerical experiments; basis for best-of-N and the low-variance reward).
- Natural cubic spline (`bc_type="natural"`) flagged as NOT ideal for a real UAV (zero-accel
  endpoints, no velocity/heading continuity across replans) but fine for the kinematic point-mass
  sim; a hardware version would use minimum-snap / velocity-continuous replanning.

## 11.8 OPEN / NEXT (synthetic pipeline)
1. Regenerate maps: run `assetplacement.py → groundtruthgrid.py → normalize_multiblob_maps.py`
   (assetplacement edit is applied + verified; the chain hasn't been RUN yet).
2. Collector edits (§11.6 TO DO): threshold 0.25, prior-mean +0.1, map range, then collect.
3. Decide + apply the prior-mean `+0.1` flip on the deployment-side scripts (§11.2) for full
   convention consistency.
4. **Retrain BC** (now mandatory post-schedule-fix) on the new synthetic dataset, then re-run DPPO
   (which also picks up the §0-9 DPPO fixes + the `VALUE_LR=1e-3` revert).
5. Watch: does the CMA-ES expert actually produce MULTIMODAL paths on the new maps? If not, the
   expert objective is likely too time-preferring (step penalty breaks the equal-target ties).
