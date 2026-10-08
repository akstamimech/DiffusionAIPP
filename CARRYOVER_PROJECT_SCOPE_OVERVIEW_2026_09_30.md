# Carryover — Project scope overview (DIPPer / DiffusionAIPP), as of 2026-09-30

This is a **top-level orientation document**, not a replacement for the detailed carryover docs already
in this repo. Its job is to give a future agent (or future you) the big picture fast, flag the single
most persistent failure pattern in this project's history, and point to where the details live. Read
this first, then go to the specific doc named in each section for the full story.

**Existing detailed carryover docs, newest-relevant first**:
`CARRYOVER_NAIP_NORMALIZATION_BUG_AND_MULTIMODAL_INVESTIGATION.md` (the normalization bug, execution-chunk
fix, multimodal investigation start),
`CARRYOVER_NAIP_SWITCH_AND_BASELINES.md` and `CARRYOVER_CHECKPOINT_SWEEPS_DIFFUSION_VS_IMITATETRANS.md`
(earlier NAIP-era sessions), `CARRYOVER_GRF_SWITCH_DPPO_REWARD_AND_INFOGRAPHICS.md`,
`CARRYOVER_CMAES_REWARD_2.md` / `_3.md` (CMA-ES objective/hyperparameter work, earliest).

## 0. What this project actually is

**DIPPer**: a diffusion-based policy for adaptive informative path planning (AIPP). An aerial agent
maintains a Gaussian Process belief (mean + variance maps) over an unknown scalar field, updated via
Kalman-style measurement updates as it flies, and needs to decide where to look next to reduce
uncertainty fastest, especially in regions that cross an importance threshold. The core idea: instead of
running an expensive online optimizer (CMA-ES over a grid-search-seeded receding horizon) at every
replanning step, train a diffusion model via imitation to reproduce that optimizer's behavior, amortizing
its cost down to a single cheap feedforward pass (measured: ~10s/replan for live CMA-ES vs. ~0.1-0.3s for
diffusion).

**Planners compared throughout**: live CMA-ES (`CMAES_classic_singlemap.py`), the diffusion policy
(`Diffusionplanner_singlemap.py`, trained by `Diffusion/threeDSparseTransDiffusion.py`), a deterministic
imitation-learning baseline (`ImitateTrans_singlemap.py`/`ImitateTrans.py`, exactly deterministic, no
sampling noise), and non-adaptive/myopic baselines (`lawnmower_singlemap.py`, `realgreedy_singlemap.py`,
`greedygradient_singlemap.py`).

**Two map sources**, selected via `MAPTYPE`: synthetic Gaussian Random Fields (`grf`) and real NAIP aerial
NDVI imagery (`NAIP`). **This axis has been the single biggest source of silent bugs in the project's
history — see §1, read it before trusting any current-state claim.**

**Repo**: `github.com/akstamimech/DiffusionAIPP`, branch `main`, git user "Akshat". A `README.md` was
added at the repo root recently (see §6). No `requirements.txt`/`pyproject.toml` exists; core deps are
`torch numpy scipy scikit-learn matplotlib cma imageio tqdm`, plus `mpi4py` for parallel data collection.
Python 3.12.

## 1. THE recurring GRF ↔ NAIP flip pattern — read this before trusting any "current default" claim

This project has switched between GRF and NAIP configuration **at least five times**, and every single
switch has left something inconsistent that took real debugging effort to find. Do not assume the
codebase is in a clean, consistent state for either map type right now.

**The mega-finding (mid-August)**: `Diffusion/threeDSparseTransDiffusion.py` and `ImitateTrans.py` both
dynamically recompute their belief-normalization constants (`mean_center`/`mean_scale`/`var_center`/
`var_scale`/`total_variance_center`/`scale`) at import time, from whatever dataset `DIFFUSION_DATASET_PATH`/
`IMITATE_DATASET_PATH` resolves to. For the entire history of this project up to that point, this env var
was never set, and the code default silently pointed at the *wrong* dataset relative to what the live
checkpoint was trained on, causing a severe (~1.6-1.8x scale error, proven via direct loss-recomputation
to inflate loss 7.4x) silent corruption of every diffusion/ImitateTrans inference run. **The proof
methodology** (recompute a checkpoint's own training loss under both the correct and incorrect
normalization, compare against its recorded training loss) is a cheap, few-minute sanity check worth
rerunning any time a checkpoint's behavior looks suspicious. Full mechanism and fix in
`CARRYOVER_NAIP_NORMALIZATION_BUG_AND_MULTIMODAL_INVESTIGATION.md` §5.

**2026-08-20 — a deliberate, verified "complete" switch back to GRF** happened (per memory
`grf-complete-switch-2026-08-20`, not directly witnessed in this transcript). Full repo grep confirmed
`LCB=True`/NAIP `MAPTYPE` defaults purged from every pipeline script at that time. **Critically, this
switch generalized the normalization-bug lesson correctly**: rather than just flipping `MAPTYPE`/`LCB`,
the dataset-path defaults were explicitly checked against what the *live* checkpoint was actually trained
on (user-confirmed: `Diffusionplanner_singlemap.py`'s default checkpoint `current_best_updated.pth` and
`ImitateTrans_singlemap.py`'s default `Imitate_best.pth`, both dated Jul 25, both trained on
`dataset_grf_60.pt`) before setting `DATASET_PATH` defaults to match. **This is the standing procedure any
future map-type switch must follow**: never re-default `DATASET_PATH` based on MAPTYPE alone, always
verify against actual checkpoint training provenance first.

Files deliberately excluded from every switch, per standing user instruction:
`DataCollector_3D_randomstart_CMAESregularized.py` and everything under `DataCollector+datasets/` (stays
NAIP), plus one-off thesis figure/cover-art generators (`generate_hero_trajectory*.py`,
`plot_thesis_cover_topdown.py`, etc. — deliberately render NAIP because real aerial imagery looks better
for a thesis cover than synthetic GRF).

**As of 2026-09-13 (this session's own direct check, not memory)**: the state had *already drifted again*
into a mix, contradicting the "complete" Aug 20 switch: `CMAES_classic_singlemap.py` and
`lawnmower_singlemap.py` defaulted `MAPTYPE` to `"NAIP"`, while `Diffusionplanner_singlemap.py` and
`ImitateTrans_singlemap.py` defaulted to `"grf"`, and `gaussianprocesstraining.py`'s `LCB` was still
`True` (NAIP convention). **This means another undocumented partial flip happened between Aug 20 and Sep
13** that isn't captured in any memory or carryover doc. Nobody should trust either "GRF" or "NAIP" as
the current live state without re-grepping fresh.

**Standing rule, non-negotiable**: before running anything or trusting any prior claim about current
settings, grep fresh for `MAPTYPE`, `LCB`, `UTILITY_THRESHOLD`, the prior-mean sign (`+0.1` vs `-0.1`),
and the `important_mask` direction (`>`/`<=`) across whichever scripts you're about to touch, **and**
separately verify `DIFFUSION_DATASET_PATH`/`IMITATE_DATASET_PATH` against whatever checkpoint is
currently defaulted. Treat every "current state" sentence in every carryover doc, including this one, as
provisional until re-checked.

## 2. Execution-horizon mismatch (tested, fix not made permanent)

`DataCollector_3D_randomstart_CMAESregularized.py` only physically commits to `EXECUTION_CHUNK_UPDATING=20`
steps of each planned round before replanning, even though it records a longer `EXECUTION_CHUNK_SCORING=40`-step
plan as the training target. `Diffusionplanner_singlemap.py`'s own `execution_chunk` default was `30` — a
real train/inference mismatch. Tested with `EXECUTION_CHUNK=20` (matching training exactly): improved
occupied-variance performance meaningfully for every checkpoint tried, and was what first let a diffusion
checkpoint *decisively* beat Lawnmower rather than merely tie it. **As of the last direct check, this had
not been made the code default** — worth verifying whether it has been since, and applying it if not.
Full detail in the normalization-bug carryover doc §6.

## 3. CMA-ES's own distance-cost objective (`COST_EXPONENT`)

`gaussianprocesstraining.py`'s grid-search and CMA-ES-refinement scoring both structurally discount
distant high-value regions relative to travel cost (`gain/distance`), which interacts badly with NAIP's
genuinely disjoint important-region clusters (unlike GRF's single connected blob — confirmed via
connected-component analysis). Softened via `COST_EXPONENT` (env-overridable, default `0.5`, `1.0`
recovers old behavior). Effect on live CMA-ES was real but modest and mixed across metrics; its larger
value is as raw material for a *future* data re-collection that hasn't happened yet — every existing
checkpoint and `FINAL_NAIP_DATASET.pt` predate this change.

## 4. Multimodal vs. unimodal training data — now tied directly to the paper's core novelty claim

This is the most important open thread right now, because it stopped being just a code curiosity and
became load-bearing for the paper (see §7).

**The mechanism**: `DataCollector_3D_randomstart_CMAESregularized.py` records near-tied alternate CMA-ES
branches at decision points (`condition_id` groups with >1 saved trajectory — confirmed count on
`dataset_grf_60.pt`: 753 of 3840; on `FINAL_NAIP_DATASET.pt`: 608 of 2800), but **only the winning branch
ever continues the chain**. A losing branch's resulting belief state is never explored further in any
recorded chain.

**Confirmed mode-collapse findings** (on GRF checkpoints, `dataset_grf_60.pt`): diffusion's multimodality
is condition-dependent, not reliable — of 3 tested conditions at `epoch_3000`, 2 showed total collapse (0
of 80 samples ever closer to the losing branch). A real accuracy-vs-mode-coverage tradeoff was confirmed
via batch scan: `epoch_1000` collapses less often (2/20 vs. 3/3) but isn't accurately reproducing *either*
mode (mean distance to its own preferred branch, 114.1, exceeds the actual branch separation, 106.6, i.e.
it's just unconverged/scattered); `epoch_3000` is accurate (~44-47% of branch-separation distance for its
one preferred mode) but single-mode only. Consistent with plain denoising-MSE training having no explicit
incentive to preserve diversity.

**The orphaned-branch hypothesis** (does the "never continued" mechanism actively hurt performance,
especially on NAIP's disjoint-cluster geography where indecision is costly): tested directly by comparing
winner-vs-runner-up branch endpoint distances between NAIP and GRF/synthetic datasets — only a modest
~10-16% difference, **not** strong support for "NAIP branches are much more spatially divergent." The
better-supported (but not directly tested) reasoning is that the *consequence* of confusion differs by
map geography, not the branching mechanism itself.

**Action taken, not yet properly evaluated**: `FINAL_NAIP_DATASET_FILTERED.pt` was created (winner-only,
`parent_beam_index==0`, all 4 parallel per-map chains preserved, only within-round alternates removed).
`UNIMODAL_NAIP_FINAL_DIFF_1000.pth` was trained on it. **The one test run so far compared this epoch 1000
checkpoint against the ORIGINAL (multimodal-trained) epoch 400 checkpoint — a mismatched comparison, not
a clean ablation.** Result was mixed (unimodal lost to Lawnmower on occupied variance, beat it on global
variance) and should not be interpreted as "filtering hurts" or "filtering helps" either way.

**This is now the single highest-value experiment to actually run**: a matched epoch-sweep comparison
(same checkpoint epochs, same map, same seed, same settings) between models trained on
`FINAL_NAIP_DATASET.pt` (multimodal) vs. `FINAL_NAIP_DATASET_FILTERED.pt` (unimodal). It would
simultaneously resolve this long-open investigation thread **and** provide the direct empirical evidence
the paper's novelty claim currently lacks (see §7).

## 5. Where diffusion vs. CMA-ES vs. Lawnmower stood (NAIP-era numbers — re-verify under whatever map
   type the paper ultimately uses)

Established, with direct evidence, under NAIP settings before the Aug 20 GRF switch:
- Live CMA-ES clearly beats Lawnmower on variance/RMSE metrics.
- The **raw CMA-ES training demonstrations themselves** (replayed, actually-executed portions only, no
  network involved) also clearly beat Lawnmower — proving the training data is genuinely high quality;
  the remaining gap lives in the imitation-learning step, not data collection.
- Ruled out as explanations for the residual diffusion-vs-CMA-ES gap: a CMA-ES budget mismatch between
  data collection and live comparison (checked, identical budgets used in both), and a
  kernel-lengthscale-favors-Lawnmower hypothesis (directly tested by re-scoring the same trajectories
  under two lengthscales — the effect went the *opposite* direction from predicted; falsified, don't
  resurrect without new evidence).
- Best result achieved: epoch 400 checkpoint, post-normalization-fix, `EXECUTION_CHUNK=20`: decisively
  beat Lawnmower on occupied variance. **This was all measured under NAIP; given the subsequent switch
  back toward GRF, these specific numbers likely need to be re-run under whichever map type the paper
  actually reports results for.**

Full detail and the checkpoint reference table in the normalization-bug carryover doc §8-9.

## 6. README and git history

A `README.md` was written and committed at the repo root (project overview, architecture summary, setup,
usage). Per explicit user instruction, **the `Co-Authored-By: Claude` trailer was stripped from 3
historical commits** the user did not want AI co-authorship on (their repo, their call on attribution).
This required a local history rewrite (`git filter-branch`, run by the user directly since the harness's
own permission classifier blocks it as an assistant-run action) and leaves local `main` diverged from
`origin/main` by design. **Check whether the resulting force-push (`git push --force-with-lease origin
main`) to sync GitHub was ever actually run** — as of the last visible exchange on this, it had been
proposed and explained but not yet confirmed/executed.

## 7. Paper positioning (RA-L submission) — new thread, not yet in any other carryover doc

Recent work has shifted from code investigation to manuscript framing. Key positions worked out so far,
all reasoned from this project's own actual findings rather than generic phrasing:

**Contribution statement** (iteratively refined): conditions on the GP belief's mean and variance maps,
plus agent position/heading/total remaining uncertainty; generates a continuous 3D trajectory by
denoising sparse control points and spline-interpolating them; core claimed benefit is **amortizing
CMA-ES's expensive per-step optimization while avoiding the mode-averaging a deterministic policy would
be forced into** on the dataset's genuinely multimodal decision points. Explicitly **not** claiming
"faithfully learns/preserves multimodal behavior" as an achieved outcome — that's the claim §4's mode-
collapse evidence argues against. "Avoids mode-averaging" is a narrower, structurally-true claim
(diffusion committing to one mode is qualitatively different from a regressor blending two modes into an
invalid third option) that survives scrutiny where the broader claim doesn't.

**Novelty vs. "Diffusion Policy + AIPP demonstrations"** (the exact question posed by the user's advisor):
worked through all five candidate axes. Adaptive replanning against updated observations is **not**
novel, that's core to Diffusion Policy itself. The control-point-plus-spline trajectory representation is
a real but modest, non-headline detail. Conditioning on an explicit *variance*/uncertainty channel (not
just a mean/state estimate) is genuinely IPP-specific and partially novel. **The strongest, headline-
worthy axis is the training/data-generation procedure**: Diffusion Policy imitates human teleoperation;
DIPPer imitates a re-runnable, objective-driven optimizer, which is what makes genuine multimodal branch
demonstrations (near-tied alternative solutions) possible to collect at all, something impractical to get
from human demonstrators at scale. This reframes the paper as amortizing an *optimizer* rather than a
domain-transfer of an existing human-imitation architecture.

**Statistical testing plan** for "diffusion works better on average" claims: data is paired by map (same
maps evaluated by every planner), not independent samples, so paired tests are required. Two-planner
comparisons: Wilcoxon signed-rank test (nonparametric, robust to the small sample sizes and likely
non-normal metrics here), not a paired t-test. Multi-planner comparisons: Friedman test, then pairwise
Wilcoxon with Holm-Bonferroni correction (or Nemenyi), following the standard Demšar (2006) protocol for
comparing multiple methods across multiple benchmarks. Two data-specific complications to handle: (1)
diffusion has 3 seeds per map vs. 1 deterministic run for other planners — aggregate to a per-map mean
before testing; (2) ImitateTrans sometimes never reaches a given completion threshold within budget —
that's censored data, not missing data; split into a binary reach-rate test (McNemar's/Fisher's exact) and
a separate continuous-metric test on a threshold-independent quantity (e.g. final variance at a fixed
time budget). Report effect size alongside any p-value.

**Hardware experiment framing** (real drone, no real sensor, simulated sensor readings): reasoned value
proposition is that this validates the *planning-and-execution pipeline under real flight dynamics*
(replanning latency, GPS/state-estimation noise, real turn/velocity constraints), not the sensing model,
which simulation already covers extensively. Concretely suggested: feed the real flown GPS trace into the
existing `important_region_variance_from_trajectories.py` Kalman-replay tool and compare against the
simulator's predicted curve for the same planner/map (reuses trusted infrastructure directly); report
onboard/telemetry replanning latency (tests the "fast enough for real-time use" claim against real
compute/communication overhead, not just simulated timing); report trajectory-tracking error (commanded
vs. actually-flown path).

**The one action item that ties §4 and this section together**: the matched multimodal-vs-unimodal
epoch-sweep ablation is simultaneously the thing needed to close the long-open mode-collapse
investigation *and* the specific experiment that would turn "avoids mode-averaging" from an argued claim
into a demonstrated one. This is the highest-priority next step across both threads.

## 8. Explicitly declined, do not re-propose

Sensor noise scaling (`noise_model(altitude)` returns a fixed absolute variance regardless of map type,
proportionally larger relative to NAIP's narrower value range than GRF's). Raised for completeness
multiple times across sessions; declined each time. Nothing new has changed this calculus.
