# Carryover — step-size multi-map validation resolved, thesis Q&A

Written for a future agent with no memory of this session. This is a short, incremental follow-on to
`CARRYOVER_CMAES_REWARD_2.md`, written earlier in the same session - **read that file first**, it has
the full context (the CMA-ES cost-normalization/boundary-penalty bugfixes, the lattice densification,
the DataCollector changes, the multi-map beta/maxiter/step-size sweep infrastructure, the thesis writing
done, and the full git/repo state). Everything in `_2` still stands; this file only records what
happened *after* it was written - one open item from its priority list got resolved, plus two
Q&A exchanges worth preserving.

## 0. Repo/git state delta since `_2`

Still no commits made. Two new untracked files, same pattern as everything else this session (not
committed, not backed up elsewhere):
- `sweep_cmaes_stepsize_maps56-59.py` - same 5-point step-size sweep as
  `sweep_cmaes_stepsize_map55.py` (from `_2`), extended to maps 56-59, parallelized
  (`ProcessPoolExecutor`, 8 workers, same safe pattern as the other multimap sweeps).
- `analyze_stepsize_maps55-59.py` - combines map 55's existing results with the new 56-59 results,
  averages across all 5 maps, writes `results_cmaes_stepsize_maps56-59/analysis/stepsize_sweep_maps55-59_table.tex`.
- New results directory: `results_cmaes_stepsize_maps56-59/` (raw outputs + analysis).

## 1. The map-55 step-size finding did NOT generalize the way it first looked - resolved, item #2 from `_2`'s priority list

`_2` flagged this as the top open methodological question: does the clean, monotonic map-55-only result
(1.5/1.2 wins on every metric, degrading roughly monotonically as step size grows) replicate across
other maps, or was it another single-map artifact like beta? Ran the same 5-point sweep
((1.5,1.2), (6.0,4.8), (10,4), (20,8), (40,16), single run each, maxiter=20, beta=1, TIMEALLOTED=200)
on maps 56-59 and combined with the existing map-55 data. **Answer: partially generalizes, but the
full picture is more nuanced than "smaller step size wins" - and the per-map view actively disagrees
with the averaged view.**

**Averaged across all 5 maps (55-59):**

| step size | mean final | std final | mean % reduced | mean AUC (timestep) | std AUC |
|---|---|---|---|---|---|
| 1.5/1.2 | **3.30** | **0.31** | **76.80%** | 7.29 | 0.80 |
| 6.0/4.8 | 3.46 | 0.38 | 75.76% | 7.44 | 0.65 |
| 10/4 | 3.39 | 0.48 | 76.39% | **7.28** | 0.79 |
| 20/8 | 3.53 | 0.74 | 75.28% | 7.49 | 0.88 |
| 40/16 | 4.76 | 0.78 | 66.67% | 8.32 | 1.10 |

1.5/1.2 wins the mean on final variance and % reduced; 10/4 essentially ties it on mean AUC (7.28 vs
7.29, within noise). **40/16 is unambiguously, robustly worst on every single metric with no
exceptions across all 5 maps** - this is now a fully validated, trustworthy finding (an oversized step
really is bad, not a single-map fluke).

**But the per-map winner (final variance) tells a different story**: 55->1.5/1.2, 56->20/8, 57->10/4,
58->10/4, 59->20/8. **1.5/1.2 is the single-map winner only once out of five** - 10/4 and 20/8 each win
twice. What actually makes 1.5/1.2 win the *average* despite rarely winning individually is
consistency: its map-to-map std (0.31) is less than half that of 20/8's (0.74). 20/8 swings between
being the best on some maps and nearly as bad as the oversized 40/16 case on others (map 58: 4.82,
worse than 1.5/1.2's 3.76 there and not far from 40/16's 5.22). **The defensible framing, if this goes
in the thesis: 1.5/1.2 is the lower-risk, more consistent default; 10/4 and 20/8 are higher-variance
but win more often individually; 40/16 is simply and robustly worse, unconditionally.** This is a
materially different (and more honest) claim than either "1.5/1.2 is best" or "no winner" - don't
collapse it to either of those simpler framings without the per-map caveat, since the mean and the
per-map mode point in different directions here.

Both the map-55-only table and the maps-55-59-averaged table exist and were both given to the user in
LaTeX form this session - if only one goes in the thesis, the averaged one
(`stepsize_sweep_maps55-59_table.tex`) is the one backed by proper multi-map validation; the map-55-only
one should probably not be used alone given how much the per-map breakdown complicates its clean story.

## 2. Two Q&A exchanges worth preserving (not code changes, but thesis-relevant content)

**Table metric clarification**: confirmed and worth remembering for any future table - "final variance"
in all of these step-size/beta sweep tables means ground-truth *occupied-region* variance
(`_important_mask_for_map`, value > 0.5), never global/total variance. The column header alone in the
LaTeX tables just says "Final variance" without the word "occupied" - fine given the caption states it,
but worth tightening the header itself if precision matters more than table width.

**Why oversized CMA-ES step sizes hurt (Popovic's own reasoning, expanded)**: Popovic's paper states
only briefly that oversized steps (their (10,10,12) case) cause "erratic paths" from "high exploratory
behavior." Expanded this into four concrete mechanisms, worth reusing if this needs explaining again:
(1) it discards the informative grid-search warm start, since the first generation scatters almost
uniformly regardless of where the warm start was; (2) independently-scattered waypoints produce
geometrically incoherent spline paths (zigzagging/backtracking), which is the literal meaning of
"erratic path," not just worse information gain; (3) **this codebase's now-fixed cost-normalized
objective (`gain/cost`, `_2` §1) specifically penalizes this** - a scattered path costs more to fly, so
it's scored worse by the same mechanism this whole session's investigation was about, which is a nice
direct tie-in between the two threads; (4) general CMA-ES theory - an oversized sigma0 wastes early
generations on a coarse search before the strategy's own covariance adaptation shrinks it back to a
useful scale, and under a fixed, small iteration budget (45, tied to a real-time replanning constraint),
there may not be enough remaining iterations to converge as tightly as a well-scaled sigma0 would have.

## 3. Updated priority list (supersedes `_2`'s §10)

1. Still top priority, unchanged from `_2`: **commit `DataCollector_3D_randomstart_CMAESregularized.py`**
   and the new sweep/analysis scripts - still completely untracked in git.
2. ~~Resolve the beta/step-size 3-way split~~ - **step-size side now resolved, see §1 above.** Beta's
   own multi-map result (`_2` §5) is still genuinely "no robust winner, map-dependent" with no further
   nuance found - that one doesn't need revisiting the same way step-size did, since beta's per-map
   winners were already all different with no consistency pattern like step-size's 10/4-and-20/8
   clustering.
3. Disentangle the diversity-regularization ensemble-vs-penalty question (`_2` §7) - still open, not
   touched this session.
4. Existing DataCollector dataset still reflects the pre-fix objective, still not re-collected, still
   not decided (`_2` §10 item 4) - unchanged.
5. Stash (`_2` §10 item 5) - unchanged, still likely droppable rather than needed.
6. Copy the thesis paragraphs (`_2` §8, plus the oversized-step-size mechanism explanation from §2 above)
   into the actual thesis document - still not done, still only exists in conversation transcripts.
