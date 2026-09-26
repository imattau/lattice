# Phase 2 (Toy-Scale Analog): IsoFLOP Sweep and Scaling-Law Fit

Status: **AR and AR+JEPA-auxiliary have nearly identical scaling exponents
at this toy scale (beta ratio 0.989, alpha ratio 0.970 — both within ~3%
of 1.0). No detectable data- or parameter-scaling advantage from the
auxiliary JEPA loss.** This is a clean, if negative, result: the plan's
Phase 2 decision-gate thresholds (beta ratio < 0.5, alpha ratio > 0.8) are
built around the idea that a JEPA-touching Core would show a *qualitative*
data- or parameter-efficiency edge over an AR baseline. At this scale, on
this corpus, it doesn't — the two arms scale the same way.

## Why this exists, and why it isn't the plan's actual Phase 2

PLAN.md's Phase 2 trains a 338M-parameter Core against a matched AR
baseline across 5-7 model sizes (50M-600M params) and 3-4 data budgets
(up to 128B tokens), at an estimated 10-30 PF-days of compute — many
orders of magnitude beyond what a single consumer GPU can produce. Rather
than attempt a version of that and silently undershoot it, this is an
explicit toy analog of the plan's actual *method* (Stage 2a's paired
IsoFLOP sweep, Stage 2b's scaling-law fit and the three decision-gate
ratios), scaled down and documented as such:

| Axis | Plan spec | This run | Ratio |
|---|---|---|---:|
| Model sizes | 5-7, from 50M-600M params | 3, from ~114K-5.1M params (backbone) | ~0.002-0.008x |
| Data budgets per size | 3-4, up to 128B tokens | 3, expressed as epoch counts (5/10/20) over a fixed ~1M-character corpus, not fresh tokens | qualitatively different, not just smaller |
| Compute budget | 10-30 PF-days | ~5.5 minutes, one RTX 5060 Ti | many orders of magnitude smaller |
| Core objective | Latent-primary JEPA (or a functoriality/STP-augmented variant) | AR+JEPA-auxiliary (token-primary, latent-auxiliary) | deliberately different, not a shortfall — see below |

**Why AR+JEPA-auxiliary and not latent-primary JEPA:** the Phase 1.5 pilot
found no surviving compositional-generalization win for pure JEPA after
the COGS-replication and held-out-layer-selection checks
(`PHASE1_5_COGS_REPLICATION_RESULTS.md`) — AR wins or ties on every
fairly-evaluated compositional task attempted. AR+JEPA-auxiliary was the
one latent-touching arm with a genuine, non-cherry-picked positive result
(robust, near-identical accuracy to AR across every layer on two
COGS-style tasks). Testing the plan's scaling-law claim at this stage
means testing the objective the toy evidence actually supports
investigating further, not resurrecting the framing the evidence argued
against.

**A further, explicit deviation from the plan's "data budget" concept:**
the plan's data budgets are fresh tokens (2B, 8B, 32B, 128B — each budget
sees new data). This run's "data budgets" are epoch counts over the same
~1M-character corpus (4,000 documents from `build_pretrain_corpus`) — more
*passes* over the same small corpus, not more *data*. This changes what
the fitted `beta` (data-scaling exponent) actually measures: it reflects
how loss improves with more optimization steps over a fixed, small,
repeatable corpus, closer to a training-curve exponent than a genuine
data-scaling exponent in the sense the plan's Chinchilla-style formulation
intends. Held-out generalization loss (see below) is still measured on a
corpus disjoint from training, so it isn't pure memorization — but it
is not the same experiment as scaling to more unique data.

## Method

`lattice/phases/phase2_toy.py` / `scripts/run_phase2_toy.py`:

- **3 model sizes** (`SIZES`): small (dim=64, 2 layers), medium (dim=128,
  4 layers), large (dim=256, 6 layers) — same `TinyTransformer` backbone
  architecture used throughout Phase 1.5, at three widths/depths.
- **3 data budgets** (`DATA_BUDGET_EPOCHS`): 5, 10, 20 epochs over a fixed
  4,000-document training corpus (1,000 each of code/JSON/BANKING77/triage
  text, same sources as Phase 1.5).
- **A genuinely held-out validation corpus**, built by
  `build_validation_corpus` — this required a fix during development:
  `build_pretrain_corpus` always reads BANKING77/triage text starting from
  row 0 regardless of seed, so a different seed alone does **not** produce
  disjoint real-text documents. `build_validation_corpus` explicitly
  offsets past the training corpus's real-text rows and uses a disjoint
  seed for the synthetic code/JSON generators, verified by
  `tests/test_phase2_toy.py::test_validation_corpus_disjoint_from_training_real_text`
  (asserts zero overlap between the two document sets, not just "should be
  fine").
- For each of the 9 (size, budget) cells, trains one AR baseline and one
  AR+JEPA-auxiliary model from scratch (matched architecture, optimizer,
  batch size), then measures both on the held-out validation corpus's
  next-token cross-entropy loss.
- Fits `L(N, D) = E + A/N^alpha + B/D^beta` (plan's Stage 2b candidate
  model) to each arm's 9 points via nonlinear least squares
  (`scipy.optimize.curve_fit`), extracting the plan's three decision-gate
  ratios: `beta_ratio` (data-efficiency), `alpha_ratio`
  (parameter-scaling), `E_ratio` (irreducible loss).

## Result

| Size | Epochs | Tokens seen | AR val CE | AR+JEPA-aux val CE | AR params | AR+JEPA-aux params |
|---|---:|---:|---:|---:|---:|---:|
| small | 5 | 2.56M | 2.623 | 2.596 | 114,048 | 130,624 |
| small | 10 | 5.12M | 2.042 | 1.958 | 114,048 | 130,624 |
| small | 20 | 10.24M | 1.536 | 1.554 | 114,048 | 130,624 |
| medium | 5 | 2.56M | 1.891 | 1.914 | 821,248 | 887,168 |
| medium | 10 | 5.12M | 1.423 | 1.443 | 821,248 | 887,168 |
| medium | 20 | 10.24M | 1.006 | 1.045 | 821,248 | 887,168 |
| large | 5 | 2.56M | 1.555 | 1.543 | 4,794,880 | 5,057,792 |
| large | 10 | 5.12M | 1.157 | 1.153 | 4,794,880 | 5,057,792 |
| large | 20 | 10.24M | 0.664 | 0.629 | 4,794,880 | 5,057,792 |

Per-cell wins split roughly evenly (AR+JEPA-aux ahead in 5 of 9 cells, AR
ahead in 4 of 9), with no consistent pattern by size or data budget —
consistent with single-seed noise rather than a systematic edge either way
at the level of individual cells.

**Scaling-law fits** (both R² > 0.98, a good fit to the 9 points each):

| Arm | E (irreducible loss) | A | alpha | B | beta |
|---|---:|---:|---:|---:|---:|
| AR | not identified (~0) | 190.4 | 0.440 | 75,223 | 0.737 |
| AR+JEPA-aux | not identified (~0) | 170.9 | 0.427 | 65,429 | 0.728 |

**Decision-gate ratios (AR+JEPA-aux / AR):**

| Metric | Plan's threshold for a "win" | This run | Verdict |
|---|---|---:|:--:|
| beta ratio (data efficiency) | < 0.5 | **0.989** | fail |
| alpha ratio (parameter scaling) | > 0.8 | **0.970** | pass (barely; effectively 1.0) |
| E ratio (irreducible loss) | ≤ 1.0 | not identified | n/a |

## Reading

- **The headline finding is a near-total tie in scaling behavior, not a
  win or loss for either arm.** Both `alpha` (~0.43-0.44) and `beta`
  (~0.73-0.74) are within 3% of each other between AR and AR+JEPA-aux.
  The plan's `alpha_ratio > 0.8` threshold is technically cleared (0.970),
  but that's because the two exponents are nearly equal, not because
  AR+JEPA-aux scales meaningfully better with parameters — reading "pass"
  off this ratio without the context that it's ~1.0 either way would be
  misleading. The `beta_ratio` (0.989) misses the plan's <0.5 threshold by
  a wide margin: there is no data-efficiency advantage from the auxiliary
  JEPA loss at this scale, on this task.
- **E (irreducible loss) is not identifiable from this sweep.** Both fits
  drove E to numerically-zero (AR: 4e-33; AR+JEPA-aux: 7e-6) — an
  artifact of too few (N, D) points over too narrow a range for the
  asymptotic/irreducible-loss term to be distinguishable from zero, not a
  real estimate of either arm's floor loss. Reporting a ratio of these two
  near-zero numbers would produce a nonsensical figure (an earlier version
  of this run did exactly that, printing a ratio of ~8e27, before this was
  caught and the reporting code was fixed to detect and flag this rather
  than print a meaningless number).
- **This is consistent with, and now quantifies, the "close to a tie"
  characterization of AR+JEPA-aux from `PHASE1_5_RESULTS.md`.** The
  Phase 1.5 pilot found AR+JEPA-aux matched or slightly beat AR on some
  probes and slightly trailed on others, at fixed scale. This sweep shows
  that relationship holding *across* scale too, at least over the narrow
  range tested here — neither arm is pulling ahead as size or (repeated)
  data increases.
- **Do not read this as closing the door on the plan's central bet.** The
  scope reduction is severe (params ~1/100,000th of the plan's smallest
  size; "data budget" here means repeated passes over ~1M characters, not
  128B fresh tokens), and a genuine scaling-law claim needs a much wider
  range of both axes than 3 sizes x 3 budgets at one seed can support —
  the R² values are good fits to *these* 9 points, not evidence the
  extracted exponents would hold at any other scale.

## What this does and doesn't tell us

- **Validated:** the toy-scale IsoFLOP sweep and scaling-law-fit machinery
  (paired training loop, held-out validation corpus construction, nonlinear
  least-squares fit to the plan's actual candidate model, ratio
  extraction) runs correctly and is architected to scale up directly if
  more compute becomes available — this is genuine progress on the
  plan's Stage 2a/2b *method*, independent of what this toy run's numbers
  show.
- **Not validated:** that AR+JEPA-auxiliary (or any latent-touching
  objective tested in this project) has a real data- or parameter-scaling
  advantage over plain AR. At this scale, it doesn't show one.
- **Directionally, this adds a fourth line of evidence (after the shallow
  task, the fairly-evaluated COGS-style tasks, and the fine-tuned
  indirection probe) that the design's central "latent objective beats or
  meaningfully augments token-level training" bet has not been supported
  by any toy-scale experiment run in this project so far**, including the
  one framed most favorably to it (AR+JEPA-auxiliary, chosen specifically
  because it was the strongest surviving candidate).

## Follow-ups

1. **Widen the range on both axes** — more model sizes (even within the
   same toy-scale budget, e.g. 5 sizes instead of 3) and, ideally, more
   *unique* data rather than more epochs over the same corpus — before
   trusting the fitted exponents as more than illustrative.
2. **More seeds per cell** — this is one seed per (size, budget) cell;
   the per-cell win/loss pattern (5 of 9 for AR+JEPA-aux) looks like noise
   but that's an assumption, not a tested claim.
3. Given four toy-scale experiments now point the same direction (no
   latent-primary or latent-auxiliary advantage over plain AR at this
   scale), the decision the plan itself calls for is now squarely on the
   table: treat the Core as a research branch that hasn't yet cleared its
   own bar, and consider the plan's stated fallback (revert to
   token-level prediction, reframe the validated decision-and-constraint
   stack as a governance/calibration layer around an LLM) as the
   evidence-supported default rather than a last resort — unless a wider
   sweep (per follow-up 1) changes this picture.
