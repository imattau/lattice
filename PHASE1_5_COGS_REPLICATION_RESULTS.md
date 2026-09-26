# Phase 1.5 Follow-up: COGS Replication + Held-Out Layer Selection

Status: **the "JEPA beats AR" headline from `PHASE1_5_COGS_STYLE_RESULTS.md`
does not survive fair layer selection, and does not replicate on a second
task. Retracted as a general finding — see the correction in that
document. What does hold up: AR+JEPA-auxiliary is the most robust
non-AR arm across both tasks and every layer, and contrastive fails
outright on the second task.**

## Motivation

The COGS-style result (JEPA's best layer beating AR's best layer, 0.922
vs. 0.859) came with two flagged caveats: the "best layer" was chosen by
scanning all 6 layers *on the task's own test set* (a selection-bias risk
already named but not tested), and it was a single task, single seed. Two
cheap follow-ups, both on the existing checkpoints, no retraining:

1. **Replication**: a second COGS-style task with different primitives
   from a different pretraining-corpus tier — JSON `name`/`status` values
   (`build_cogs_style_task_v2`), verified present in the synthetic-JSON
   slice of `build_pretrain_corpus`
   (`test_cogs_v2_task_primitives_are_present_in_the_actual_pretraining_corpus`),
   vs. the first task's triage-derived department/urgency values.
2. **Held-out layer selection**: pick each arm's layer using BANKING77
   intent-probe accuracy (a *different* task) instead of the COGS task's
   own test accuracy, then evaluate the COGS tasks at that
   independently-chosen layer. This removes the "the layer was picked
   using the same numbers being reported" objection.

`scripts/diagnose_phase1_5_cogs_replication.py` — full numbers in
`PHASE1_5_COGS_REPLICATION_RESULTS.json`.

## Result

| Arm | Task | Top layer | Own-best layer (selection-biased) | Held-out-selected layer (fair) |
|---|---|---:|---:|---:|
| AR | dept/urgency | 0.844 | 0.859 | 0.797 |
| JEPA | dept/urgency | 0.688 | 0.922 | **0.688** |
| Contrastive | dept/urgency | 0.625 | 0.625 | 0.438 |
| AR+JEPA-aux | dept/urgency | 0.828 | 0.828 | 0.828 |
| AR | json name/status | 1.000 | 1.000 | 1.000 |
| JEPA | json name/status | 0.500 | 1.000 | **0.500** |
| Contrastive | json name/status | 0.000 | 0.000 | 0.000 |
| AR+JEPA-aux | json name/status | 1.000 | 1.000 | 1.000 |

Reading:

- **Under fair, held-out layer selection, JEPA loses to AR on both
  tasks** — 0.688 vs. 0.797 on department/urgency, 0.500 vs. 1.000 on JSON
  name/status. The BANKING77-selected layer for JEPA turned out to be
  layer 6 on this run — its *worst* layer for the COGS-style tasks, not
  its best (layer 2). That is exactly the failure mode the "best own
  layer" framing was vulnerable to: the layer that scores best on one
  task is not the layer that scores best on another, so cherry-picking
  per-task isn't a fair way to compare representations, and a real
  deployed system has no oracle to do that picking for it.
- **This does not replicate.** The second task (JSON name/status) makes
  the point starkly: AR sits at a perfect 1.000 on *every single layer*,
  while JEPA swings from 1.000 (layer 2 only) down to 0.500 (layers 5-6)
  depending on depth. JEPA's own-best-layer success on the first task was
  real for that specific layer, but it is not a stable property of JEPA's
  representation that generalizes across tasks or survives a
  non-cherry-picked read.
- **AR+JEPA-auxiliary is the standout of this follow-up.** Its top layer,
  own-best layer, and held-out-selected layer are *identical* on both
  tasks (0.828 / 0.828 / 0.828 and 1.000 / 1.000 / 1.000) — it doesn't
  need cherry-picking at all, because every layer already performs
  equally well. This is a genuinely positive, robust result for the
  token-primary/latent-auxiliary reframe: it gets AR-comparable
  performance on both COGS-style tasks *and* AR-like depth-robustness,
  properties pure JEPA has neither of once layer selection is done fairly.
- **Contrastive fails outright on the second task** — flat 0.000 across
  every layer, its most severe failure in this entire project (previously
  its worst results were well above zero). Combined with every prior
  result, contrastive is now a five-for-five replicated negative finding:
  weakest or worst arm on every single task tested, at every scale of
  scrutiny applied.

## Correcting `PHASE1_5_COGS_STYLE_RESULTS.md`

That document's headline — "JEPA's best layer (0.922) exceeds AR's best
layer (0.859), the first result anywhere in Phase 1.5 where a
latent-primary objective beats AR" — does not survive this follow-up and
is retracted as a general claim. What remains true from that document:
department/urgency accuracy well above chance for all four arms (the
exposure diagnosis from `PHASE1_5_DEEP_COMPOSITION_RESULTS.md` still
holds — that part didn't depend on layer selection). What does not hold:
the specific claim that JEPA's representation is "more compositional than
AR's" in any layer-selection-independent sense. Under the fairer
protocol, **AR wins or ties on both COGS-style tasks**, and the earlier
result was an artifact of comparing each arm's best cherry-picked layer
rather than a comparable, deployable read.

## Where this leaves the design question

- **JEPA has no surviving positive result on any compositional task in
  this pilot.** The shallow task, the deep (indirection) task, and now
  the fairly-evaluated COGS-style tasks all favor AR or tie. The one
  moment it looked ahead (COGS-style, own-best-layer) was a selection
  artifact.
- **AR+JEPA-auxiliary is the one latent-touching arm with a clean,
  replicated, non-cherry-picked positive result**: AR-comparable accuracy
  on both COGS-style tasks, at every layer, no selection needed. This is
  the strongest evidence yet in favor of the token-primary/latent-auxiliary
  reframe specifically (not latent-primary JEPA).
- **Contrastive remains, now more emphatically, the weakest arm on every
  single test run in Phase 1.5** without exception.

## Follow-ups

1. Given JEPA has no surviving compositional-generalization win, the
   fine-tuned-probe experiment (from the earlier split-question review) is
   now more interesting to run on AR+JEPA-auxiliary specifically than on
   pure JEPA — it's the arm with an actual positive signal worth
   understanding further.
2. The harder indirection task (genuine "no single token predicts the
   answer," primitives in corpus) remains the load-bearing unresolved
   experiment for the design's central bet. Nothing in this follow-up
   changes that.
3. Report "held-out-selected layer" accuracy, not "own-best-layer"
   accuracy, in any future toy-scale comparison — this follow-up shows the
   difference between the two is not a minor technicality.
