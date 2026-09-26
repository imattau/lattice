# Phase 1.5 Results: Toy-Scale JEPA vs. AR vs. Contrastive Falsification

Status: **toy-scale pilot complete, now with a third (contrastive) arm.
Directional result: AR beats both latent-representation objectives on
2-3 of 3 probes, and contrastive is worse than JEPA on all 3 — the
"is JEPA specifically the problem, or is latent representation learning
the problem at this scale" question has a fairly clean answer: the latter.
Scope is far below the plan's spec (see below) — treat this as a pilot
that validates the training/eval pipeline and produces a caution signal,
not as the actual Phase 1.5 decision-gate run.**

## Objective (PLAN.md "Phase 1.5: Toy-Scale Falsification")

Test the central premise of Phase 2 at the smallest possible scale: does a
JEPA-trained language model develop representations comparable to an
autoregressive model of the same size and compute? No JEPA language model
has been shown to develop general representations at any scale — this is a
premise, not a finding, and the plan calls for testing it cheaply before
any large-scale training.

**Extension beyond the plan's text:** after the first two-arm run showed
JEPA losing to AR on 2 of 3 probes, a third arm — a contrastive objective
(InfoNCE on augmented-view positive pairs) at the same architecture, data,
and training budget — was added to distinguish two hypotheses the two-arm
comparison alone cannot separate:
- If contrastive **beats** AR where JEPA lost: the problem is JEPA's
  specific objective, not "latent representation learning" as a category.
  The Core should consider a contrastive objective instead.
- If contrastive **also loses**: the gap is more likely about token-level
  vs. latent-level training for language compositionality at this scale,
  independent of which latent objective is used — a broader and more
  concerning signal for the plan's central bet.

## Honest scope reduction

The plan specifies 10M-50M-parameter models, 100M-1B training tokens, and a
1-3 PF-day compute budget. None of that is executable in a single
interactive session on one consumer GPU (RTX 5060 Ti, 16GB) — 1 PF-day
alone is roughly 500-1000x more compute than this machine can produce in an
hour. Rather than silently run a smaller experiment and describe it as if
it satisfied the plan (same principle as the Phase 1 corpus-generation
deviation and the Phase 1 calibration bug), the reduction is stated
directly:

| Axis | Plan spec | This run | Ratio |
|---|---|---|---:|
| Trained parameters | 10M-50M | 4.8M (AR) / 5.1M (JEPA, context+predictor) / 5.1M (contrastive, encoder+projector) | ~0.1-0.5x |
| Training tokens | 100M-1B | ~3.07M characters | ~0.003-0.03x |
| Compute budget | 1-3 PF-days | ~25 min wall-clock, one RTX 5060 Ti, three arms | several orders of magnitude smaller |
| Data sources | The Stack (code), C4 (NL), JSON/tables | synthetic code/JSON templates + real BANKING77/triage text (all offline, no download) | narrower, more repetitive |
| Downstream probes | SST-2, MRPC, MNLI, CoLA, STS-B | BANKING77 intent (77-way), Phase 1 triage department (10-way), synthetic color/shape composition | different tasks — no internet dataset access assumed |
| Masking/loss-weight grid | pure JEPA vs JEPA+NextLat+MLM | pure JEPA only | grid not run |
| Contrastive batch size (negatives) | not specified; SimCLR-style methods typically use large batches (thousands) for enough negatives | 64 (126 negatives/anchor) | likely undersized for InfoNCE specifically — see caveat below |
| Cross-modal transfer | text -> code and back | not run | open |
| Loss-scaling curve | clean power law across data budgets | single run, no data-budget sweep | not run |

**This is a toy-of-the-toy.** It does not by itself satisfy or fail the
plan's Phase 1.5 decision gate — that requires the real-scale run. What it
does provide: (a) working, tested JEPA and contrastive training pipelines
at matched architecture, ready to scale up, and (b) a genuine, if small,
three-way empirical data point that should inform — not be overridden by —
optimism about scaling up either latent objective.

## Architecture and training

`lattice/tiny_transformer.py`: a single `TinyTransformer` backbone (6
layers, 256-dim, 8 heads, 192-token context) used identically by all three
arms — same depth/width/heads/context length, so any measured difference
is attributable to the objective, not capacity.

- **AR baseline** (`ARLanguageModel`): the backbone with a causal mask and
  a tied LM head, trained with next-character cross-entropy.
- **JEPA Core** (`phase1_5_toy.JepaModel`): a context encoder (same
  backbone, bidirectional) + an EMA target encoder (exact architectural
  copy, weights updated by EMA only, no gradient) + a 2-layer GELU
  predictor. Multi-block masking (one contiguous span per example, 15-20%
  of sequence length) replaces the masked span with a MASK token in the
  context encoder's input; loss is negative cosine similarity between the
  predictor's output at masked positions and the target encoder's
  (stop-gradient) output there. EMA schedule 0.996 -> 1.0, linear (I-JEPA
  schedule, per spec).
- **Contrastive arm** (`phase1_5_toy.ContrastiveModel`, new this pass): the
  same backbone (bidirectional, no target network) + a 2-layer GELU
  projector. Two augmented views per document — independent random
  contiguous crops (70-90% of sequence, rest set to PAD) plus 10%
  per-character token dropout within the kept span — are mean-pooled and
  projected, and trained with InfoNCE (NT-Xent) treating the other view of
  the same document as the positive and every other view in the batch as a
  negative. Downstream probes use the encoder's raw pooled output, not the
  projection (standard SimCLR practice — the projection head is discarded
  for representation quality evaluation).
- All three trained with AdamW, same LR (3e-4), same batch size (64), same
  epoch count (20) over the same corpus. JEPA and contrastive both do more
  compute per step than AR (JEPA: extra no-grad forward through the target
  encoder; contrastive: two forward passes, one per view), consistent with
  the plan's own compute-accounting table (latent objectives cost more per
  step than a matched AR baseline).
- Character-level tokenizer (`lattice.toy_corpus.CharTokenizer`, no
  external dependency): keeps nearly all trained capacity in the
  transformer body rather than an embedding table at this scale, at the
  cost of being unrealistic for a real LM.
- Corpus (`lattice.toy_corpus.build_pretrain_corpus`): 16,000 documents —
  4,000 each of synthetic code snippets, synthetic JSON records, real
  BANKING77 customer-service queries, and real Phase 1 triage document
  bodies.

## A bug found and fixed before reporting

The first run of the compositional-generalization probe reported **0.0%
accuracy for both models**. Assigning each (color, shape) pairing a single
joint class label meant the 4 held-out pairings were classes that *never
appeared during probe training at all* — a linear softmax head cannot
predict a class it never received a gradient signal for, regardless of
representation quality. That is a test-design bug, not a finding (same
"decompose before reporting" discipline as the Phase 1 calibration bug).
Fixed by probing color and shape as two independent 4-way attributes (each
value appears in several *other* training pairings, just never combined
with its held-out partner) and reporting joint accuracy (both attributes
correct). Covered by
`tests/test_phase1_5_toy.py::test_composition_accuracy_recovers_separable_attributes`,
which fails against the old joint-label formulation.

## Results (seed 0, 20 epochs, three arms — `PHASE1_5_RESULTS.json`)

| | AR | JEPA | Contrastive | Plan's "within 10%" gate (vs. AR) |
|---|---:|---:|---:|:--:|
| Final training loss | 0.403 (CE, nats) | -0.929 (cosine) | 0.172 (InfoNCE) | n/a — different loss scales |
| BANKING77 intent probe (77-way) | 0.077 | 0.067 | 0.025 | JEPA fail (87%*), contrastive fail (33%) |
| Triage department probe (10-way) | 1.000 | 0.947 | 0.848 | JEPA pass (95%), contrastive fail (85%) |
| Composition — color accuracy | 1.000 | 0.750 | 0.813 | both fail |
| Composition — shape accuracy | 1.000 | 0.938 | 0.438 | JEPA pass, contrastive fail |
| **Composition — joint accuracy** | **1.000** | **0.688** | **0.188** | **both fail, contrastive far worse** |

\* This second run's JEPA/AR ratio on BANKING77 (87%) differs from the
first two-arm run (68%) despite the same seed and config — the corpus
build and probe-split code are deterministic, but probe training itself
(`linear_probe_accuracy`) does not fix a seed for its own `torch.nn.Linear`
initialization, so probe-level noise of this size is expected on a 600-row
test set across 77 classes. Treat single-point BANKING77 percentages as
noisy; the composition-task and triage-department gaps are larger and more
stable across runs.

| Representation geometry | AR | JEPA | Contrastive |
|---|---:|---:|---:|
| Uniformity (Wang & Isola; more negative = more uniform) | -0.012 | -0.623 | **-3.035** |
| Effective rank (of 256 dims) | 96.0 | 92.1 | **165.2** |

Reading:

- **The hypothesis test resolves cleanly in one direction: contrastive is
  not a fix.** Contrastive loses to AR on every probe, and loses to JEPA
  on every probe too (worse BANKING77, worse triage department, worse
  composition joint accuracy — dramatically so: 0.188 vs 0.688). Per the
  framing this experiment was designed to test: this is evidence for "the
  problem is latent representation learning at this scale in general" over
  "the problem is JEPA's specific objective." JEPA is, on this evidence,
  the *better* of the two latent objectives, not the worse one.
- **This does not clear JEPA — it lowers the shine on the alternative.**
  AR still wins comprehensively. The result argues against a "swap JEPA
  for contrastive and move on" pivot, not for JEPA's viability. Both
  latent objectives underperform token-level training at this toy scale.
- **Important confound: the contrastive arm may be batch-size-starved.**
  InfoNCE's negative-sampling signal is well known to scale with batch
  size (SimCLR reports large accuracy gains going from small to
  large-batch training, because more negatives per anchor sharpen the
  contrastive signal). This run used batch size 64 (126 negatives/anchor)
  to keep the comparison matched to AR/JEPA's budget, which is far below
  typical contrastive-learning batch sizes (often thousands). **Contrastive
  losing this badly may be partly an artifact of an undersized batch, not
  purely evidence against contrastive objectives for language.** This
  should be re-tested with a larger batch (see follow-ups) before treating
  "contrastive also loses" as fully settled — the JEPA-vs-AR gap is on
  firmer ground since JEPA's objective doesn't have this specific
  batch-size dependency.
- **Representation geometry is not just uninformative here — it's
  anti-correlated with usefulness, and the contrastive arm makes this
  vivid.** Contrastive has by far the best uniformity (-3.035, an order of
  magnitude more negative than JEPA) and by far the highest effective rank
  (165 of 256, vs. ~92-96 for the other two) — exactly what InfoNCE is
  designed to optimize for (it directly rewards spreading negatives apart).
  It is also the *worst* representation on every actual downstream probe.
  If Phase 2's Latent Probing Suite weighted geometry alongside probe
  accuracy, contrastive would look like the strongest candidate by two of
  four metrics and the weakest by the ones that actually matter. This
  confirms and sharpens the two-arm run's finding: geometry metrics should
  not be used as a proxy for representation quality in this evaluation
  protocol without validating them against real probe accuracy first.
- **Do not read any of this as a decisive falsification of either latent
  objective.** The corpus is ~1000x smaller than spec, heavily templated,
  no masking/loss-weight grid was run for JEPA, and the contrastive arm's
  batch size is a known confound. A single-seed, single-configuration toy
  run per objective is a data point, not the Phase 1.5 decision-gate
  result.

## What this does and doesn't tell us

- **Validated:** both the JEPA and contrastive training pipelines (EMA
  target encoder / augmented-view InfoNCE) run, are stable, and are wired
  at matched architecture with the AR baseline, so a fair three-way
  comparison is possible. Phase 2 doesn't need to build this machinery
  from scratch for either objective.
- **Directionally supported, not proven:** the gap looks more like a
  general latent-vs-token-level property at this scale than a JEPA-specific
  defect, since contrastive fails at least as hard. This should raise, not
  lower, the bar for justifying the JEPA Core before committing Phase 2
  compute to it — but the contrastive arm's batch-size confound means this
  is not yet a clean result.
- **Not validated:** that either JEPA or contrastive develops
  representations comparable to AR at any scale, including this one. The
  plan's actual Phase 1.5 decision gate — real 10M-50M-parameter /
  100M-1B-token scale, with a masking/loss grid and the real probe suite —
  has not been executed.

## Follow-ups still open

1. **Re-run the contrastive arm with a larger batch size** (or a memory
   bank / momentum queue for more effective negatives) before treating
   "contrastive also loses" as settled — the current result may
   understate contrastive's ceiling due to too few negatives per anchor.
2. **Run at real Phase 1.5 scale** (10M-50M params, 100M-1B tokens,
   1-3 PF-days) if/when that compute becomes available — the pipeline
   (`lattice/tiny_transformer.py`, `lattice/phases/phase1_5_toy.py`) is
   architected to scale up directly (swap the corpus loader, bump
   `ToyConfig`), not rewritten.
3. **Masking-rate / loss-weight grid for JEPA** (pure JEPA vs
   JEPA+NextLat vs JEPA+NextLat+MLM) — not run in this pilot.
4. **Cross-modal transfer** (train on text, probe on code or vice versa)
   — not implemented in this pass.
5. **Loss-scaling curve** — this pilot is a single run at one data budget;
   the plan's "both models follow a clean power law" criterion needs a
   sweep across data budgets at fixed model size.
6. **Investigate the BANKING77 vs. triage-department probe gap** for
   JEPA specifically (close to AR on triage department, far behind on the
   harder 77-way BANKING77 task) — worth checking whether this is a
   fine-grained-label problem or a domain-familiarity effect.
7. Model checkpoints from this run are saved to `PHASE1_5_RESULTS.pt`
   (gitignored, like other `*.pt` artifacts) so follow-up diagnostics can
   reuse the trained models without a ~25-minute retrain.
