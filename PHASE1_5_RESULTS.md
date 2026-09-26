# Phase 1.5 Results: Toy-Scale JEPA vs. AR vs. Contrastive vs. AR+JEPA-Aux

Status: **toy-scale pilot complete with four arms.** Latent-primary
objectives (JEPA, contrastive) lose to token-level AR, with contrastive
losing worse than JEPA — a fairly clean signal that the gap is about
latent-vs-token training in general, not JEPA specifically. The fourth
arm — AR with the JEPA objective folded in as an auxiliary loss
("token-primary, latent-auxiliary", the ordering published work like
LLM-JEPA actually uses) — lands **close to a tie with plain AR overall**:
matches or slightly beats it on two probes, but underperforms it on the
compositional-generalization probe specifically. Scope is far below the
plan's spec (see below) — treat all of this as a pilot that validates the
training/eval pipeline and produces caution signals, not as the actual
Phase 1.5 decision-gate run.

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

Contrastive also lost (see below), which pointed at a further question:
published latent-objective successes (LLM-JEPA and similar) are
**token-primary, latent-auxiliary** — the latent loss is added on top of
next-token prediction, not substituted for it, unlike this project's
original latent-primary framing. A fourth arm tests that reframe directly:
same architecture/data/budget, `L = L_AR + alpha * L_NextLat` instead of
pure JEPA's `L = L_JEPA` or pure AR's `L = L_AR` alone. Three outcomes were
possible: beats plain AR (the reframe is validated — adopt it), ties (the
auxiliary adds nothing detectable at this scale), or loses (latent
objectives are actively harmful in any configuration at toy scale). The
result, below, lands closest to a tie with a specific weak spot.

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
| Trained parameters | 10M-50M | 4.8M (AR) / 5.1M (JEPA, context+predictor) / 5.1M (contrastive, encoder+projector) / 5.1M (AR+JEPA-aux, backbone+predictor) | ~0.1-0.5x |
| Training tokens | 100M-1B | ~3.07M characters | ~0.003-0.03x |
| Compute budget | 1-3 PF-days | ~32 min wall-clock, one RTX 5060 Ti, four arms | several orders of magnitude smaller |
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
layers, 256-dim, 8 heads, 192-token context) used identically by all four
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
- **AR+JEPA-auxiliary arm** (`phase1_5_toy.ArJepaModel`, new this pass): a
  standard causal AR model (identical to the AR baseline) trained with
  next-token cross-entropy **plus** a "NextLat" auxiliary loss — an EMA
  target encoder (same causal backbone, EMA-updated, no gradient) provides
  a latent target at each position, and the causal hidden state at
  position *t* is trained (via the same 2-layer GELU predictor pattern) to
  predict the target encoder's latent at position *t+1*, with cosine loss.
  No masking is needed, unlike the bidirectional JEPA arm — causal
  structure already defines "next position" the way it defines "next
  token." `L = L_AR + alpha * L_NextLat`, alpha=0.5. This is the
  "token-primary, latent-auxiliary" ordering — the reverse of the JEPA
  arm's "latent-primary" loss — matching the plan's own
  `L = L_JEPA + alpha*L_NextLat + beta*L_MLM` structure but with AR as the
  dominant term instead of JEPA.
- All four trained with AdamW, same LR (3e-4), same batch size (64), same
  epoch count (20) over the same corpus. JEPA, contrastive, and AR+JEPA-aux
  all do more compute per step than plain AR (each does an extra no-grad
  forward through a target/second-view encoder), consistent with the
  plan's own compute-accounting table (latent objectives cost more per
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

## Results (seed 0, 20 epochs, four arms — `PHASE1_5_RESULTS.json`)

| | AR | JEPA | Contrastive | AR+JEPA-aux |
|---|---:|---:|---:|---:|
| Final training loss | 0.403 (CE) | -0.929 (cosine) | 0.176 (InfoNCE) | -0.202 (ce=0.266, latent=-0.937) |
| BANKING77 intent probe (77-way) | 0.075 | 0.058 | 0.027 | **0.080** |
| Triage department probe (10-way) | 1.000 | 0.952 | 0.823 | **1.000** |
| Composition — color accuracy | 1.000 | 0.750 | 0.625 | 1.000 |
| Composition — shape accuracy | 0.938 | 0.938 | 0.438 | 0.750 |
| **Composition — joint accuracy** | **0.938** | **0.688** | **0.188** | **0.750** |

Run-to-run probe noise is real here (BANKING77 and composition-task
percentages have moved by single-digit points across separate runs of the
*same* AR/JEPA/contrastive config, because `linear_probe_accuracy` doesn't
fix its own probe-initialization seed) — read exact figures as
approximate, but the *ordering* (AR ~ AR+JEPA-aux > JEPA > contrastive on
BANKING77 and triage; AR > AR+JEPA-aux > JEPA > contrastive on composition
joint accuracy) has been stable across the three separate runs behind this
document.

| Representation geometry | AR | JEPA | Contrastive | AR+JEPA-aux |
|---|---:|---:|---:|---:|
| Uniformity (more negative = more uniform) | -0.010 | -0.624 | **-2.996** | -0.004 |
| Effective rank (of 256 dims) | 95.7 | 92.1 | **165.1** | 98.9 |

Reading:

- **Contrastive loses to both AR and JEPA on every probe** (worse
  BANKING77, worse triage department, worse composition joint accuracy —
  0.188 vs. JEPA's 0.688). This points at "latent representation learning
  underperforms token-level training at this scale in general," not "JEPA
  specifically is broken" — JEPA is the better of the two pure-latent
  objectives, not the worse one. (Caveat: InfoNCE is known to need many
  negatives per anchor to perform well, and this run's batch size of 64
  is well below typical contrastive-learning setups — see follow-ups.)
- **The AR+JEPA-auxiliary arm lands close to a tie with plain AR, not a
  clean win.** It matches or marginally beats AR on BANKING77 (0.080 vs.
  0.075) and triage department (1.000 vs. 1.000), which supports "the
  auxiliary latent loss doesn't hurt, and may help a little, on tasks the
  pretraining corpus already resembles." But on the compositional
  generalization probe — the task this project cares most about, since it
  is the one that actually tests structural/compositional understanding
  rather than topic classification — it underperforms plain AR (0.750 vs.
  0.938 joint accuracy), closer to (though still clearly better than)
  pure JEPA's 0.688. Per the three outcomes this arm was designed to
  distinguish (beats / ties / loses), **this is closest to "ties," with a
  specific, reproducible weak spot on composition** — not the clean
  validation of the token-primary/latent-auxiliary reframe that would
  justify adopting it as settled, but also not the "actively harmful"
  outcome that would argue against ever trying it.
- **The auxiliary loss does not corrupt AR's representation geometry** —
  AR+JEPA-aux's uniformity (-0.004) and effective rank (98.9) sit right
  next to plain AR's (-0.010, 95.7), both far from JEPA's and contrastive's
  geometry. This is a useful sanity check: folding in the latent loss as
  an auxiliary term keeps the representation "AR-shaped" rather than
  pulling it toward JEPA's or contrastive's geometry, consistent with the
  auxiliary loss being a secondary signal rather than the dominant one.
- **Representation geometry is not just uninformative here — it's
  anti-correlated with usefulness, and the four-arm spread makes this
  sharper.** Contrastive has by far the best uniformity and effective rank
  (exactly what InfoNCE is built to optimize) while being the *worst*
  representation on every real probe; AR and AR+JEPA-aux have the *worst*
  uniformity (near-collapsed, close to 0) while being the two best
  performers. If Phase 2's Latent Probing Suite weighted geometry
  alongside probe accuracy, it would rank these four arms almost exactly
  backwards from their actual downstream usefulness. Geometry metrics
  should not be used as a stand-in for probe accuracy in this evaluation
  protocol without validating them against real probe numbers first.
- **Do not read any of this as a decisive falsification of any objective.**
  The corpus is ~1000x smaller than spec, heavily templated, no
  masking/loss-weight grid was run for JEPA or for the auxiliary weight
  alpha, and the contrastive arm's batch size is a known confound. Four
  single-seed, single-configuration toy runs are four data points, not
  the Phase 1.5 decision-gate result.

## What this does and doesn't tell us

- **Validated:** the JEPA, contrastive, and AR+JEPA-auxiliary training
  pipelines all run, are stable, and are wired at matched architecture
  with the AR baseline, so a fair four-way comparison is possible. Phase 2
  doesn't need to build any of this machinery from scratch.
- **Directionally supported, not proven:** pure latent-primary objectives
  (JEPA, contrastive) underperform token-level AR at this scale, and
  contrastive underperforms JEPA — evidence for a general latent-vs-token
  property rather than a JEPA-specific defect. This should raise, not
  lower, the bar for justifying a latent-primary JEPA Core before
  committing Phase 2 compute to it.
- **A specific, reproducible weak spot, not a clean resolution:** folding
  the JEPA objective in as an auxiliary loss on top of AR neither clearly
  helps nor clearly hurts on topic-style probes, but it measurably hurts
  compositional generalization relative to plain AR (0.750 vs. 0.938) in
  this run. This is not evidence to adopt the token-primary/latent-auxiliary
  reframe as a fix, and not evidence to abandon it either — it is a
  specific result (auxiliary latent loss costs something on the
  compositional task, at this alpha, at this scale) that would need a
  weight sweep and the real-scale probes to interpret with confidence.
- **Not validated:** that any of JEPA, contrastive, or AR+JEPA-auxiliary
  develops representations comparable to (or better than) plain AR at any
  scale, including this one. The plan's actual Phase 1.5 decision gate —
  real 10M-50M-parameter / 100M-1B-token scale, with a masking/loss grid
  and the real probe suite — has not been executed.

## Follow-up: is the gap a missing representation, or a readout-depth artifact?

**Done — see `PHASE1_5_LAYER_PYRAMID_RESULTS.md`.** All probes above only
ever read the top layer's pooled output. Probing every layer of the
already-trained checkpoints (no retraining) shows the compositional signal
is not fully absent for JEPA/contrastive — it's present at earlier layers
and degrades toward the top, in an order (contrastive >> JEPA > AR ~ 0)
that exactly matches how strongly each objective's top-layer loss rewards
invariance to input perturbation over token fidelity. Reading from layer 1
instead of the top layer recovers +0.375 composition-joint-accuracy for
contrastive and +0.188 for JEPA, but does not close the gap to AR's own
best layer — a readout pyramid would narrow this gap, not eliminate it.

## Follow-ups still open

1. **Sweep `ar_jepa_alpha`** (currently fixed at 0.5) before drawing any
   conclusion about whether the auxiliary loss helps, hurts, or is neutral
   — the composition-task cost could be alpha-specific (too much weight on
   the latent term) rather than inherent to the token-primary/
   latent-auxiliary structure itself.
2. **Re-run the contrastive arm with a larger batch size** (or a memory
   bank / momentum queue for more effective negatives) before treating
   "contrastive also loses" as settled — the current result may
   understate contrastive's ceiling due to too few negatives per anchor.
3. **Run at real Phase 1.5 scale** (10M-50M params, 100M-1B tokens,
   1-3 PF-days) if/when that compute becomes available — the pipeline
   (`lattice/tiny_transformer.py`, `lattice/phases/phase1_5_toy.py`) is
   architected to scale up directly (swap the corpus loader, bump
   `ToyConfig`), not rewritten.
4. **Masking-rate / loss-weight grid for JEPA** (pure JEPA vs
   JEPA+NextLat vs JEPA+NextLat+MLM) — not run in this pilot.
5. **Cross-modal transfer** (train on text, probe on code or vice versa)
   — not implemented in this pass.
6. **Loss-scaling curve** — this pilot is a single run at one data budget;
   the plan's "both models follow a clean power law" criterion needs a
   sweep across data budgets at fixed model size.
7. **Investigate the BANKING77 vs. triage-department probe gap** for
   JEPA specifically (close to AR on triage department, far behind on the
   harder 77-way BANKING77 task) — worth checking whether this is a
   fine-grained-label problem or a domain-familiarity effect.
8. Model checkpoints from this run are saved to `PHASE1_5_RESULTS.pt`
   (gitignored, like other `*.pt` artifacts) so follow-up diagnostics can
   reuse the trained models without a ~32-minute retrain.
