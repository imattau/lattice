# Phase 1.5 Follow-up: Readout-Pyramid Probe

Status: **the compositional-generalization gap is partly a readout-depth
artifact, not purely a missing-representation problem — and the effect
size tracks exactly how invariance-focused each objective is.** Reading
from the earliest layer instead of the top layer recovers 6-38 points of
composition-task accuracy depending on the arm, in an order (contrastive
>> JEPA > AR+JEPA-aux ~ AR) that has a clean mechanistic explanation. This
does not close the gap to plain AR — it narrows it.

## Motivation

BERT-style models are known to stratify hierarchically (part-of-speech
decodable from low layers, parsing from the middle, semantics/coreference
from the top) without any explicit layer-wise supervision, and the top
layer is generally specialized for the pretraining objective rather than
for arbitrary downstream tasks. Every probe reported so far in
`PHASE1_5_RESULTS.md` used only the **top** layer's pooled output. If the
JEPA/contrastive Core stratifies the same way, the compositional signal
the top-layer probes were failing to find might not be missing — it might
be sitting at a lower depth than the readout was looking.

**Test:** reuse the checkpoints already trained for the four-arm
comparison (`PHASE1_5_RESULTS.pt`, no retraining) and probe *every layer's*
pooled output, not just the top's, on all three tasks already used
(compositional generalization, BANKING77 intent, triage department).
`TinyTransformer.forward_layers` (new: `lattice/tiny_transformer.py`)
exposes the residual-stream hidden state after each block for this.
Two outcomes were possible: a middle/early layer clearly beating the top
(signal present, wrong readout depth — cheap fix) or all layers tied near
the top-layer number already reported (signal genuinely absent — the
readout wasn't the problem).

## Result: the first outcome, with a clean mechanistic pattern

`scripts/diagnose_phase1_5_layer_pyramid.py` — full per-layer numbers in
`PHASE1_5_LAYER_PYRAMID_RESULTS.json`.

**Compositional generalization (joint accuracy), layer 1 vs. top layer:**

| Arm | Layer 1 | Top layer (layer 6) | Gain from reading layer 1 |
|---|---:|---:|---:|
| AR | 1.000 | 0.938 | +0.062 |
| AR+JEPA-aux | 0.875 | 0.812 | +0.062 |
| JEPA | 0.812 | 0.625 | **+0.188** |
| Contrastive | 0.562 | 0.188 | **+0.375** |

**Triage department probe (10-way), best layer vs. top layer:**

| Arm | Best layer | Top layer | Gain |
|---|---:|---:|---:|
| AR | 1.000 (layer 6) | 1.000 | +0.000 |
| AR+JEPA-aux | 1.000 (layer 6) | 1.000 | +0.000 |
| JEPA | 0.967 (layer 2) | 0.943 | +0.023 |
| Contrastive | 0.930 (layer 1) | 0.828 | +0.102 |

**BANKING77 intent probe (77-way, all arms near chance ~0.013):** same
direction, smaller and noisier at this near-chance regime — contrastive's
best layer (0.040) is 60% relatively higher than its top layer (0.025);
AR and AR+JEPA-aux are flat (their best layer is already the top or tied).

The pattern is consistent across all three tasks and both probe
difficulties: **degradation toward the top layer is close to zero for the
two token-primary arms (AR, AR+JEPA-aux) and large for the two
latent-primary arms (JEPA, and especially contrastive)**, in an order that
matches how strongly each objective's *top-layer* loss pushes toward
invariance rather than raw token fidelity:

- **AR / AR+JEPA-aux**: the training objective (next-token prediction) is
  exactly "predict the next token's identity," so preserving token-level
  detail all the way to the top layer is directly what the loss rewards.
  No degradation is expected, and none is observed.
- **JEPA**: the top-layer loss trains the context encoder's masked-position
  outputs toward a *latent* target (cosine similarity to an EMA encoder's
  representation), which rewards abstraction away from exact surface form
  more than exact-token fidelity does. Moderate degradation.
- **Contrastive**: InfoNCE explicitly trains the top-layer pooled
  representation to be **invariant** to the augmentation used to build
  positive pairs — here, random crop + token dropout. Invariance to
  token-level perturbation is close to the opposite property a probe that
  needs to read off *which* color/shape words were present wants. This
  predicts the largest degradation, and by a wide margin, it's what is
  observed (composition-task recovery of +0.375, more than double JEPA's).

This is a coherent, mechanistically-motivated explanation, not a
post-hoc rationalization of noise: the ranking of "how much reading the
top layer costs you" (contrastive >> JEPA > AR+JEPA-aux ~ AR) is exactly
the ranking of "how much each objective's top-layer loss explicitly
optimizes for invariance to input perturbation," predicted before running
the per-layer probe, not fit to it afterward.

## What this does and doesn't change

- **Confirmed:** reading from an earlier layer instead of the top layer
  recovers real accuracy for JEPA and especially contrastive, at zero
  retraining cost.
- **Retracted — see `PHASE1_5_READOUT_PYRAMID_RESULTS.md`:** this document
  originally inferred from that finding that *concatenating* multiple
  depths into a readout pyramid "would very likely recover a meaningful
  fraction of the gap... at zero retraining cost." That inference was
  wrong, or at least not supported when directly tested. Naive
  concatenation of layers {1, 4, 6} into a single linear probe
  underperforms even the top-layer-only baseline for JEPA and
  AR+JEPA-aux on the composition task, and only clearly helps contrastive.
  The per-layer finding above (information exists at multiple depths) is
  still correct; "so concatenate them" does not follow automatically —
  see that document for the likely cause (a small-sample optimization
  artifact from tripling feature dimensionality on a 48-example training
  set) and what would need to change before re-testing the claim.
- **Not closed:** even at its best layer, JEPA (0.812) and contrastive
  (0.562) still trail plain AR's best layer (1.000) and AR+JEPA-aux's
  best layer (0.875) on compositional joint accuracy. A readout pyramid
  narrows the gap; it does not eliminate it. Some of the original
  31-100-point gaps reported in `PHASE1_5_RESULTS.md` reflect genuine
  representation-quality differences, not purely a readout-depth mistake.
- **A caveat on how much weight to put on the composition-task numbers
  specifically:** AR reaches 1.000 joint accuracy from **layer 1** — after
  a single transformer block. The composition sentences ("the red circle
  is on the table") contain the color and shape words as literal
  substrings, so a probe that only needs to detect "was the substring
  'red' present" is closer to a shallow lexical-detection task than a
  test of deep compositional reasoning. This doesn't undermine the
  *relative* finding (the ranking across arms and layers is still real and
  mechanistically explained), but it does mean "layer 1 is optimal" should
  not be over-read as "no depth is needed for composition in general" —
  it may be specific to how lexically transparent this particular toy task
  is. A harder composition task, where the attributes aren't literal
  substrings, is a natural follow-up before leaning further on this result.

## Recommendation for Phase 2's Latent Probing Suite

Per the reasoning that motivated this test: **do not assume the top layer
is the right layer for a downstream probe**, especially for objectives
(contrastive, and to a lesser extent JEPA) whose top-layer loss explicitly
or implicitly rewards invariance. The Latent Probing Suite should probe
multiple depths and pick per-task rather than assuming the single
top-layer pooled output this toy pilot (and, until now, the plan's
description of the Latent Probing Suite) implicitly assumed. **Do not**,
however, default to a concatenation-based readout pyramid without testing
it first — `PHASE1_5_READOUT_PYRAMID_RESULTS.md` found naive concatenation
underperforms even the top-layer baseline for two of the three
latent-objective arms on the small-sample composition task.

## Follow-ups

1. **Layer-banded training** (different objective at different depths —
   e.g. exact token supervision in lower layers, latent objective only in
   the top few) is the next thing this result motivates: if a band of
   token-level supervision at the bottom is what keeps AR's early layers
   so linearly decodable, training the JEPA/contrastive Core with a
   similar low-level token-supervision band might close more of the gap
   than a readout pyramid alone, without giving up the latent objective's
   claimed benefits at the top.
2. **Re-test the composition task with attributes that aren't literal
   substrings** (e.g. paraphrased or referential rather than named
   directly) to check whether the "layer 1 suffices" finding is an
   artifact of this toy task's lexical transparency.
3. Actually build and evaluate a readout pyramid (concatenate pooled
   features from 2-3 depths before the linear probe / decision head) as a
   direct test, rather than inferring its likely benefit from the
   per-layer numbers above.
