# Phase 1.5 Follow-up: Does a Readout Pyramid Actually Help?

Status: **no — not the naive version tested here. Concatenating pooled
features from layer 1, a middle layer, and the top layer, then training a
plain linear probe on the concatenation, does not reliably beat even the
top-layer-only baseline, and on the (small-sample) composition task it is
worse than top-layer-only for two of the three latent-objective arms. This
contradicts the working assumption from the previous follow-up
(`PHASE1_5_LAYER_PYRAMID_RESULTS.md`) that a readout pyramid would recover
most of the per-layer gains "for free."**

## Motivation (Test B from the encoder/decoder-split follow-up)

The readout-pyramid *diagnostic* (`PHASE1_5_LAYER_PYRAMID_RESULTS.md`)
showed that reading from layer 1 instead of the top layer recovers real
accuracy for JEPA and contrastive on several probes, and inferred that "a
readout pyramid... would very likely recover a meaningful fraction of the
compositional-generalization gap." That inference was never directly
tested — it compared single layers to each other, not a concatenated
multi-depth feature to a single layer. This follow-up runs the actual
comparison: for each arm, on the same three probes, in the same run (so
probe-initialization noise doesn't confound the comparison):
- **top-layer-only** — the baseline used throughout `PHASE1_5_RESULTS.md`
- **best single layer** — found by scanning all 6 layers (test-set-informed,
  see caveat below)
- **pyramid** — layers {1, 4, 6} (bottom, middle, top) pooled features
  concatenated, one linear probe over the concatenation

`scripts/diagnose_phase1_5_readout_pyramid.py`; full numbers in
`PHASE1_5_READOUT_PYRAMID_RESULTS.json`.

## Result

| Arm | Task | Top-only | Best single layer | Pyramid (1+4+6) |
|---|---|---:|---:|---:|
| AR | composition joint | 0.938 | 1.000 (layer 1) | 0.938 |
| JEPA | composition joint | 0.688 | 0.812 (layer 1) | **0.562** |
| Contrastive | composition joint | 0.188 | 0.500 (layer 1) | **0.312** |
| AR+JEPA-aux | composition joint | 0.812 | 0.875 (layer 1) | **0.688** |
| AR | triage department | 1.000 | 1.000 | 1.000 |
| JEPA | triage department | 0.938 | 0.962 (layer 2) | 0.962 |
| Contrastive | triage department | 0.830 | 0.932 (layer 1) | 0.930 |
| AR+JEPA-aux | triage department | 1.000 | 1.000 | 0.997 |
| AR | BANKING77 intent | 0.073 | 0.082 (layer 4) | 0.085 |
| JEPA | BANKING77 intent | 0.062 | 0.062 | 0.080 |
| Contrastive | BANKING77 intent | 0.030 | 0.038 (layer 1) | 0.030 |
| AR+JEPA-aux | BANKING77 intent | 0.082 | 0.082 | 0.083 |

Reading:

- **On the composition task, the pyramid is worse than top-layer-only for
  JEPA, contrastive-relative-to-best, and AR+JEPA-aux** — and worse than
  the best single layer for every arm without exception, sometimes
  sharply (JEPA 0.812 -> 0.562, AR+JEPA-aux 0.875 -> 0.688). This directly
  contradicts the earlier follow-up's inference. **Do not adopt a
  concatenation-based readout pyramid as a Phase 2 default on the strength
  of the per-layer diagnostic alone** — that diagnostic showed information
  exists at multiple depths, not that naively combining depths is a good
  way to use it.
- **On BANKING77 and triage (larger training sets, ~2,400 examples), the
  pyramid is roughly neutral** — small wins and losses, all within the
  noise band already established for these probes. The composition task's
  training set is much smaller (48 examples across 12 in-distribution
  pairings), and a 3x-wider input (768-dim instead of 256-dim) with the
  same fixed 300-epoch probe-training budget is a plausible recipe for the
  probe to fit *less* well, not more — more parameters, same data, same
  optimization budget. **The most likely explanation for the composition
  task's pyramid underperformance is a small-sample optimization artifact
  specific to that task, not evidence that multi-depth information is
  actively harmful to combine.** This should be re-tested with a probe
  regularization/epoch budget scaled to the pyramid's dimensionality, or
  with a larger composition-task training set, before concluding pyramids
  don't help at all.
- **A methodological asymmetry worth naming:** "best single layer" is
  chosen by evaluating all 6 layers *on the test set* and reporting the
  winner — this is a mild form of test-set-informed selection (like
  reporting the best of several models by test accuracy) that the single,
  fixed pyramid feature and the fixed top-layer feature don't get the
  benefit of. "Best single layer" is consequently an optimistic,
  non-deployable number (you can't know in advance which layer will win
  without already having test labels) — the fairer comparison is
  pyramid-vs-top-layer, not pyramid-vs-best-layer. Even on that fairer
  comparison, though, the pyramid still loses to top-layer-only for JEPA
  and AR+JEPA-aux on composition, and only clearly helps contrastive
  (0.188 -> 0.312, still far below AR's 0.938).
- **Contrastive is the one arm where the pyramid clearly helps** even
  against the fair top-layer baseline (composition 0.188 -> 0.312;
  triage 0.830 -> 0.930). This is consistent with the earlier finding that
  contrastive's top layer is the most invariance-stripped of the four
  (largest single-layer degradation in `PHASE1_5_LAYER_PYRAMID_RESULTS.md`)
  — it has the most to gain from any mechanism that reintroduces
  lower-layer detail, pyramid or otherwise.

## Test A (Talker read-point) was not run

The encoder/decoder-split follow-up's Test A — moving a generation
decoder's ("Talker's") read-point across depths and measuring generation
quality — requires a generation decoder. This toy Phase 1.5 harness never
built one: it trains encoders and evaluates them with linear probes only,
with no reconstruction/generation path at all. Testing Test A would mean
building a toy Talker first (a real, separate piece of work, not a
same-day extension of the existing checkpoints), so it wasn't attempted
here rather than being silently skipped or faked.

## What this changes

- **The "cheap architectural fix" framing from the split follow-up is not
  supported by this test.** A readout pyramid is not a free win at this
  toy scale with naive concatenation; it needs its own tuning (probe
  capacity/regularization matched to input dimensionality) before it can
  be evaluated fairly, and even then it may only help the arm whose top
  layer is most invariance-stripped (contrastive), not JEPA.
- **The per-layer diagnostic's core finding stands**: information the top
  layer doesn't expose exists at earlier depths, in an order that tracks
  each objective's invariance pressure. What doesn't follow automatically
  is that concatenating those depths is the right way to expose it — that
  needs its own validated design, not an assumption.
- **This does not change the Core question.** The deep (non-lexical)
  composition task remains the experiment that actually decides whether
  AR's advantage is real composition or a lexical-detection artifact
  (`PHASE1_5_DEEP_COMPOSITION_RESULTS.md`), and that experiment is still
  unresolved (floored for every arm, likely due to an out-of-distribution
  task-format confound, not settled either way).

## Follow-ups

1. Re-test the pyramid with probe capacity/regularization/training budget
   scaled to its input dimensionality (or with more composition-task
   training examples) before concluding naive concatenation doesn't help
   on small-sample tasks specifically.
2. If a Talker/decoder component is ever built for this toy harness, run
   Test A (generation read-point) properly rather than inferring it from
   the encoder-only pyramid result.
3. Try a learned combination (small MLP over the concatenated pyramid, or
   attention-weighted layer combination) instead of a single linear layer
   over raw concatenation, which may be more robust to the
   dimensionality/sample-size issue than a bigger linear input.
