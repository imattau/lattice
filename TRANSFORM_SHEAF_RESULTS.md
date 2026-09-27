# Exploring Alternatives: Transformation Sheaf (No Trained Core)

Status: **built and ran exactly as specified. Result: negative, and it
replicates on the project's own real domain corpus with an even larger
gap than on the toy corpus supplied with the reference implementation.**
Per the reference implementation's own stated interpretive framework, this
is the "operations are missing what matters" outcome, not the "record
compresses structural information efficiently" outcome.

## What this tests

A fixed library of six linguistic operations (negate, passivize, embed,
quantify, cleft, question), applied exhaustively to every document. Each
operation reports APPLIES/BLOCKS/CONTRADICTS. A "sheaf" coherence check
verifies the document-level operation signature glues consistently from
its sentence-level signatures. A small set of derived features (apply/
block/contradict counts, coherence flag) plus the raw (operation, outcome)
incidence matrix — 23 features total — feed a plain multinomial logistic
regression. **No trained encoder, no learned representation anywhere
upstream of the classifier.** The hypothesis: this fixed, non-learned
structural record carries enough signal to compete with representation-
based approaches.

Built exactly per the provided reference implementation
(`transform_sheaf/`), including its own test suite
(`transform_sheaf/tests/`).

## A bug found in the reference implementation's own test suite

`test_negate_blocks_without_verb` used `"the refund"` as a no-verb fixture
and failed: `"refund"` is deliberately in the `VERBS` lexicon (needed to
detect billing-related verbs elsewhere), so this noun-phrase usage was
misclassified as containing a verb. This is a concrete instance of exactly
the limitation the module's own docstrings already flag ("heuristics...
documented so their limits are visible") — not a new defect, but worth
surfacing rather than silently patching. Fixed the test fixture (`"the
money"`, no lexicon collision) rather than the operation logic or lexicon,
since altering either would touch the mechanism under test. All 10 tests
pass after the fix.

## Result 1: the reference implementation's own toy corpus

`python -m transform_sheaf.run_triage` (4-class: billing/technical/
account/general, 320 documents, 36 templates):

| Feature set | # features | Accuracy |
|---|---:|---:|
| Transformation record | 23 | **0.469** |
| Bag of words | 102 | **1.000** |

- **Apply-count distribution**: `embed` and `cleft` applied to **224/224**
  training documents — literally every one, contributing zero
  discriminative information (a constant feature). `quantify` (211/224)
  and `passivize` (189/224) are also near-constant. Only `negate` (171/224)
  and `question` (81/224) show real variation.
- **Coherence**: **224/224** documents were sheaf-coherent — the
  coherence signal is completely uninformative on this corpus (constant,
  like the near-universal operations above).
- Per the reference implementation's own stated diagnostic: *"If the gap
  is small, the record compresses structural information efficiently. If
  it's large, the operations are missing what matters"* — the gap here is
  53 points. *"If most documents are coherent, the sheaf signal is
  uninformative on this corpus"* — 100% coherent. Both of the
  implementation's own negative-outcome conditions are met simultaneously.

## Result 2: the project's own real domain corpus (Phase 1 triage)

Same pipeline, same protocol, run against `data/triage/train.jsonl` /
`test.jsonl` (10-way department routing, 3,000 train / 1,000 test
documents, `transform_sheaf/run_lattice_triage.py`) — a more linguistically
diverse corpus than the 36-template toy set, and the actual task this
entire project has been validating against since Phase 1:

| Feature set | # features | Accuracy |
|---|---:|---:|
| Transformation record | 23 | **0.302** |
| Bag of words | 331 | **0.978** |
| BOW + record combined | 354 | **0.976** |
| Trained calibrated readout head (Phase 1, post-calibration-fix) | — | **0.997** |

- The gap is **larger on real data than on the toy corpus** (67.6 points
  vs. 53.1 points), not smaller — the opposite of what "the toy corpus was
  too easy for bag-of-words" would predict.
- Coherence is less degenerate here (1,143/3,000 = 38.1% coherent, vs.
  100% on the toy corpus) — the sheaf check does discriminate *something*
  on more diverse real text — but department accuracy from the record
  alone (0.302) is barely 3x the 10-way chance rate (0.10), nowhere near
  competitive.
- **Bag-of-words alone (0.978) nearly matches the fully trained, calibrated
  readout head (0.997)** on this exact task. This isn't a new finding
  specific to this exploration — it's consistent with the project's
  earliest result (`PHASE1_5_LAYER_PYRAMID_RESULTS.md`: AR aces the
  original shallow composition task from layer 1 alone) — department
  routing in this synthetic corpus is largely a lexical-detection problem
  (department names and topic words are strong direct cues), so a strong
  lexical baseline was always going to be hard to beat here regardless of
  what replaces it structurally.

## Reading

- **This is the negative outcome the reference implementation's own
  documentation explicitly anticipated and pre-authorized as informative**:
  *"If the record classifier underperforms bag-of-words, that does not
  kill the hypothesis. It means the specific operation library is
  insufficient... it would need a larger library, a proper grammar, and
  composition of operations to be tested fairly."* Taken at face value,
  that's the correct scoping — six hand-picked heuristic operations with
  no composition is a small hypothesis space, and this result doesn't
  refute richer versions of the same idea (a real grammar, operation
  composition, a larger library).
- **What it does rule out, on this evidence**: the *specific* six-operation,
  no-composition library tested here is not competitive with even a
  trivial lexical baseline on either a toy corpus or the project's real
  domain corpus, and the gap does not shrink on the more realistic
  corpus — it grows. Anyone extending this hypothesis should not expect
  "more realistic text" alone to close the gap; the operations themselves
  need to carry more signal (composition, a real grammar, or a larger
  library, per the reference implementation's own honest list of what
  isn't in it).
- **Consistent with this project's broader pattern**: representation
  quality has mattered least on tasks that are substantially lexical
  (department routing, the original shallow composition task) and most on
  tasks requiring genuine structural combination (the deep indirection
  task, the COGS-style tasks). A structural-record approach with no
  lexical features at all is, unsurprisingly, at its weakest exactly where
  lexical cues already do most of the work — the interesting test for this
  hypothesis would be a task where lexical bag-of-words itself struggles
  (e.g. the COGS-style or indirection tasks), not one where it's already
  near-ceiling.

## Also tested: the weaker, more defensible framing (record adds to BOW, not instead of it)

The hypothesis doesn't require the record to *replace* lexical features —
only to add structural signal a lexical baseline can't already get.
Concatenating BOW and record features (`transform_sheaf/run_lattice_triage.py`)
and training one classifier on the combination: **0.976 accuracy — no
better than BOW alone (0.978), and marginally worse** (within
run-to-run noise for a 354-feature logistic fit, but directionally showing
no benefit, not even a small one). Even in this weaker framing, on this
task, the record features contribute nothing measurable on top of what
lexical features already capture.

## Follow-ups

1. **Test on a task where bag-of-words itself is weak**, not one where
   it's already at 0.976-1.000 — the COGS-style tasks or the indirection
   task would be a fairer test of whether structural operations add
   anything a lexical baseline can't already get, since here BOW's
   near-ceiling performance leaves little room for any alternative
   (record-alone or record-added-to-BOW) to look good by comparison.
2. Per the reference implementation's own list of what's missing: test
   with a larger operation library and/or operation composition before
   treating six uncomposed heuristic operations as a fair test of the
   general "process, not representation" hypothesis.
