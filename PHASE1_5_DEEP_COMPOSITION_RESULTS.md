# Phase 1.5 Follow-up: Deep (Non-Lexical) Compositional Generalization

Status: **inconclusive by construction — all four arms floor at or below
chance, including AR. This does not resolve whether the shallow-task
compositional gap is real; it surfaces a different, more basic confound
(likely out-of-distribution task format) that has to be fixed before this
question can be answered.**

## Motivation

The readout-pyramid follow-up (`PHASE1_5_LAYER_PYRAMID_RESULTS.md`) flagged
a real caveat: AR reaches 1.000 joint accuracy on the original
compositional-generalization task from **layer 1** — after a single
transformer block — because the color and shape words are literal
substrings of every input ("the red circle is on the table"). That task
can be solved by detecting which color/shape words are present; it doesn't
require combining information across positions, so it may be measuring
lexical detection rather than composition. The follow-up review's
recommendation: build a task where **no single token predicts the
answer**, and re-run all four arms at every layer. If AR's advantage
survives on a genuinely non-lexical task, the design's central bet is
weakened for real; if AR also fails, the earlier gap was a task artifact.

## The deep task: two-hop variable binding

`build_deep_composition_task` (`lattice/phases/phase1_5_toy.py`): every
example is a link->value mapping table plus two variable assignments, e.g.

```
x=p y=m p=red q=blue r=green s=yellow m=circle n=square o=triangle w=star
```

Every one of the 4 colors and 4 shapes appears as a substring in **every**
example, regardless of the answer — "is `red` present" is true 100% of the
time, train and test alike, so a bag-of-words / single-position read
cannot shortcut this the way it could the original task. The correct
answer (what color does `x` resolve to; what shape does `y` resolve to)
requires finding `x`'s link token, then finding which value that link maps
to *in this specific example* (the link->value permutation is randomized
per example). Same held-out (color, shape) pairings as the original task,
so the generalization structure is otherwise comparable.
`tests/test_phase1_5_toy.py::test_deep_composition_task_removes_lexical_shortcut`
verifies every color and shape word is present in every generated example.

## Result: floor performance for every arm, at every layer

`scripts/diagnose_phase1_5_deep_composition.py` — reuses the four
checkpoints already trained (`PHASE1_5_RESULTS.pt`, no retraining), probes
every layer, 192 train / 64 test examples. Chance level: 0.25 for each
4-way attribute, 0.0625 for joint (both attributes) accuracy.

| Arm | Joint accuracy range across layers | Top-layer joint | Color accuracy range | Shape accuracy range |
|---|---:|---:|---:|---:|
| AR | 0.000 - 0.047 | 0.031 | 0.281 - 0.375 | 0.234 - 0.328 |
| JEPA | 0.016 - 0.047 | 0.047 | 0.234 - 0.312 | 0.234 - 0.375 |
| Contrastive | 0.016 - 0.047 | 0.016 | 0.188 - 0.266 | 0.266 - 0.328 |
| AR+JEPA-aux | 0.016 - 0.062 | 0.031 | 0.391 - 0.500 | 0.188 - 0.234 |

**Every arm's joint accuracy sits at or below the 0.0625 chance level, at
every layer, with no readout-depth effect worth reporting** (deltas from
top-to-best layer are 0.000-0.031, indistinguishable from noise given
n=64 test examples). This includes AR, which had a perfect 1.000 on the
shallow task. One partial exception worth naming rather than burying:
AR+JEPA-aux's *color* accuracy alone (not joint) reaches 0.39-0.50, mildly
above the 0.25 chance level for that single attribute — a faint, isolated
signal that doesn't move the joint metric (its shape accuracy stays at or
below chance) and shouldn't be over-read from one run.

## This does not answer the question it was built to answer — and here's why

A clean result would have been either "AR keeps winning" (the shallow-task
gap is real) or "AR also drops to JEPA's level" (the shallow-task gap was
a lexical artifact). Instead, **AR drops all the way to chance alongside
everything else**, which is a different outcome from both predictions and
needs its own explanation before the original question can be
re-approached.

**Most likely explanation: the deep task's text format is far more
out-of-distribution for every arm's pretraining corpus than the shallow
task's was.** The pretraining corpus
(`lattice.toy_corpus.build_pretrain_corpus`) is synthetic code, synthetic
JSON, and real natural-language business text (BANKING77, Phase 1 triage
documents) — none of it contains anything resembling a dense chain of
`token=token` assignments (`x=p y=m p=red q=blue ...`). By contrast, the
shallow task's sentences ("the red circle is on the table") are ordinary,
grammatical English, structurally close to the natural-language portion of
the pretraining mix, even though the exact sentences were never seen
during pretraining. AR's perfect shallow-task score shows its next-token
training generalizes readily to novel but *in-register* text; the deep
task's dense symbolic format may simply never have produced any linearly
decodable feature in *any* frozen encoder, regardless of objective,
because none of them ever built representations for this kind of string.
A frozen linear probe can only recover what's already linearly present in
frozen features — if no arm's pretraining ever produced features relevant
to `x=p`-style syntax, a uniform floor across every arm and every layer is
exactly what a distribution-mismatch failure looks like, and it looks
identical to "none of these objectives can do 2-hop reasoning" from the
outside. **This experiment cannot distinguish those two explanations as
constructed**, which is the actual, honest conclusion — not "composition
fails for everyone" and not "the shallow-task gap is refuted."

## What would actually resolve this

1. **Reformat the deep task in natural-language register** — e.g. "the
   item labeled p is red. the item labeled q is blue. ... x refers to
   item p. what color is x?" — still non-lexically-shortcuttable (every
   color/shape word still appears in every example), but grammatically
   ordinary text closer to what the pretraining corpus actually contains,
   isolating "can any of these objectives do indirection" from "was this
   exact syntax ever seen."
2. **Give the deep-task format some exposure during pretraining** (mix a
   modest fraction of link-style assignment text into
   `build_pretrain_corpus`) before evaluating with a frozen probe, so a
   floor result can't be blamed on zero exposure.
3. **Fine-tune rather than frozen-probe** on the deep task, to separate
   "does the frozen representation already linearly contain the answer"
   from "can this architecture learn the relationship given more direct
   training signal" — the frozen-probe design used throughout Phase 1.5
   is deliberately conservative (spec's "linear probe on a frozen Core" is
   exactly the intended evaluation protocol) but it is the more likely
   source of this floor result than any property of the objectives
   themselves.
4. Until one of the above is run, **the shallow-task compositional gap
   (AR 0.938-1.000 vs. JEPA 0.625-0.812 vs. contrastive 0.188-0.562,
   narrowed but not closed by the readout-pyramid follow-up) remains the
   best available toy-scale evidence** — weakened by the lexical-shortcut
   caveat, but not superseded by this null result, since the null result
   has its own unresolved confound.

## Bottom line for the design question

This does not move the JEPA-vs-AR question in either direction. It
surfaces a methodological limitation (out-of-distribution task format for
a frozen-probe evaluation) that needs fixing before a genuinely deep
compositional task can produce an interpretable answer at this toy scale.
The recommended next step is (1) above — reformat in natural-language
register — since it is the cheapest way to remove the confound without
retraining any of the four arms.
