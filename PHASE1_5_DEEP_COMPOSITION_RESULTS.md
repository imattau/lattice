# Phase 1.5 Follow-up: Deep (Non-Lexical) Compositional Generalization

Status: **inconclusive, and now more specifically so.** All four arms
floor at or below chance in both the original symbolic phrasing and a
natural-language reformulation of the identical task (see "Re-run in
natural-language register" below) — including AR, which dominated the
shallow lexical task. Reformatting to ordinary English ruled out "the
symbolic assignment syntax confused every model" as the explanation
without changing the result, which narrows the likely cause to zero
pretraining exposure to any two-hop binding structure at all (in either
register) combined with a frozen-linear-probe evaluation, rather than a
surface-level text-format problem. This still does not resolve whether the
shallow-task compositional gap is real.

**Update — see `PHASE1_5_COGS_STYLE_RESULTS.md` and
`PHASE1_5_COGS_REPLICATION_RESULTS.md`:** a follow-up task using
primitives verified present in the actual pretraining corpus (only the
*pairing* held out, COGS-style) confirms the exposure diagnosis directly:
all four arms clear chance by a wide margin on that task. (An initial
reading of that result — JEPA's best layer beating AR's — did not survive
a fairness check and was retracted; under held-out layer selection and on
a replication task, AR wins or ties instead.) The floor result on *this*
document's task reflects zero exposure to the indirection *structure*, not
a general inability of any objective to generalize compositionally at
this scale — that diagnosis stands independent of the retraction above.
The stronger "no single token predicts the answer" property this document
targets remains untested, not refuted.

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

## Re-run in natural-language register

`build_deep_composition_task(..., style='natural')` keeps the identical
two-hop binding structure and anti-shortcut property (every color/shape
word still present in every example) but replaces the dense assignment
syntax with ordinary declarative English:

```
p is red. q is blue. r is green. s is yellow. m is circle. n is square.
o is triangle. w is star. x is p. y is m.
```

`scripts/diagnose_phase1_5_deep_composition.py --style natural` — same
four checkpoints, no retraining, 192 train / 64 test examples (chance:
0.25 per attribute, 0.0625 joint).

| Arm | Joint accuracy range | Color accuracy range | Shape accuracy range |
|---|---:|---:|---:|
| AR | 0.016 - 0.078 | 0.234 - 0.312 | 0.312 - 0.438 |
| JEPA | 0.000 - 0.047 | 0.250 - 0.297 | 0.281 - 0.391 |
| Contrastive | 0.016 - 0.062 | 0.188 - 0.281 | 0.297 - 0.438 |
| AR+JEPA-aux | 0.031 - 0.078 | 0.219 - 0.297 | 0.266 - 0.344 |

**Still floor for every arm at every layer.** Joint accuracy remains at or
near the 0.0625 chance level throughout, with no readout-depth effect
(all top-to-best deltas are +0.03 to +0.05, indistinguishable from noise
at n=64). One consistent, minor pattern worth naming: *shape* accuracy
alone sits mildly above its 0.25 chance level for every arm (0.28-0.44),
while *color* accuracy alone stays almost exactly at chance (0.19-0.31) —
a faint, uniform signal that isn't specific to any one objective and
doesn't move the joint metric, since both attributes must be correct
simultaneously.

**This changes the diagnosis, not the conclusion.** Reformatting to
ordinary English removed the "confusing symbolic syntax" explanation
without changing the outcome at all — the floor is not an artifact of
`x=p`-style notation specifically. That narrows the likely cause from "the
text format was out-of-distribution" to something closer to "the *task
structure itself* (resolving a variable through an indirection table) was
never encountered during pretraining, in any register, by any of the four
arms, and a frozen linear probe on 192 examples cannot make up that gap
regardless of which objective produced the frozen features." This is a
different, more specific null result than the symbolic-only run
suggested — and it is still a null result, not a finding that composition
"fails" for any objective: the experiment as designed (frozen probe, zero
pretraining exposure to indirection, ~200 training examples) may simply be
under-powered to detect this ability in *any* of the four arms, AR
included, regardless of whether that ability exists.

## What would actually resolve this

1. ~~Reformat the deep task in natural-language register~~ — **done above;
   ruled out text format as the explanation, did not resolve the question.**
2. **Give the deep-task's *structure* some exposure during pretraining**
   (mix a modest fraction of indirection-style text — natural-language
   register, per above — into `build_pretrain_corpus`) before evaluating
   with a frozen probe, so a floor result can no longer be blamed on zero
   exposure to the task type at all. This is now the most likely candidate
   fix, having ruled out (1).
3. ~~Fine-tune rather than frozen-probe on the deep task~~ — **done, see
   `PHASE1_5_FINETUNE_INDIRECTION_RESULTS.md`. Result: overfitting, not
   learning — both AR and AR+JEPA-aux reach perfect train accuracy and
   chance-or-worse test accuracy. This is a third inconclusive outcome for
   a different reason (data scarcity relative to the task's combinatorial
   structure), not a resolution.**
4. **More training examples** (currently 192) — 2-hop indirection with a
   4x4 permutation space is a combinatorially richer rule than the shallow
   task's direct lexical mapping; it may need more than 12 pairings x 16
   replicates to be learnable via a frozen linear probe at all, independent
   of the pretraining-exposure question.
5. Until one of the above changes the floor result, **the shallow-task
   compositional gap (AR 0.938-1.000 vs. JEPA 0.625-0.812 vs. contrastive
   0.188-0.562, narrowed but not closed by the readout-pyramid follow-up)
   remains the best available toy-scale evidence** — weakened by the
   lexical-shortcut caveat, but not superseded by either deep-task run,
   since both have their own unresolved confound (zero task-structure
   exposure during pretraining, and possibly too little probe-training
   data).

## Bottom line for the design question

This still does not move the JEPA-vs-AR question in either direction. Two
rounds of this experiment (symbolic, then natural-language) have narrowed
the explanation for the floor result from "confusing text format" to "zero
pretraining exposure to this task structure, in a frozen-probe evaluation,
on a small training set" — a real, if less flashy, finding about the
experiment's own limits. The next move that could actually produce a
signal is (2) above: give the pretraining corpus some exposure to
indirection-style text before evaluating, since both text-format and
"just add more data" explanations have now been addressed or ruled out
without resolving the floor.
