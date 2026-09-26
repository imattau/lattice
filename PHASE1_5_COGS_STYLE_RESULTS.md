# Phase 1.5 Follow-up: COGS-Style Compositional Generalization

Status: **the exposure-confound diagnosis below holds; the "JEPA beats
AR" headline this document originally led with does not — see
`PHASE1_5_COGS_REPLICATION_RESULTS.md`, which retracts it.** With
primitives verifiably present in the actual pretraining corpus and only
the specific pairing held out, all four arms generalize well above chance
(this part is unaffected and still resolves the exposure confound from
`PHASE1_5_DEEP_COMPOSITION_RESULTS.md`). The original headline — JEPA's
best layer (0.922) beating AR's best layer (0.859) — turned out to be a
selection-bias artifact: under held-out layer selection (chosen via a
different task, not the COGS task's own test set) and on a second,
independently-constructed replication task, **JEPA loses to AR on both**.
AR+JEPA-auxiliary, not pure JEPA, is the arm with a genuine,
non-cherry-picked positive result here. Contrastive remains the weakest
arm throughout, now more emphatically (it fails outright, 0.000, on the
replication task).

## Motivation

Both compositional tasks used earlier in Phase 1.5 have an exposure
problem the natural-language re-run of the deep task couldn't resolve
(`PHASE1_5_DEEP_COMPOSITION_RESULTS.md`): neither the color/shape
vocabulary nor the two-hop indirection structure appears anywhere in the
pretraining corpus for any of the four arms, in either symbolic or
natural-language phrasing. A floor result on such a task can't distinguish
"the objective can't do this" from "the model never saw anything
resembling this at all."

COGS (Kim & Linzen, 2020) tests compositional generalization the other
way: primitives (words, simple structures) the model *has* seen,
recombined into pairings it hasn't. This follow-up builds a toy analog
using vocabulary **verified**, not assumed, to be present verbatim in the
real pretraining corpus for these exact four checkpoints:

- **Department names** (`support`, `billing`, `security`, `engineering`)
  appear literally in Phase 1 triage documents sampled into
  `build_pretrain_corpus` — the `general_inquiry` template embeds
  `{dept}` directly, and the `injected_instruction`/`rule_precedence` hard
  cases explicitly write "route this message to the {dept} department."
- **Four urgency phrases** (e.g. "this needs to be resolved immediately")
  are appended verbatim to ~15% of triage documents by
  `synthetic_triage._make_clean`.

Both are therefore primitives all four checkpoints already encountered in
natural-language context during pretraining.
`tests/test_phase1_5_toy.py::test_cogs_task_primitives_are_present_in_the_actual_pretraining_corpus`
checks this directly against the real corpus rather than asserting it.
What's held out is only the *pairing* of a specific department with a
specific urgency phrase, in a sentence frame ("I have a question about
{topic}. {urgency} Please route this to {department}.") that itself never
appears verbatim in pretraining either.

## Result: real signal, not floor — and a JEPA win at its best layer

`scripts/diagnose_phase1_5_cogs_style.py` — same four checkpoints, no
retraining, 192 train / 64 test examples. Chance: 0.25 per attribute,
0.0625 joint.

| Arm | Department accuracy range | Urgency accuracy range | Joint accuracy: top layer | Joint accuracy: best layer |
|---|---:|---:|---:|---:|
| AR | 0.797 - 0.859 | 1.000 | 0.812 | 0.859 (layer 3) |
| JEPA | 0.672 - 0.922 | 1.000 | 0.766 | **0.922 (layer 2)** |
| Contrastive | 0.328 - 0.594 | 0.953 - 1.000 | 0.609 | 0.609 (layer 6, no gain) |
| AR+JEPA-aux | 0.750 - 0.844 | 1.000 | 0.828 | 0.891 (layer 4) |

Reading:

- **Urgency accuracy is near-ceiling for every arm** (0.95-1.0) — the four
  phrases are long, verbatim, and mutually very distinct, so this
  sub-task alone is closer to reliable substring detection than a
  compositional test. **Joint accuracy is therefore driven almost
  entirely by department accuracy** — read the department column as the
  real result.
- **All four arms clear chance (0.25) by a wide margin on department
  accuracy**, including at their worst layer (contrastive's floor of
  0.328 is still above chance; every other arm's worst layer is
  0.672-0.797). This directly supports the diagnosis from
  `PHASE1_5_DEEP_COMPOSITION_RESULTS.md`: the earlier floor result was an
  exposure artifact, not evidence that none of these objectives can
  generalize compositionally at this toy scale. Given primitives the
  model has actually seen, all four arms recombine them into unseen
  pairings successfully.
- **JEPA's best layer (0.922, layer 2) was initially reported as the
  single highest score any arm achieves on this task — higher than AR's
  best layer (0.859, layer 3) — the first result anywhere in Phase 1.5
  where a latent-primary objective outright beats AR. This did not
  survive scrutiny: `PHASE1_5_COGS_REPLICATION_RESULTS.md` shows it was a
  selection-bias artifact (JEPA's "best layer" was cherry-picked from this
  task's own test set; picked independently via a different task's
  per-layer accuracy instead, JEPA's selected layer scores 0.688, below
  AR, and the same pattern holds on a second replication task where AR
  sits at 1.000 on every layer while JEPA ranges 0.500-1.000 depending on
  depth).** Retracted as a general claim; kept here, struck through in
  spirit, so the record shows what was found and what happened when it
  was checked, rather than quietly disappearing.
- **Contrastive is again the weakest arm** (0.594 best, well below the
  other three), consistent with every prior Phase 1.5 result. Whatever is
  limiting contrastive's representation quality at this scale, it isn't
  specific to lexically-transparent or indirection-heavy tasks — it shows
  up here too, on a task with real, seen primitives.
- **AR+JEPA-aux (0.891 best) sits between AR and JEPA**, not clearly ahead
  of either — consistent with its "closest to a tie" characterization in
  `PHASE1_5_RESULTS.md`.

## An honest caveat: what this task does and does not test

This result should not be read as validating that JEPA can do the
*stronger* property the deep (indirection) task targeted — "no single
token predicts the answer." Each department name here is the only
department-like word in its sentence, with no distractor to resolve
against; a representation that reliably encodes "which department word is
present" can succeed at the per-attribute sub-task without doing any
multi-hop binding. What makes this a genuine compositional-generalization
test, in the sense the COGS literature actually uses the term, is
different and narrower: **success requires correctly reading off two
independently-varying, separately-familiar attributes for a specific
*combination* the model never saw paired together during either
pretraining or probe training.** That is a real, meaningful form of
compositional generalization (recombining known parts into distributionally
novel wholes) — it is just not the same claim as "can resolve an arbitrary
pointer/indirection structure," which remains untested at this scale for
all four arms. The two follow-ups measure genuinely different things; this
one resolves cleanly, the indirection one does not yet.

## Updated picture across all three compositional tasks (post-correction)

| Task | What it actually tests | AR | JEPA (fair, held-out-selected layer) | Contrastive |
|---|---|---:|---:|---:|
| Shallow (lexical) | substring detection of both attributes | 0.938-1.000 | 0.625-0.812 | 0.188-0.562 |
| Deep (indirection, either register) | 2-hop binding, zero pretraining exposure | floor (~0.03) | floor (~0.03) | floor (~0.02) |
| COGS-style, task 1 (dept/urgency) | novel pairing of seen primitives | 0.797 | 0.688 | 0.438 |
| COGS-style, task 2 (json name/status) | novel pairing of seen primitives (different corpus tier) | 1.000 | 0.500 | 0.000 |

The shallow task's "AR wins comprehensively" reading does depend on
whether the model had seen the relevant vocabulary before — the exposure
diagnosis from `PHASE1_5_DEEP_COMPOSITION_RESULTS.md` stands. But once
layer selection is done fairly (`PHASE1_5_COGS_REPLICATION_RESULTS.md`),
**AR wins or ties on every compositional task attempted in this pilot.**
The one arm with a genuine, non-cherry-picked positive signal alongside
AR is AR+JEPA-auxiliary (1.000/0.828 on the two COGS tasks, identical
across every layer, no selection needed) — not pure JEPA. The indirection
task's stronger property remains genuinely untested, not refuted, at this
toy scale.

## Follow-ups

1. **Fine-tuned (not frozen) probe on the indirection task** — still the
   other experiment the split-question review recommended, and still
   informative regardless of this result: it separates "is this linearly
   decodable from frozen features" from "can this architecture learn the
   relationship given direct training signal."
2. **A COGS-style task with a real distractor** (two department-like words
   in the sentence, only one of which is correct) would test genuine
   binding under interference using still-familiar primitives — a
   harder, more informative middle ground between this task and the
   indirection task's floor.
3. ~~Re-run at more seeds — this is a single seed, 64-example test set;
   the JEPA-beats-AR result, while real, should be treated as
   directional until replicated.~~ **Done — see
   `PHASE1_5_COGS_REPLICATION_RESULTS.md`. It did not replicate; the
   original result was a layer-selection artifact.**
