# Exploring Alternatives: VSA Binding on the COGS-Style Task

Status: **on a task with real, learnable signal, the VSA-augmented
encoder does not tie the plain backbone — it loses to it, clearly and
consistently, on both COGS-style tasks.** Task 1 (department/urgency):
plain 0.406 vs. VSA 0.266 test joint accuracy. Task 2 (JSON name/status,
replication): plain 0.750 vs. VSA 0.250 — an even wider gap. Combined with
`VSA_INDIRECTION_RESULTS.md` (a tie at floor on indirection), the VSA
binding mechanism as integrated here has not shown an advantage on any
task tested, and shows a real disadvantage on the tasks where a
difference could be measured at all.

## Why this task, and what's different about training here vs. earlier COGS results

`VSA_INDIRECTION_RESULTS.md` found VSA-augmented and plain encoders fail
identically on the indirection task — both floor. That result can't
distinguish "VSA doesn't help" from "nothing could help on a task with
zero pretraining exposure and 192 examples" (the exposure confound
documented throughout `PHASE1_5_DEEP_COMPOSITION_RESULTS.md`). The
COGS-style tasks (`build_cogs_style_task` / `build_cogs_style_task_v2`)
are the one place in this project where a compositional-generalization
task reliably produces real, well-above-chance signal — worth testing
the same VSA-vs-plain comparison there.

**Important framing difference from the earlier COGS-style results**
(`PHASE1_5_COGS_STYLE_RESULTS.md`, `PHASE1_5_COGS_REPLICATION_RESULTS.md`):
those experiments froze an already-*pretrained* checkpoint and trained
only a linear probe on top. This experiment trains `VSAEncoder` and
`BidirectionalPooledBackbone` **from scratch**, directly on each COGS
task's own 192 training examples, matching the from-scratch protocol used
for the indirection-task comparison (there is no separate pretraining
phase for either model here). This is a different, harder regime for
both architectures — no general-language prior, only the task's own small
training set — but it keeps the VSA-vs-plain comparison apples-to-apples
with the indirection-task experiment, which is the comparison this
exploration is actually about.

## Result

`scripts/diagnose_vsa_indirection.py --task cogs` / `--task cogs_v2`,
matched training (lr=1e-3, 500 epochs, from scratch, same protocol as the
indirection-task comparison).

| Task | Model | Params | Train joint acc. | Test joint acc. |
|---|---|---:|---:|---:|
| Task 1 (dept/urgency) | VSA-augmented | 134,400 | 0.927 | **0.266** |
| Task 1 (dept/urgency) | Plain backbone | 117,760 | 1.000 | **0.406** |
| Task 2 (json name/status) | VSA-augmented | 134,400 | 1.000 | **0.250** |
| Task 2 (json name/status) | Plain backbone | 117,760 | 1.000 | **0.750** |

(Chance = 0.0625 for joint accuracy.) Both models clear chance by a wide
margin on both tasks — unlike the indirection task, there is real signal
here for both architectures to find. **The plain backbone finds
substantially more of it.** The gap is not small (14 points on task 1,
50 points on task 2) and it replicates in the same direction on an
independently-constructed second task using different primitives from a
different corpus tier — this is not noise.

## Reading

- **This is the first task in this exploration where VSA and plain
  produce a real, measurable difference — and it favors plain, not
  VSA.** The indirection-task comparison was a tie at floor; this is a
  clear loss. Combined, VSA binding as integrated here has not won on
  anything tested.
- **Plausible mechanism: these tasks don't need binding, and the added
  pathway may actively hurt.** Department/urgency classification is
  closer to detecting two independently-present lexical features (similar
  in spirit to the original shallow composition task, which every
  architecture solved easily from layer 1 in
  `PHASE1_5_LAYER_PYRAMID_RESULTS.md`) than to resolving an indirection
  chain. `VSAMemoryModule` adds a retrieval pathway summed residually into
  every position's hidden state regardless of whether the task benefits
  from it — and per `VSA_INDIRECTION_RESULTS.md`'s finding that
  superposed-memory retrieval carries substantial cross-talk that doesn't
  average down with dimension, that pathway's output is inherently noisy.
  For a task that doesn't need retrieval, injecting that noise into an
  otherwise cleanly-separable representation is a plausible, mechanistic
  explanation for a real generalization cost, not just "more parameters,
  more overfitting" (the plain backbone, with fewer parameters, actually
  overfits *less*: its train-test gap is smaller on task 1, 0.594 vs.
  VSA's 0.661, and dramatically smaller on task 2, 0.250 vs. 0.750).
- **This is a genuine, useful negative finding, not a wash.** An explicit
  inductive bias suited to one class of problem (indirection/binding) can
  actively hurt on a different class of problem (direct feature
  detection) that doesn't need it — exactly the kind of result that argues
  against adding VSA binding as a general-purpose, always-on Core
  component, rather than a targeted mechanism invoked only when binding is
  actually needed (e.g. gated, or added only in an ablation-tested
  configuration).

## Where this leaves the VSA exploration

Across both tasks tested (indirection: tie at floor; COGS-style: clear
loss, replicated), **the explicit VSA binding mechanism as integrated in
this exploration (residual addition after the backbone, always active)
has not shown a benefit anywhere, and has shown a real cost on the one
task type where a difference was measurable.** This doesn't rule out VSA
mechanisms more broadly — per `VSA_INDIRECTION_RESULTS.md`'s follow-ups,
a gated or role-segregated integration might behave differently — but it
does mean this specific design shouldn't be adopted as a Core component
without a version that fixes the demonstrated cost on non-binding tasks.

## Follow-ups

1. **Gate the memory pathway** — a learned scalar (or small MLP) deciding
   how much of `VSAMemoryModule`'s output to add at each position, so the
   model can suppress it on tasks/positions where it isn't useful instead
   of always injecting retrieval noise. Directly targets the mechanism
   proposed above for why VSA lost on these two tasks.
2. **Ablate whether the gap is from the memory pathway's noise or from
   optimization difficulty specifically** — train `VSAEncoder` with the
   memory module's output zeroed out (equivalent to the plain backbone
   plus dead-but-present extra parameters) to check whether the loss is
   simply "harder to optimize with these extra unused parameters" vs.
   "the retrieved signal itself is actively harmful."
3. Given VSA has now been tested on the one task type with real signal
   and lost, further VSA exploration should prioritize fixing this
   specific deficit (via follow-up 1) before testing additional tasks —
   testing more tasks with the same un-gated design is unlikely to change
   this picture.
