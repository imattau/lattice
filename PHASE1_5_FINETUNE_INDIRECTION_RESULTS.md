# Phase 1.5 Follow-up: Fine-Tuned Probe on the Indirection Task

Status: **fine-tuning doesn't rescue the indirection task either — it just
overfits.** Both AR+JEPA-auxiliary and plain AR reach perfect training
accuracy (1.000) and near-chance-or-worse test accuracy (0.000 and 0.016;
chance is 0.0625). This is memorization, not generalization, and it
happens identically for both arms — the fine-tuned-probe experiment
doesn't distinguish the objectives on this task; it reveals a shared
limitation of the experimental setup at this data scale.

## Motivation

Every prior compositional-generalization result in Phase 1.5 used a
**frozen** linear probe — the encoder's weights never update; only a small
head on top is trained. That protocol can only recover what's already
linearly decodable in the frozen representation, and the indirection task
floored at chance for every arm under it, in both symbolic and
natural-language register (`PHASE1_5_DEEP_COMPOSITION_RESULTS.md`).
Fine-tuning answers a different, complementary question: **given direct
gradient signal on the task itself, not just a frozen readout, how
learnable is the relationship from these starting weights?** Per the
COGS-replication follow-up's prioritization (pure JEPA has no surviving
compositional win; AR+JEPA-auxiliary is the one latent-touching arm with a
genuine positive result), this was run on AR+JEPA-auxiliary specifically,
with plain AR alongside as a reference point.

`finetune_composition_accuracy` (`lattice/phases/phase1_5_toy.py`)
unfreezes a copy of the loaded checkpoint's backbone (the checkpoint
itself is never mutated) and trains it end-to-end, jointly with a small
two-head classifier, directly on the indirection task's own 192 training
examples — full-batch AdamW, natural-language register, 500 epochs,
lr=1e-4. `scripts/diagnose_phase1_5_finetune_indirection.py`.

## Result

| Arm | Train joint accuracy | Test joint accuracy | Final loss |
|---|---:|---:|---:|
| AR+JEPA-aux | 1.000 | **0.000** | 0.020 |
| AR | 1.000 | **0.016** | 0.013 |

(Chance = 0.0625 for joint accuracy on this 4x4 task.) Both arms drive
training loss to near-zero and reach perfect per-attribute training
accuracy (1.000 color, 1.000 shape, for both arms) — confirming the model
has more than enough raw capacity to fit 192 examples exactly. Both then
score at or below chance on the 64 held-out test examples. **This is
textbook memorization**: with direct gradient access to only 192 training
examples covering a 4x4 permutation-based rule, both models learned to
answer the specific training examples rather than the general "resolve
the pointer, then look up its value" rule, and generalize no better than
random guessing to examples that require applying that rule to a
combination never seen during training.

## Reading

- **This does not distinguish AR+JEPA-auxiliary from AR.** Both fail in
  the same way, to a similar degree (AR's test accuracy of 0.016 is
  technically slightly worse than AR+JEPA-aux's 0.000, but both are
  indistinguishable from chance noise at n=64). Whatever governs whether
  either model can learn genuine two-hop indirection, it isn't being
  revealed by this experiment — the experiment is dominated by an
  overfitting failure mode common to both.
- **This does not resolve the frozen-probe floor's ambiguity either.**
  The frozen result couldn't tell you whether the representation lacked
  the answer or the model couldn't learn the relationship at all. This
  result shows the model *can* drive training loss to zero (so it isn't
  simply incapable of representing the mapping), but *cannot* generalize
  from 192 examples to unseen combinations of it — a third, different
  finding, not a resolution of the first two.
- **Most likely cause: too little data for this task's combinatorial
  structure, not a property of either objective.** 192 training examples
  drawn from 12 in-distribution (color, shape) pairings is very little
  data for a rule defined over a 4x4x4x4-ish permutation space (each
  example uses an independently randomized link-to-value mapping). Full
  fine-tuning with no regularization beyond weight decay defaults, no
  early stopping on a validation split, and no architectural bias toward
  pointer-following will predictably memorize under these conditions
  regardless of pretraining objective.

## What would actually test the intended question

1. **Early stopping on a validation split**, not a fixed epoch count —
   500 epochs was chosen to drive training loss down "enough to see if
   the task is learnable at all," but it also guarantees overfitting once
   training accuracy hits 1.000 well before epoch 500. Stopping when
   validation accuracy peaks (using a held-out slice of the *training*
   distribution, distinct from the test pairings) would show whether
   either arm ever passes through a state with real generalization before
   memorizing.
2. **More training examples per pairing** (or more in-distribution
   pairings) — directly addresses the combinatorial-data-scarcity
   explanation above, independent of any change to the objectives being
   compared.
3. **Regularize more aggressively** (dropout, stronger weight decay, or
   freezing most layers and fine-tuning only the last one or two) to bias
   the optimization away from memorization.
4. Until one of the above changes the outcome, **this experiment is
   inconclusive about whether AR+JEPA-auxiliary's representation supports
   genuine indirection better than AR's** — it only shows that naive
   full fine-tuning on 192 examples overfits for both, which was a live
   possibility going in and is itself worth knowing before investing in a
   larger version of this specific setup.

## Where this leaves the design question

The indirection task remains the load-bearing unresolved experiment for
the design's central bet, exactly as it was before this follow-up — three
different evaluation protocols (frozen probe x2 registers, full
fine-tuning) have now been tried on it, and all three have failed to
produce an interpretable signal for a different reason each time (zero
pretraining exposure; zero pretraining exposure again in natural
register; catastrophic overfitting under fine-tuning). The COGS-style
result remains the only place in this pilot where a compositional
question was actually answered cleanly, and it favored AR
(`PHASE1_5_COGS_REPLICATION_RESULTS.md`).
