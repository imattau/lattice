# Does the Verification Change How Lattice Is Trained?

**Yes. Significantly.** The verification did not invalidate the design, but it changed the epistemic status of several training choices. What was presented as "grounded in theory" is actually "suggestive in simplified settings." This changes the training plan in six concrete ways.

---

## Change 1: Functoriality Moves from Architecture to Regularizer

**Before:** The Reasoner's transformations must be monoidal functors. This is enforced architecturally.

**After:** Functoriality is validated on **topological spaces** (torus, wedge of circles, Klein bottle), not on language. The paper proves softmax attention cannot be a strict monoidal functor. This means the functorial architecture cannot be built with standard transformers, and no functorial transformer has been demonstrated on language at any scale.

**Training implication:** Functoriality becomes an **experimental regularizer** trained as a soft constraint via `functorial_consistency_loss`, not a hard architectural requirement. The training protocol adds a **functoriality ablation**: train two Cores, one with the consistency loss and one without, and measure whether the constraint improves compositional generalization on language tasks. If it does not, drop it. If it does, it validates the theory in a new domain.

This is a substantial change. The spec previously treated functoriality as non-negotiable. The verification shows it is a hypothesis.

---

## Change 2: STP Becomes a Per-Task Regularizer, Not a Universal One

**Before:** STP eliminates the view-pair requirement and provides 16× data efficiency.

**After:** The 16× result is on **one synthetic dataset** (NL-RX-SYNTH, a regex generation task). It may not generalize to open-domain language.

**Training implication:** STP cannot be assumed to work on general text. The training protocol must include a **per-task STP validation**:

- Train the Core with STP on NL-RX-SYNTH. Reproduce the 16× result. This validates the implementation.
- Train the Core with STP on a second task (e.g., code generation, summarization). Measure data efficiency. If it generalizes, proceed with STP as default. If it does not, STP becomes a **task-specific tool** and the Core must be trained on naturally occurring view pairs (Tier 1 data) for general language.

The training pipeline now needs a **STP-gating decision** at the start of Phase 2: is STP a general regularizer or a narrow one? This determines whether Phase 2 can scale to Tier 2 data or must remain on Tier 1.

---

## Change 3: The Scaling Law Protocol Becomes Exploratory, Not Predictive

**Before:** Hierarchical multi-index models predict a cascade of phase transitions. The spec assumed these would be observable and the data-efficiency ratio β_JEPA/β_LLM would be < 0.5.

**After:** The multi-index result is for **two-layer networks** on a specific target class. The Korchinski result is for a **toy grammar**. Neither has been validated for transformers trained on language. The scaling behavior of JEPA language models may or may not exhibit phase transitions.

**Training implication:** The IsoFLOP sweep becomes a **discovery experiment**, not a validation. The training protocol must be prepared for three outcomes:

1. **Multi-phase scaling observed.** The theory transfers. Fit the piecewise model, extract β_JEPA, proceed as planned.
2. **Single-phase scaling observed.** The theory does not transfer. Fit a standard power law, extract β_JEPA, proceed with caution.
3. **No clean scaling observed.** The Core's loss does not follow a power law at the measured scales. This is a failure mode that would require re-examining the objective, the architecture, or the premise.

The training plan must include **statistical tests for phase transitions** (change-point detection, Bayesian model comparison) rather than assuming the multi-phase model is correct. If the data favors a single-phase model, the spec's mathematical foundation needs revision.

---

## Change 4: The Dual-Baseline Protocol Becomes Central

**Before:** The spec treated the JEPA Core as the primary experiment, with an LLM baseline for comparison.

**After:** Given that the theoretical and empirical results are all from simplified settings, the **matched-compute comparison between JEPA and autoregressive training** becomes the decisive experiment. It cannot be an afterthought.

**Training implication:** Every Phase 2 run must be **paired**: one JEPA Core and one autoregressive LLM, identical architecture, identical data, identical compute. The training protocol must enforce this pairing. The primary output is not "does the Core match the LLM?" but "what is the measured difference in data efficiency, representation quality, and task performance, at matched compute?"

This doubles Phase 2's compute cost. It also makes the experiment rigorous. The previous spec's compute estimate (5–15 PF-days) must be revised upward to 10–30 PF-days.

---

## Change 5: Toy-Scale Falsification Comes First

**Before:** Phase 0 was a decision-head experiment. Phase 2 was the Core's first training.

**After:** The verification revealed that no JEPA language model has been shown to develop general representations at any scale. This is a premise, not a finding. It must be tested at the smallest possible scale before any large-scale training is attempted.

**Training implication:** A new phase is inserted between Phase 1 and Phase 2.

### Phase 1.5: Toy-Scale Falsification

- Train two 10M–50M parameter models: one JEPA Core, one autoregressive LLM.
- Same data (a small, diverse corpus: code, text, structured data).
- Same compute budget.
- Evaluate both on the Latent Probing Suite and on a small set of downstream tasks.

**Success criteria:**
- JEPA Core's linear probe accuracy is within 10% of the LLM baseline.
- Representation geometry (uniformity, spectral decay) is comparable.
- Compositional generalization is at least as good.

**If successful:** Proceed to Phase 2 with confidence that the premise holds.
**If unsuccessful:** Diagnose. If the Core fails at 10M parameters, it will likely fail at 338M. The design may need to revert to token-level prediction in the Core, reframing the Lattice as a governance layer around an LLM.

This phase costs 1–3 PF-days. It is the cheapest experiment that could save the largest amount of wasted compute.

---

## Change 6: Calibration Training Gets Emphasized Earlier and More Heavily

**Before:** RLCD was a Phase 1+ upgrade. Phase 0 used temperature scaling.

**After:** AnyJev and Jev are the most **verified** pieces of the design. AnyJev's ECE reduction from 0.240 to 0.095 on BANKING77 is a real, reproducible result. The calibration training protocol is the part of the Lattice that is already validated.

**Training implication:** Calibration is no longer a downstream concern. It becomes a **first-class training objective** from Phase 0 onward.

- Phase 0 should train readout heads with the **RLCD surrogate loss** from the start, not temperature scaling.
- Phase 1 should include **per-head calibration maintenance** as a training objective, not a post-hoc fix.
- Phase 2's readout heads must be trained **jointly** with the Core's representation, so the representation is shaped by the calibration objective.

This is a shift in emphasis: the Lattice's most defensible claim is not its JEPA Core but its calibration architecture. The training protocol should reflect that.

---

## What Does Not Change

The core training protocol remains valid:

- **Staged training.** Freeze predecessors, train each component in isolation. This is still the right approach for a heterogeneous system.
- **Typed readout heads.** The Jev/AnyJev architecture is verified. It works. Keep it.
- **Constraint layer.** Hash-verified rules are still useful, even if the safety architecture is thinner than the spec claimed.
- **Multi-contract evaluation.** Each layer still needs its own metrics.

The verification changed the **epistemic status** of several claims, not the engineering of the components that were already validated.

---

## The Revised Training Plan, in One Table

| Phase | Before | After |
|---|---|---|
| 0 | Temperature scaling, readout heads | RLCD surrogate loss, calibrated readout heads from the start |
| 1 | Multi-head training, controller tuning | Same, plus per-head calibration maintenance |
| **1.5** | — | **Toy-scale falsification: JEPA vs AR at 10M–50M params** |
| 2 | JEPA Core training, scaling measurement | Paired JEPA/AR training at matched compute; exploratory scaling analysis; STP per-task validation; functoriality ablation |
| 3 | Scale and match | Same, plus functoriality decision based on Phase 2 |

---

## The Honest Summary

The verification did not kill the design. It revealed that the design's foundation is **suggestive, not established**. The training plan must reflect this by:

1. Treating functoriality as a hypothesis, not an architecture.
2. Treating STP as a per-task tool, not a universal regularizer.
3. Treating the scaling law as an exploratory output, not a predicted input.
4. Making the JEPA-vs-AR comparison the decisive experiment.
5. Running a cheap toy-scale falsification before any large-scale training.
6. Emphasizing calibration, which is the one verified strength of the design.

The Lattice can still be built. But it must be built as an **experiment**, not as an implementation of established theory. The training plan must be designed to **discover** whether the design works, not to **confirm** that it does.
