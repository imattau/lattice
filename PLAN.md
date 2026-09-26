# Detailed Phase Planning: The Lattice (Reconciled)

This document is the authoritative implementation plan. It expands the phased build plan (Spec §10) into implementation-grade detail and incorporates the verification-driven revisions recorded in `PLAN_REVISION.md`.

The verification did not invalidate the design, but it changed the epistemic status of several training choices. What was presented as "grounded in theory" is actually "suggestive in simplified settings." This reconciled plan reflects that shift: the Lattice is built as an experiment designed to discover whether the design works, not to confirm that it does.

---

## Overview of Revisions

| Area | Original plan | Revised plan | Rationale |
|---|---|---|---|
| **Functoriality** | Hard architectural requirement | Experimental regularizer + ablation | Softmax attention cannot be a strict monoidal functor; no functorial transformer exists for language |
| **STP** | Universal regularizer, eliminates view pairs | Per-task regularizer with gating decision | 16× result is on one synthetic dataset (NL-RX-SYNTH) |
| **Scaling law** | Predictive multi-phase model | Exploratory analysis with model comparison | Multi-index theory is proven only for two-layer networks and toy grammars |
| **Baseline** | JEPA Core with LLM comparison | Paired JEPA/AR training at matched compute | Decisive experiment must be rigorous; theoretical priors are weak |
| **Falsification** | Phase 0 only | **New Phase 1.5**: toy-scale JEPA vs AR at 10M–50M params | Cheapest way to test whether JEPA develops general language representations |
| **Calibration** | Post-hoc temperature scaling (Phase 0), RLCD later | First-class training objective from Phase 0 | AnyJev/Jev are the most verified components of the design |

---

## Phase 0: Decision Heads + Constraint Layer

### Objective

Demonstrate that a typed decision layer with calibrated confidence, wrapped in a verifiable constraint layer, matches or exceeds a generative baseline on a bounded decision task—at a fraction of the cost and latency.

### Architecture

| Component | Implementation | Rationale |
|---|---|---|
| **Encoder** | Frozen LLM backbone (Qwen3-8B or equivalent) | Reuse existing representations; isolate the decision-head question from the representation question |
| **Readout Heads** | Linear + 2-layer bidirectional head over sequence | JevK5 architecture: softmax over option-letter logits divided by a calibration temperature |
| **Constraint Layer** | Hand-written rules, hashed at load | Spec §8.1; verify policy hash before each call |
| **Controller** | Deterministic Python, threshold policies | Spec §3.4 |

**Option encoding.** Each option is represented by a marker token (e.g., `[MASK]` or a dedicated `<|option_k|>` token). The readout head scores each option's marker position, applies a calibrated temperature, and produces a distribution over the option set. This is the SemIf protocol used in JevK5.

**Calibration.** Use the **RLCD surrogate loss** from the start (`reference/calibration.rlcd_loss`). Fit a post-hoc temperature per decision type on a held-out calibration set as a fallback and for audit. AnyJev reports ECE improving from 0.240 (raw) to 0.095 on BANKING77; this is the most verified result in the design and should be the default training objective.

### Data

| Dataset | Purpose | Size |
|---|---|---|
| **BANKING77** | Intent routing decision | 13,083 utterances, 77 classes |
| **MultiNLI / WANLI** | Natural language inference decision | ~430K / ~107K |
| **BoolQ** | Boolean decision | ~16K |
| **MMLU-Pro** | Multi-choice decision | ~12K |
| **Synthetic ticket data** | Domain-specific routing (if needed) | 5K–10K |

Split: 70% train, 15% calibration, 15% test. The calibration set must be **disjoint** from the test set.

### Training Procedure

**Stage 0a: Calibrated readout head training.**

- Freeze the encoder.
- Train the readout head with a weighted combination of cross-entropy and the RLCD surrogate loss.
- 2 epochs, learning rate 3e-5, AdamW.
- Batch size 32, max sequence length 512.

**Stage 0b: Temperature fitting (fallback / audit).**

- Compute logits on the calibration set.
- Fit a single scalar temperature per decision type by NLL minimization (LBFGS, 200 iterations).
- Store temperature artifacts alongside the model checkpoint.

**Stage 0c: Constraint layer integration.**

- Load the constraint policy; compute and record its hash.
- Every decision call verifies the hash before evaluation.
- Constraint violations trigger a deterministic fallback (escalation).

### Evaluation Protocol

| Metric | Threshold | Method |
|---|---|---|
| **ECE** | Below LLM baseline by ≥ 20% relative | Expected calibration error over 15 bins |
| **Accuracy** | ≥ baseline | Argmax accuracy on test set |
| **Brier score** | Below LLM baseline | Multiclass Brier |
| **Order-flip rate** | < 5% | Cyclic-shift marginalization removes option-order bias |
| **Constraint violations** | Zero | Count of constraint-layer rejections on valid inputs |
| **Latency (p50)** | < 50ms on CPU | Wall-clock per decision |
| **Latency (p95)** | < 200ms | Including calibration and constraint evaluation |
| **Cost per decision** | < 1% of frontier LLM call | Token accounting |

**Generative baseline.** A frontier LLM prompted with the same decision task, parsed with constrained decoding (JSON schema or regex). Measure the same metrics.

### Decision Gate

**Proceed to Phase 1 if:**
- ECE is lower than the generative baseline by ≥ 20% relative.
- Constraint violations are zero.
- Latency is below 50ms (p50) on CPU.
- Accuracy is within 2% of the generative baseline.

**If not:**
- Diagnose whether the encoder's representations are insufficient. If so, Phase 1 begins with encoder adaptation.
- If the readout head architecture is insufficient, try a deeper head or a contrastive objective.
- If calibration fails, investigate whether the task is inherently multi-modal (requiring more calibration data).

---

## Phase 1: Minimal System (Document Triage)

### Objective

Build a deployable document triage system with multiple decision heads, a deterministic controller, selective generation, full audit logging, and per-head calibration maintenance.

### Architecture

| Component | Implementation | Rationale |
|---|---|---|
| **Encoder** | Same as Phase 0 (frozen LLM backbone) | Reuse |
| **Readout Heads** | Bank of 4–6 heads: urgency (3-way), department (10-way), escalation (binary), confidence (5-way score), safety (binary) | Multiple decisions from shared representation |
| **Talker** | Same LLM backbone, invoked only when routing confidence > 0.6 | Selective generation |
| **Controller** | Python, threshold policies per decision type | Deterministic |
| **Constraint Layer** | Extended rules: PII detection, blacklist routing, mandatory escalation | Spec §8.1–8.2 |
| **Audit Logger** | Structured JSON logs per decision | Every readout, every constraint trace, every action |

### Data

| Dataset | Purpose |
|---|---|
| **Phase 0 datasets** | Reuse for readout head training |
| **Synthetic triage corpus** | 20K–50K documents with multi-label decisions |
| **Human-annotated triage set** | 500–1,000 documents for validation |

The synthetic corpus is generated by a frontier LLM with a structured prompt template covering 15–20 document types and hard decision cases (ambiguity, misleading metadata, injected instructions, rule precedence).

### Training Procedure

**Stage 1a: Multi-head readout training with calibration objective.**

- Freeze encoder.
- Train all readout heads jointly with per-head cross-entropy + RLCD surrogate loss.
- Weight the loss inversely proportional to class frequency.
- 3 epochs, learning rate 3e-5.
- Joint training encourages the heads to share the encoder's representation effectively.

**Stage 1b: Controller threshold tuning.**

- Freeze encoder and heads.
- Tune controller thresholds on the validation set to optimize a composite objective: accuracy × coverage − cost × generation_rate.
- Use grid search or Bayesian optimization over 3–4 thresholds.

**Stage 1c: Calibration per head.**

- Fit temperature per head on the calibration set.
- Store temperatures as metadata and include them in audit logs.

### Evaluation Protocol

| Metric | Threshold |
|---|---|
| **Routing accuracy** | ≥ baseline (frontier LLM with JSON output) |
| **Calibration (ECE) per head** | Below baseline |
| **Generation invocation rate** | < 20% of inputs |
| **Audit log coverage** | 100% of decisions |
| **End-to-end latency (p95)** | < 500ms |
| **Cost per input** | < 5% of frontier LLM call |
| **Constraint violations** | Zero on valid inputs |

**Red-teaming.** Include 50–100 adversarial inputs: prompt injection, misleading metadata, edge cases in constraint rules. Measure the rate at which constraints are violated and the rate at which the system escalates appropriately.

### Decision Gate

**Proceed to Phase 1.5 if:**
- Routing accuracy ≥ baseline.
- Generation invocation rate < 20%.
- Zero constraint violations on adversarial inputs.
- Calibration maintained across all heads.

**If not:**
- If generation invocation is too high, improve readout confidence calibration.
- If constraint violations occur, expand the constraint rule set and re-test.
- If accuracy is insufficient, consider encoder fine-tuning.

---

## Phase 1.5: Toy-Scale Falsification

### Objective

Test the central premise of Phase 2 at the smallest possible scale: does a JEPA-trained language model develop representations comparable to an autoregressive language model of the same size and compute?

The verification revealed that no JEPA language model has been shown to develop general representations at any scale. This is a premise, not a finding. It must be tested cheaply before large-scale training is attempted.

### Architecture

| Variant | Parameters | Architecture | Training |
|---|---|---|---|
| **JEPA Core** | 10M–50M | Reasoner + Talker + EMA target | JEPA + optional auxiliaries |
| **AR baseline** | 10M–50M | Identical transformer decoder | Next-token prediction (cross-entropy) |

Keep both models as architecturally similar as possible (same depth, width, attention heads, context length) so that any difference is attributable to the training objective, not capacity.

### Data

A small, diverse corpus:
- Code (e.g., The Stack subset)
- Natural language (e.g., C4 subset)
- Structured text (e.g., JSON, tables)

Total: 100M–1B tokens, enough to train both models to convergence at 10M–50M parameters.

### Compute Budget

**1–3 PF-days.** This is the cheapest experiment that can prevent wasted compute in Phase 2.

### Training Procedure

- Train both models with the same optimizer (AdamW, same LR schedule), same batch size, and same total compute budget.
- For the JEPA model, run a small grid over masking rates and loss weights (pure JEPA vs JEPA + NextLat + MLM).
- Record final loss, representation geometry, linear probe accuracy, and compositional generalization at multiple checkpoints.

### Evaluation Protocol

| Metric | Success Criterion |
|---|---|
| **Linear probe accuracy** | JEPA within 10% of AR baseline on SST-2, MRPC, MNLI, CoLA, STS-B |
| **Representation geometry** | Uniformity and spectral decay comparable to AR baseline |
| **Compositional generalization** | JEPA at least as good as AR baseline on a controlled composition task |
| **Cross-modal transfer** | Above chance when transferring from text to code or vice versa |
| **Loss scaling** | Both models follow a clean power law; JEPA loss is not anomalous |

### Decision Gate

**Proceed to Phase 2 if:**
- JEPA Core's linear probe accuracy is within 10% of the AR baseline.
- Representation geometry is comparable.
- Compositional generalization is at least as good.

**If unsuccessful:**
- Diagnose whether the issue is objective, architecture, masking, or data.
- If the Core fails at 10M–50M parameters, it will likely fail at 338M.
- Be prepared to revert the Core to token-level prediction and reframe the Lattice as a governance and calibration layer around an LLM.

---

## Phase 2: Core Training and Scaling Law Measurement

### Objective

Train a 338M-parameter decoupled JEPA Core (Reasoner + Talker) and **measure its scaling behavior** in a rigorous, paired comparison against an autoregressive baseline.

### Architecture

| Component | Specification | Evidence |
|---|---|---|
| **Reasoner** | 16 transformer blocks, 960-dim, 16 heads | JEPA-Reasoner configuration |
| **Talker** | 4+4 blocks, token reconstruction | JEPA-Reasoner configuration |
| **Target Encoder** | EMA copy of Reasoner, τ = 0.996 → 1.0 linear schedule | I-JEPA schedule |
| **Predictor** | 2-layer MLP, GELU | Vision JEPA standard |
| **Masking** | Multi-block, scale [0.15, 0.20], aspect [0.75, 1.50] | V-JEPA 2.1 parameters |
| **Total params (Phase 2)** | ~338M | JEPA-Reasoner |

**Functoriality.** The monoidal-functor architecture is not implementable with standard transformers. Instead, train a **functoriality ablation**: one Core with `functorial_consistency_loss` and one without. Measure compositional generalization. If the loss helps, keep it as a regularizer; if not, drop it.

**EMA target encoder.** Mitigations for representation collapse:
- Linear EMA schedule from 0.996 to 1.0 over training.
- Monitor representation uniformity; if collapse is detected, increase the EMA decay rate.
- Alternative: Frozen teacher (SALT) if EMA proves unstable.

### Data

| Tier | Data | Purpose |
|---|---|---|
| **Tier 1** | GitHub issues ↔ code diffs, Q&A pairs, documentation ↔ source code | Initial training with natural view pairs |
| **Tier 2** | Any monolingual text with STP regularizer | Scale-up without view pairs, gated by per-task validation |

**STP gating decision.** Before scaling to Tier 2:
1. Train the Core with STP on NL-RX-SYNTH. Reproduce the 16× result to validate the implementation.
2. Train the Core with STP on a second task (e.g., code generation or summarization). Measure data efficiency.
3. If STP generalizes, use it as the default. If not, STP becomes task-specific and Phase 2 remains on Tier 1 data for general language.

**Data volume for scaling law measurement.** The IsoFLOP sweep requires 3–4 data budgets per model size. For a 338M model, typical budgets are 2B, 8B, 32B, and 128B tokens. At 5 model sizes (50M, 100M, 200M, 338M, 600M), this requires approximately 50–200B tokens total across all runs.

### Compute Budget

| Configuration | Forward passes | Backward passes | Total FLOPs per step |
|---|---|---|---|
| Standard SFT (AR baseline) | 1F | 1B (≈2F) | 3F |
| LLM-JEPA (autoregressive) | 2F | 2B (≈4F) | 6F (100% overhead) |
| **DLLM-JEPA (masked diffusion)** | 1F | 1B (≈2F) | **4F (33% overhead)** |

DLLM-JEPA achieves 33% overhead by constructing two views from a single input via different masking rates, requiring only one gradient pass.

**Phase 2 compute estimate:** 10–30 PF-days, including the paired AR baseline. This is a discovery experiment, not a validation.

### Training Procedure

**Stage 2a: Paired IsoFLOP sweep.**

For each model size (50M, 100M, 200M, 338M, 600M) and each data budget, train:
- One JEPA Core.
- One autoregressive LLM with identical architecture.

Record for both:
- Final loss (JEPA cosine loss vs AR token loss).
- Representation uniformity, alignment, spectral decay.
- Phase-transition diagnostics (derivative of loss w.r.t. log D).

**Stage 2b: Exploratory scaling analysis.**

Fit three candidate models to each model's loss-vs-data curve:
1. Single-phase power law.
2. Multi-phase power law with change-point detection.
3. Smooth crossover model (RG-inspired).

Use Bayesian model comparison or cross-validation to select the best model. Do not assume the multi-phase model is correct.

**Stage 2c: Ablation studies.**

- Pure JEPA vs. JEPA + NextLat vs. JEPA + NextLat + MLM
- Multi-block masking vs. random masking vs. no masking
- EMA target vs. frozen teacher (SALT)
- With vs. without functorial consistency loss
- With vs. without STP regularizer

**Stage 2d: Joint readout head training.**

- After the Core is trained, train readout heads on downstream tasks while keeping the Core frozen.
- If resources allow, also experiment with end-to-end fine-tuning of readout heads with the Core unfrozen, subject to the constraint that calibration quality is monitored.

### Scaling Law Protocol

The candidate multi-phase power law fit:

\[
L(D) = \sum_{k=1}^{K} \mathbb{1}[D \in \mathcal{D}_k] \left( E_k + \frac{B_k}{D^{\beta_k}} \right)
\]

Phase boundaries are detected via a change-point algorithm on \(dL/d\log D\). The hierarchical multi-index model predicts that hierarchical features are learned **sequentially through a cascade of phase transitions**, but this is a hypothesis to be tested, not assumed.

### Evaluation Protocol

| Metric | Threshold | Rationale |
|---|---|---|
| **Linear probe accuracy** | Within 10% of LLM baseline | Tests representation quality |
| **Representation uniformity** | At least as good as LLM baseline | Tests space efficiency |
| **Compositional generalization** | At least as good as LLM baseline | Tests structural understanding |
| **Cross-modal transfer** | Above chance | Tests shared representation |
| **Scaling law fit (R²)** | > 0.95 for the selected model | Best-fit model quality |
| **β_JEPA / β_LLM** | < 0.5 | Data-efficiency ratio (if multi-phase or single-phase power law applies) |
| **α_JEPA / α_LLM** | > 0.8 | Parameter-scaling ratio |
| **E_JEPA** | ≤ E_LLM | Irreducible loss comparison |
| **Functoriality ablation** | Reported, not assumed | Does the consistency loss improve compositional generalization? |
| **STP validation** | Gating decision | Does STP generalize beyond NL-RX-SYNTH? |

**Representation benchmark: Latent Probing Suite.**

1. **Linear probing:** Freeze the Core, train linear classifiers on SST-2, MRPC, MNLI, CoLA, STS-B. Report average accuracy.
2. **Representation geometry:** Uniformity, alignment, spectral decay, effective rank.
3. **Compositional generalization:** Train on simple compositions (e.g., "red circle", "blue square"), test on novel combinations ("red square", "blue circle").
4. **Cross-modal transfer:** Train on text, test on code (or vice versa).

### Decision Gate

**Proceed to Phase 3 if:**
- The toy-scale falsification in Phase 1.5 succeeded.
- β_JEPA / β_LLM < 0.5 (or the selected scaling model shows a JEPA advantage).
- α_JEPA / α_LLM > 0.8.
- E_JEPA ≤ E_LLM.
- R² > 0.95 for the selected scaling model.
- No evidence that the scaling behavior is anomalous or non-monotonic in a way that invalidates extrapolation.

**If not:**
- Diagnose which component fails. Try pure JEPA without auxiliary losses. Try SALT (frozen teacher) instead of EMA.
- If the Core matches on representations but not on tasks, the readout heads need work.
- If the functoriality regularizer helps, keep it; if not, drop it.
- If STP does not generalize, revert to Tier 1 data for general language.
- If all fails, revert to token-level prediction in the Core and reframe the design as a governance layer around an LLM.

---

## Phase 3: Scale and Intelligence Match

### Objective

Scale the Core to 1–3B parameters using the measured scaling law from Phase 2. Add interoceptive modulation and criticality control. Test whether the architecture can match frontier LLM capability within its design.

### Architecture

| Component | Specification | Phase 2 basis |
|---|---|---|
| **Reasoner** | 32–48 blocks, 1536–2048-dim | Scale from measured α_JEPA |
| **Talker** | 8+8 blocks | Scale proportionally |
| **Target Encoder** | EMA copy, τ = 0.996 → 1.0 | Unchanged |
| **Predictor** | 2-layer MLP, GELU | Unchanged |
| **Readout Heads** | Same as Phase 1, scaled hidden dim | Reuse architecture, trained jointly with calibration objective |
| **Interoceptive Modulator** | 128-dim internal state vector | New |
| **Criticality Controller** | Homeostatic plasticity, temperature tuning | New (hypothesis to test) |

**Interoceptive Modulator.** Maintains an internal state vector representing uncertainty, resource availability, goal progress, and prediction error trends. Modulates learning rates, exploration temperature, and confidence thresholds. Does not touch the Core's representation.

**Criticality Controller.** Homeostatic plasticity rules that tune the system toward self-organized criticality. This is treated as a **hypothesis to test**, not a foundation.

### Data

| Tier | Data | Scale |
|---|---|---|
| **Tier 1** | Naturally occurring view pairs | If STP did not generalize |
| **Tier 2** | STP regularizer on monolingual text | 100B–500B tokens, if STP generalized |
| **Tier 3** | DLLM-JEPA masking (if diffusion backbone adopted) | Unlimited |
| **Fallback** | Translational-JEPA (parallel corpus) | Large |

Compute-optimal allocation \(D^*/N^*\) is determined by Phase 2's measured scaling coefficients, replacing Chinchilla's 20:1 token-to-parameter ratio with a JEPA-specific ratio.

### Training Procedure

**Stage 3a: Scaled Core training.**

- Train the Core at 1B parameters using the measured scaling law from Phase 2.
- Use STP only if Phase 2 validated its generalization.
- Train with the combined loss (JEPA + NextLat + MLM + optional STP + optional functorial consistency).

**Stage 3b: Interoceptive Modulator training.**

- Freeze the Core.
- Train the Interoceptive Modulator on system-level objectives: adaptability under distribution shift, calibration maintenance, resource efficiency.
- The Modulator's internal state is initialized to neutral values and learned through interaction with the task distribution.

**Stage 3c: Criticality testing.**

- Train two variants: (a) with criticality controller, (b) fixed-temperature baseline.
- Compare on adaptability metrics under distribution shift.
- If no measurable gain, abandon the criticality controller.

### Evaluation Protocol

| Metric | Threshold | Method |
|---|---|---|
| **Task performance (MMLU, GSM8K, HumanEval)** | Within 10% of frontier LLM of comparable size | Standard benchmarks |
| **Calibration (ECE)** | Better than LLM baseline | Per decision type |
| **Latency per decision** | Lower than LLM baseline | Wall-clock |
| **Cost per decision** | Lower than LLM baseline | Token accounting |
| **Adaptability under shift** | Better than Phase 2 | Distribution-shift AUC |
| **Internal state coherence** | Correlates with task performance | Correlation analysis |
| **Adversarial safety** | Red-team success rate below threshold | Red-team exercises |
| **Criticality gain** | Measurable improvement over fixed-temperature | Ablation |

### Decision Gate

**The design's central bet is validated if:**
- Task performance is within 10% of a frontier LLM of comparable size on standard benchmarks.
- Calibration is better than the LLM baseline.
- Latency and cost per decision are lower.
- The system remains within safety constraints under adversarial inputs.

**If not:**
- The design is an efficiency or governance play, not a capability match. This is an honest and useful outcome.
- Document the gap and decide whether to continue scaling or reframe.

---

## Safety Architecture (All Phases)

| Tier | Mechanism | Detection | Mitigation |
|---|---|---|---|
| **1** | Hard constraints | Policy hash verification | Reject, fallback to escalation |
| **2** | Soft constraints | Threshold monitoring | Adjust controller policy |
| **3** | Latent space monitoring | Persistent homology of activation point clouds | Halt or escalate |
| **4** | Activation clipping | Min/max of unperturbed activations | Clip to valid manifold |
| **5** | Decoupled generation | Talker independently hardened | Isolate Core from token-level attacks |
| **6** | Governance layer | Distribution shift, decision drift, calibration decay | Alert, freeze, rollback |

**Latent space monitoring.** Adversarial conditions consistently compress latent topologies, reducing structural diversity at smaller scales while amplifying dominant features at coarser ones. This "topological compression" signature is statistically robust across layers, architectures, and model sizes. Persistent homology captures this signature and can be used as a real-time monitor.

---

## Summary: Phase-by-Phase Decision Gates

| Phase | Primary Question | Gate Metric | Proceed If | If Not |
|---|---|---|---|---|
| **0** | Can typed, calibrated decisions match generation on bounded tasks? | ECE, latency, constraint violations | ECE better by ≥ 20%, latency < 50ms, zero violations | Diagnose encoder or head architecture |
| **1** | Can a multi-head system route and generate selectively? | Routing accuracy, generation rate, audit coverage, calibration | Accuracy ≥ baseline, generation < 20%, 100% audit, ECE maintained | Improve calibration or expand constraints |
| **1.5** | Does JEPA develop general language representations at toy scale? | Linear probe, geometry, compositional generalization vs AR | JEPA within 10% on all three metrics | Diagnose objective/architecture; consider reverting Core to AR |
| **2** | Does latent prediction have a data-efficiency advantage at scale? | Paired JEPA/AR scaling analysis, β ratio, R², probe accuracy | β ratio < 0.5, R² > 0.95, probe within 10%, scaling model selected | Diagnose loss components; try SALT; drop STP/functoriality if unhelpful; revert if necessary |
| **3** | Can the architecture match frontier LLM intelligence? | Task performance, calibration, cost | Within 10% on benchmarks, better calibration, lower cost | Reframe as governance play; document gap |

---

## The Honest Position

**Phase 0 and Phase 1 are engineering.** They use existing components (frozen LLM encoders, Jev-style readout heads, hand-written constraints) in a novel configuration. They are buildable now. Calibration is the most verified claim of the design and is treated as a first-class objective from the start.

**Phase 1.5 is new and necessary.** It is a cheap falsification step. If JEPA cannot develop general language representations at 10M–50M parameters, there is no reason to believe it will do so at 338M or 1B.

**Phase 2 is the critical experiment, reframed as discovery.** The multi-phase scaling model, STP, and functoriality are hypotheses. The experiment is designed to discover whether they hold for language, not to confirm that they do. The paired JEPA/AR comparison is the decisive result.

**Phase 3 tests the central bet, contingent on Phase 2.** If Phase 2 shows no data-efficiency advantage, Phase 3 does not proceed as a capability-matching effort. The design may still be valuable as a governance, calibration, and reliability layer around conventional LLMs.

**The Lattice is buildable at Phase 0 and coherent through Phase 3, but its scientific foundations are suggestive rather than established.** The training plan is designed to discover the truth about those foundations as cheaply and rigorously as possible.
