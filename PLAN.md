# Detailed Phase Planning: The Lattice

This document expands the phased build plan (Spec §10) into implementation-grade detail. Each phase is specified with concrete architecture, data, compute, training procedure, evaluation protocol, and decision gates. Where the spec made a design choice, the implementation follows it; where the spec left a gap, the implementation specifies the choice and states its rationale.

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

**Calibration.** Post-hoc temperature scaling per decision type. Fit on a held-out calibration set of 100–500 labeled examples per question type. AnyJev reports ECE improving from 0.235 (raw) to 0.100 (L1 calibrated) on a routing task.

### Data

| Dataset | Purpose | Size |
|---|---|---|
| **BANKING77** | Intent routing decision | 13,083 utterances, 77 classes |
| **MultiNLI / WANLI** | Natural language inference decision | ~430K / ~107K |
| **BoolQ** | Boolean decision | ~16K |
| **MMLU-Pro** | Multi-choice decision | ~12K |
| **Synthetic ticket data** | Domain-specific routing (if needed) | 5K–10K |

Split: 70% train (for readout head), 15% calibration (for temperature scaling), 15% test. The calibration set must be **disjoint** from the test set.

### Training Procedure

**Stage 0a: Readout head training.**

- Freeze the encoder.
- Train the readout head with cross-entropy on option-letter logits.
- 2 epochs, learning rate 3e-5, AdamW.
- Batch size 32, max sequence length 512.

**Stage 0b: Temperature fitting.**

- Compute logits on the calibration set.
- Fit a single scalar temperature per decision type by NLL minimization (LBFGS, 200 iterations).
- Temperature is stored as a small artifact alongside the model checkpoint.

**Stage 0c: Constraint layer integration.**

- Load the constraint policy; compute and record its hash.
- Every decision call verifies the hash before evaluation.
- Constraint violations trigger a deterministic fallback (escalation).

### Evaluation Protocol

| Metric | Threshold | Method |
|---|---|---|
| **ECE** | Below LLM baseline | Expected calibration error over 15 bins |
| **Accuracy** | ≥ baseline | Argmax accuracy on test set |
| **Brier score** | Below LLM baseline | Multiclass Brier |
| **Order-flip rate** | < 5% | Cyclic-shift marginalization removes option-order bias |
| **Constraint violations** | Zero | Count of constraint-layer rejections on valid inputs |
| **Latency (p50)** | < 50ms on CPU | Wall-clock per decision |
| **Latency (p95)** | < 200ms | Including calibration and constraint evaluation |
| **Cost per decision** | < 1% of frontier LLM call | Token accounting |

**Generative baseline.** A frontier LLM (GPT-4-class or Claude-class) prompted with the same decision task, parsed with constrained decoding (JSON schema or regex). Measure the same metrics.

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

Build a deployable document triage system with multiple decision heads, a deterministic controller, selective generation, and full audit logging.

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

The synthetic corpus is generated by a frontier LLM with a structured prompt template covering 15–20 document types (invoices, contracts, support tickets, policy documents, etc.) and hard decision cases (ambiguity, misleading metadata, injected instructions, rule precedence).

### Training Procedure

**Stage 1a: Multi-head readout training.**

- Freeze encoder.
- Train all readout heads jointly with per-head cross-entropy loss.
- Weight the loss inversely proportional to class frequency.
- 3 epochs, learning rate 3e-5.
- Joint training encourages the heads to share the encoder's representation effectively.

**Stage 1b: Controller threshold tuning.**

- Freeze encoder and heads.
- Tune controller thresholds on the validation set to optimize a composite objective: accuracy × coverage − cost × generation_rate.
- Use grid search or Bayesian optimization over 3–4 thresholds.

**Stage 1c: Calibration per head.**

- Fit temperature per head on the calibration set.
- Store temperatures as metadata.

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

**Proceed to Phase 2 if:**
- Routing accuracy ≥ baseline.
- Generation invocation rate < 20%.
- Zero constraint violations on adversarial inputs.
- Calibration maintained across all heads.

**If not:**
- If generation invocation is too high, improve readout confidence calibration.
- If constraint violations occur, expand the constraint rule set and re-test.
- If accuracy is insufficient, consider encoder fine-tuning (this would blur the Phase 0/2 boundary).

---

## Phase 2: Core Training and Scaling Law Measurement

### Objective

Train a 338M-parameter decoupled JEPA Core (Reasoner + Talker) and **measure its scaling behavior** to determine whether latent prediction has a structural data-efficiency advantage over next-token prediction.

### Architecture

| Component | Specification | Evidence |
|---|---|---|
| **Reasoner** | 16 transformer blocks, 960-dim, 16 heads | JEPA-Reasoner configuration |
| **Talker** | 4+4 blocks, token reconstruction | JEPA-Reasoner configuration |
| **Target Encoder** | EMA copy of Reasoner, τ = 0.996 → 1.0 linear schedule | I-JEPA schedule |
| **Predictor** | 2-layer MLP, GELU | Vision JEPA standard |
| **Masking** | Multi-block, scale [0.15, 0.20], aspect [0.75, 1.50] | V-JEPA 2.1 parameters |
| **Total params (Phase 2)** | ~338M | JEPA-Reasoner |

**EMA target encoder.** The EMA target encoder prevents representation collapse but can also cause early collapse if the target encoder stays too close to the main encoder. Mitigations:
- Linear EMA schedule from 0.996 to 1.0 over training.
- Monitor representation uniformity; if collapse is detected, increase the EMA decay rate.
- Alternative: Frozen teacher (SALT) if EMA proves unstable.

### Data

| Tier | Data | Purpose |
|---|---|---|
| **Tier 1** | GitHub issues ↔ code diffs, Q&A pairs, documentation ↔ source code | Initial training with natural view pairs |
| **Tier 2** | Any monolingual text with STP regularizer | Scale-up without view pairs |

**Data volume for scaling law measurement.** The IsoFLOP sweep requires 3–4 data budgets per model size. For a 338M model, typical budgets are 2B, 8B, 32B, and 128B tokens. At 5 model sizes (50M, 100M, 200M, 338M, 600M), this requires approximately 50–200B tokens total across all runs.

### Compute Budget

| Configuration | Forward passes | Backward passes | Total FLOPs per step |
|---|---|---|---|
| Standard SFT (AR baseline) | 1F | 1B (≈2F) | 3F |
| LLM-JEPA (autoregressive) | 2F | 2B (≈4F) | 6F (100% overhead) |
| **DLLM-JEPA (masked diffusion)** | 1F | 1B (≈2F) | **4F (33% overhead)** |

DLLM-JEPA achieves 33% overhead by constructing two views from a single input via different masking rates, requiring only one gradient pass. This is the most compute-efficient JEPA configuration reported.

**Phase 2 compute estimate:** 5–15 PF-days, treated as a pilot measurement experiment.

### Training Procedure

**Stage 2a: IsoFLOP sweep.**

Train the Core at 5 model sizes (50M, 100M, 200M, 338M, 600M) with 3–4 data budgets each, at fixed compute. For each run, record:
- Final loss (JEPA cosine loss)
- Representation uniformity, alignment, spectral decay
- Phase-transition diagnostics (derivative of loss w.r.t. log D)

**Stage 2b: Baseline comparison.**

Train an autoregressive LLM of identical architecture and size on the same data. Fit the same power law. Compare coefficients.

**Stage 2c: Ablation studies.**

- Pure JEPA vs. JEPA + NextLat vs. JEPA + NextLat + MLM
- Multi-block masking vs. random masking vs. no masking
- EMA target vs. frozen teacher (SALT)

### Scaling Law Protocol

The multi-phase power law fit:

\[
L(D) = \sum_{k=1}^{K} \mathbb{1}[D \in \mathcal{D}_k] \left( E_k + \frac{B_k}{D^{\beta_k}} \right)
\]

Phase boundaries are detected via a change-point algorithm on \(dL/d\log D\). The hierarchical multi-index model predicts that hierarchical features are learned **sequentially through a cascade of phase transitions**. Each phase transition corresponds to the Core learning a new level of the compositional hierarchy.

### Evaluation Protocol

| Metric | Threshold | Rationale |
|---|---|---|
| **Linear probe accuracy** | Within 10% of LLM baseline | Tests representation quality |
| **Representation uniformity** | At least as good as LLM baseline | Tests space efficiency |
| **Compositional generalization** | At least as good as LLM baseline | Tests structural understanding |
| **Cross-modal transfer** | Above chance | Tests shared representation |
| **Scaling law fit (R²)** | > 0.95 | Multi-phase model fit quality |
| **β_JEPA / β_LLM** | < 0.5 | Data-efficiency ratio |
| **α_JEPA / α_LLM** | > 0.8 | Parameter-scaling ratio |
| **E_JEPA** | ≤ E_LLM | Irreducible loss comparison |

**Representation benchmark: Latent Probing Suite.**

1. **Linear probing:** Freeze the Core, train linear classifiers on SST-2, MRPC, MNLI, CoLA, STS-B. Report average accuracy.
2. **Representation geometry:** Uniformity (average pairwise cosine similarity), alignment (cosine similarity of paraphrases), spectral decay (ratio of first to last singular value), effective rank (dims for 90% variance).
3. **Compositional generalization:** Train on simple compositions (e.g., "red circle", "blue square"), test on novel combinations ("red square", "blue circle").
4. **Cross-modal transfer:** Train on text, test on code (or vice versa).

### Decision Gate

**Proceed to Phase 3 if:**
- β_JEPA / β_LLM < 0.5 (structural data-efficiency advantage).
- α_JEPA / α_LLM > 0.8 (parameter scaling is comparable).
- E_JEPA ≤ E_LLM (no irreducible loss penalty).
- R² > 0.95 for the multi-phase power-law fit.

**If not:**
- Diagnose which component fails. Try pure JEPA without auxiliary losses. Try SALT (frozen teacher) instead of EMA.
- If the Core matches on representations but not on tasks, the readout heads need work.
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
| **Readout Heads** | Same as Phase 1, scaled hidden dim | Reuse architecture |
| **Interoceptive Modulator** | 128-dim internal state vector | New |
| **Criticality Controller** | Homeostatic plasticity, temperature tuning | New (hypothesis to test) |

**Interoceptive Modulator.** Maintains an internal state vector representing uncertainty, resource availability, goal progress, and prediction error trends. Modulates learning rates, exploration temperature, and confidence thresholds. Does not touch the Core's representation.

**Criticality Controller.** Homeostatic plasticity rules that tune the system toward self-organized criticality. The critical brain hypothesis proposes that biological neural networks operate near a phase transition between order and chaos to optimize information processing. This is treated as a **hypothesis to test**, not a foundation.

### Data

| Tier | Data | Scale |
|---|---|---|
| **Tier 2** | STP regularizer on monolingual text | 100B–500B tokens |
| **Tier 3** | DLLM-JEPA masking (if diffusion backbone adopted) | Unlimited |
| **Fallback** | Translational-JEPA (parallel corpus) | Large |

### Compute Budget

| Phase | Model Scale | Estimated Compute |
|---|---|---|
| Phase 3 (pilot) | 1B params | 50–200 PF-days |
| Phase 3 (scale) | 3B params | 200–800 PF-days |

Compute-optimal allocation \(D^*/N^*\) is determined by Phase 2's measured scaling coefficients, replacing Chinchilla's 20:1 token-to-parameter ratio with a JEPA-specific ratio.

### Training Procedure

**Stage 3a: Scaled Core training.**

- Train the Core at 1B parameters using the measured scaling law from Phase 2.
- Use STP regularizer to eliminate view-pair requirement.
- Train with the combined loss (JEPA + NextLat + MLM + STP).

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
| **0** | Can typed decisions match generation on bounded tasks? | ECE, latency, constraint violations | ECE better, latency < 50ms, zero violations | Diagnose encoder or head architecture |
| **1** | Can a multi-head system route and generate selectively? | Routing accuracy, generation rate, audit coverage | Accuracy ≥ baseline, generation < 20%, 100% audit | Improve calibration or expand constraints |
| **2** | Does latent prediction have a data-efficiency advantage? | β_JEPA / β_LLM, R², linear probe accuracy | β ratio < 0.5, R² > 0.95, probe within 10% | Diagnose loss components; try SALT; revert if necessary |
| **3** | Can the architecture match frontier LLM intelligence? | Task performance, calibration, cost | Within 10% on benchmarks, better calibration, lower cost | Reframe as governance play; document gap |

---

## The Honest Position

**Phase 0 and Phase 1 are engineering.** They use existing components (frozen LLM encoders, Jev-style readout heads, hand-written constraints) in a novel configuration. They are buildable now.

**Phase 2 is the critical experiment.** It asks whether latent prediction has a structural data-efficiency advantage over next-token prediction. The theoretical result (constant sample complexity) and empirical evidence (STP's 16× data efficiency) suggest the answer is yes, but the IsoFLOP sweep is designed to produce a clear answer, not to confirm a hypothesis.

**Phase 3 tests the central bet.** If Phase 2 validates the scaling advantage, Phase 3 asks whether the architecture can match frontier LLM capability. If not, the design remains valuable as a governance and reliability layer around conventional LLMs.

**The spec is buildable at Phase 0 and coherent through Phase 3.** Phase 0 requires no novel research. Phase 1 adds engineering. Phase 2 is the decisive experiment. Phase 3 is contingent on Phase 2's outcome.
