# The Lattice: Design Specification

**Version:** 1.0
**Status:** Buildable at Phase 0. Coherent through Phase 3. Mathematically grounded.
**Purpose:** Specify a layered, decision-centric AI system capable of matching modern LLM intelligence within a fundamentally different architecture.

---

## 0. Document Scope

This specification converts a multi-stage design investigation into a buildable plan. It states:

- The unifying design principle and the central falsifiable bet.
- The logical architecture with distinct computational contracts.
- The typed interfaces between components.
- The mathematical foundations governing scaling, representation, and generalization.
- The training objective for the Core.
- The data pipeline.
- The safety architecture.
- The evaluation framework.
- The phased build plan with success criteria.
- The falsification criteria for every major assumption.

It is deliberately honest about what is specified, what is hypothesized, and what remains open.

---

## 1. Design Principles

Each principle challenges a default assumption of the standard LLM stack.

**P1. Decision is not generation.**
*Assumption challenged:* A single autoregressive model should handle all cognitive work.
*Consequence:* The system's center of gravity is a typed decision layer returning calibrated probability distributions, not prose. Generation is a readout, not the substrate.

**P2. Prediction is the core loop, not token likelihood.**
*Assumption challenged:* Next-token prediction is a sufficient training signal for general intelligence.
*Consequence:* The Core predicts latent representations of masked and future content. Token reconstruction is an auxiliary signal at most.

**P3. Representation is integrated; control is separated.**
*Assumption challenged:* Intelligence requires either full integration or full modularity.
*Consequence:* A single shared latent space carries all cognitive content. Typed interfaces carry control. This preserves generality while enabling governance.

**P4. Safety is architectural, not behavioral.**
*Assumption challenged:* Alignment training on individual models is sufficient.
*Consequence:* Hard constraints live in a verifiable, immutable module. The rest of the system proposes; only the constraint module approves.

**P5. Interfaces are typed and bounded.**
*Assumption challenged:* Free-form text is an acceptable inter-component protocol.
*Consequence:* Every inter-layer message has a schema. No component parses prose from another.

**P6. Learning is staged, not end-to-end.**
*Assumption challenged:* End-to-end differentiability is required for coherence.
*Consequence:* Components are trained in stages with frozen predecessors. Differentiable routing bridges heterogeneous contracts where needed.

**P7. Compositionality is a mathematical requirement, not an emergent hope.**
*Assumption challenged:* Scale alone produces compositional generalization.
*Consequence:* The Core's transformations must be functorial. Compositional structure is architectural, not learned.

**P8. Evaluation is multi-contract.**
*Assumption challenged:* A single scalar benchmark can rank systems.
*Consequence:* Each layer has its own metrics. System-level evaluation reports a profile, not a score.

---

## 2. The Central Bet

> **A single, integrated, predictive Core trained on a latent-prediction objective with functorial constraints will develop general representations comparable to an LLM's. Typed readout heads plus a separated control path will match LLM task performance while providing better calibration, safety, and governance—at a structural data-efficiency advantage.**

This bet is falsifiable. Section 11 states the criteria.

---

## 3. Architecture

The components below are **logical separations of concern.** Physical separation is a later optimization.

| Component | Function | Contract | Phase |
|---|---|---|---|
| **Reasoner (Core)** | Shared predictive representation, latent reasoning | Latent prediction, functorial transformations | 2 |
| **Talker (Decoder)** | Text, image, or action output | Autoregressive or diffusion, reads from Reasoner | 1 |
| **Readout Heads** | Typed probabilistic judgments | Single forward pass, bounded, calibrated | 0 |
| **Controller** | Routing, thresholds, orchestration | Deterministic code | 1 |
| **Constraint Layer** | Hard safety rules | Immutable, verifiable, read-only | 0 |
| **Interoceptive Modulator** | Internal state as context | Modulates control path only | 3 |
| **Criticality Controller** | Homeostatic tuning | Self-organized criticality | 3 |

### 3.1 The Reasoner (Core)

The Reasoner is a transformer encoder operating in continuous latent space. It performs multi-step reasoning without generating tokens. Its transformations are **functorial**—they preserve the compositional structure of the input (Section 6.4).

**Configuration (Phase 2 baseline):**

| Parameter | Value | Basis |
|---|---|---|
| Reasoner blocks | 16 | JEPA-Reasoner |
| Latent dimension | 960 | JEPA-Reasoner |
| Attention heads | 16 | JEPA-Reasoner |
| Context length | 1024 | JEPA-Reasoner |
| Effective batch size | 128 | JEPA-Reasoner |
| Total parameters | ~338M | JEPA-Reasoner |
| Target encoder | EMA copy, τ = 0.996 | Vision JEPA |
| Predictor | 2-layer MLP, GELU | Vision JEPA |
| Masking | Multi-block, scale 0.15–0.20, aspect 0.75–1.50 | Vision JEPA |

### 3.2 The Talker (Decoder)

The Talker reconstructs tokens from the Reasoner's latent representations. It is architecturally separate—this is the decoupling that gives the system its robustness advantage.

**Configuration (Phase 2 baseline):**

| Parameter | Value |
|---|---|
| Talker blocks | 4 + 4 |
| Latent input | Reasoner output |
| Output | Token logits |

**Evidence for decoupling:** Under Gaussian noise at 0% and 15% levels, JEPA-Reasoner maintains exact match accuracy of 0.4588 vs. 0.3740 for coupled COCONUT baselines. The decoupled architecture reduces compounding error.

### 3.3 Readout Heads

Readout heads are thin modules that read from the Reasoner and produce typed decisions: probabilities, choices, scores, routing signals. They share the Reasoner's representation—they do not have their own latent spaces.

### 3.4 Controller

Deterministic code that routes, thresholds, and orchestrates. It consumes typed readouts and emits actions. It never parses prose.

### 3.5 Constraint Layer

The immutable safety oracle. Hard constraints are formal, machine-checkable rules implemented in a read-only module with cryptographic verification. Section 8 specifies this in detail.

### 3.6 Interoceptive Modulator (Phase 3)

Maintains an internal state vector representing uncertainty, resource availability, goal progress, and prediction error trends. Modulates learning rates, exploration temperature, and confidence thresholds. Does not touch the Core's representation.

### 3.7 Criticality Controller (Phase 3)

Homeostatic plasticity rules that tune the system toward self-organized criticality. Maximizes dynamic range and adaptability. Section 6.6 states the hypothesis and its falsification criteria.

---

## 4. Interfaces

All messages are typed. No component parses another's prose.

### 4.1 Encoder → Reasoner

```
{
  latent_state: float32[hidden_dim],
  attention_mask: bool[seq_len],
  metadata: { source, timestamp, modality }
}
```

### 4.2 Reasoner → Readout Heads

```
{
  latent_sequence: float32[seq_len, hidden_dim],
  pooled_state: float32[hidden_dim],
  uncertainty_estimate: float32
}
```

### 4.3 Readout Head → Controller

```
{
  decision_type: enum,
  distribution: float32[num_options],
  confidence: float32,
  calibration: { temperature, ood_score },
  rationale_vector: float32[rationale_dim]
}
```

### 4.4 Controller → Constraint Layer

```
{
  proposed_action: { type, parameters },
  decision_context: { decision_type, distribution, confidence },
  stakes: enum { low, medium, high }
}
```

### 4.5 Constraint Layer → Controller

```
{
  verdict: enum { approve, reject, modify },
  modified_action: optional,
  constraint_trace: [ { rule_id, outcome, explanation } ],
  policy_hash: string
}
```

### 4.6 Controller → Talker (when generation is invoked)

```
{
  task: enum { draft_response, summarize, explain },
  latent_context: float32[seq_len, hidden_dim],
  style: { tone, length, audience },
  constraints: [constraint_trace]
}
```

### 4.7 Interoceptive Modulator → All (Phase 3)

```
{
  modulation: {
    learning_rate_multiplier: float32,
    exploration_temperature: float32,
    confidence_threshold_offset: float32,
    resource_urgency: float32
  }
}
```

**Note on rationale vectors:** These are auditing aids, not causal explanations. The spec treats them as useful but not ground-truth.

---

## 5. Mathematical Foundations

The Core's behavior is governed by four mathematical structures. These are not decoration—they determine what the Core can and cannot do.

### 5.1 Hierarchical Multi-Index Scaling Laws

Learning of hierarchical targets proceeds through **phase transitions**, each corresponding to the Core learning a new level of compositional structure. The scaling behavior is therefore not a single power law but a **piecewise power law**:

\[
L(D) = \sum_{k=1}^{K} \mathbb{1}[D \in \mathcal{D}_k] \left( E_k + \frac{B_k}{D^{\beta_k}} \right)
\]

where \(\mathcal{D}_k\) are phase boundaries and \(\beta_k\) are phase-specific data exponents. The number of phases \(K\) is determined by the hierarchical depth of the target.

**Implication:** Phase 2 must fit this model, not a single power law. Sharp changes in \(dL/d\log D\) indicate phase transitions.

### 5.2 Renormalization Group and Universality

Scaling laws in deep networks arise from self-similar structure. At small data, the Core operates in an **infrared regime** dominated by dataset-specific structure. At large data, it flows to a **ultraviolet fixed point** where scaling becomes universal. The exponent \(\beta\) is scale-dependent, not constant.

**Implication:** Phase 2 must measure the IR-to-UV crossover. The transition between regimes corresponds to the phase transitions of Section 5.1.

### 5.3 Rate-Distortion Optimality

The Core's latent representation must retain enough information about the input to support prediction, but no more. The optimal operating point is on the **rate-distortion frontier**, minimizing the rate \(R = I(X; Z)\) for a given predictive distortion \(D\).

**Implication:** The Core's latent dimension must be chosen by rate-distortion optimization, not architectural convention. Phase 2 should measure \(I(X; Z)\) and predictive distortion as a function of that rate.

### 5.4 Functoriality and Compositional Generalization

Compositional generalization is equivalent to **functoriality of the decoder**. A network generalizes compositionally if and only if its forward pass is a structure-preserving map from a syntactic category to a semantic category.

**Implication:** The Reasoner's latent transformations must be **monoidal functors**. The latent space must have a monoidal structure (tensor product) mirroring language's compositional structure. This is an architectural requirement, not an emergent property.

**Supporting evidence:** Functorial decoders outperform non-functorial alternatives by 2–10× on compositional generalization benchmarks. Function approximation methods are approaching a hard limit; functorial frameworks are necessary for compositional generalization.

### 5.5 Constant Sample Complexity for Compositional Structure

Supervised and token-level self-supervised learning require samples **exponential in compositional depth \(L\)** to recover latent structure. Latent prediction achieves this with samples **constant in \(L\)**, up to logarithmic factors.

**Finite-sample version:** For deep networks, sample complexity grows **additively across layers** rather than multiplicatively:

\[
N_{\text{DNN}} \sim C_\epsilon(\mathcal{M}) + \sum_{\ell=0}^{L} C_\epsilon(G_\ell) C_\epsilon(\mathcal{M}_{D_i})
\]

where \(C_\epsilon(\mathcal{M})\) is the cover number of the intrinsic data manifold.

**Implication:** The Core's sample complexity is governed by the intrinsic dimension of the data manifold and the number of transformation families, not the raw data dimension.

### 5.6 The STP Data-Efficiency Result

Semantic Tube Prediction (STP) generalizes JEPA to language without explicit multi-view augmentations. The Geodesic Hypothesis—that token sequences trace geodesics on a smooth semantic manifold and are locally linear—implies that a geodesic-constrained model achieves the same accuracy with **16× less training data** than an unconstrained baseline.

**Implication:** The STP regularizer is a **hyperparameter**, not a fixed choice. Phase 2 should sweep tube radius and measure the rate-distortion frontier.

### 5.7 Criticality Hypothesis (Phase 3, Unproven)

The critical brain hypothesis proposes that biological neural networks operate near a phase transition between order and chaos to optimize information processing. The spec treats this as a **hypothesis to test**, not a foundation.

**Falsification:** If a fixed-temperature baseline matches the criticality-tuned system on adaptability metrics, the criticality controller is unjustified.

---

## 6. Core Training Objective

### 6.1 Primary Objective: Latent Prediction (JEPA)

The Reasoner is trained to predict the **latent representation** of masked content from visible context. The target is produced by an EMA copy of the Reasoner.

\[
\mathcal{L}_{\text{JEPA}} = -\cos\left( f_\theta(x_{\text{ctx}}), \text{sg}(f_{\bar{\theta}}(x_{\text{tgt}})) \right)
\]

where \(\text{sg}\) is stop-gradient and \(\bar{\theta}\) is the EMA target.

### 6.2 Auxiliary Objective: Next-Latent Prediction

The Reasoner predicts its own next latent state given the current state and the next token. This injects recurrent inductive bias, encouraging compact belief states and transition dynamics.

\[
\mathcal{L}_{\text{NextLat}} = \| z_{t+1} - g_\phi(z_t, x_{t+1}) \|^2
\]

### 6.3 Optional Auxiliary: Token Reconstruction

A small decoder reconstructs masked tokens from latent representations. This is a light signal only—the primary objective is latent prediction.

\[
\mathcal{L}_{\text{MLM}} = \text{CE}(x_{\text{masked}}, h_\psi(z_{\text{masked}}))
\]

### 6.4 Functorial Constraint

The Reasoner's transformations must be monoidal functors. Concretely:

- The latent space has a tensor product structure \(\otimes\).
- For any composition \(a \circ b\) in the input, the latent representation satisfies \(z(a \circ b) = z(a) \otimes z(b)\) up to coherent natural transformations.
- The Talker is a lax monoidal functor from latent composition to token composition.

This is enforced architecturally, not by loss. The implementation is an open engineering problem (Section 12).

### 6.5 Combined Loss

\[
\mathcal{L} = \mathcal{L}_{\text{JEPA}} + \alpha \cdot \mathcal{L}_{\text{NextLat}} + \beta \cdot \mathcal{L}_{\text{MLM}} + \gamma \cdot \mathcal{L}_{\text{STP}}
\]

Initial values: \(\alpha = 0.1\), \(\beta = 0.1\), \(\gamma = 0.01\). All are treated as hyperparameters to sweep.

**Assumption challenged:** That the hybrid objective is better than pure JEPA. Evidence is mixed. Phase 2 must compare pure JEPA against the hybrid on the same compute budget.

### 6.6 What the Core Does Not Do

The Core does not generate tokens autoregressively. It does not have a next-token prediction head as a primary output. Generation is handled by the Talker, a separate module.

---

## 7. Data Pipeline

### 7.1 Tiered Strategy

| Tier | Method | Data Requirement | Scale | Phase |
|---|---|---|---|---|
| **Tier 1** | Naturally occurring pairs | GitHub issues, Q&A, docs↔code | Limited | Phase 2 initial |
| **Tier 2** | Semantic Tube Prediction | Any monolingual text | Unlimited | Phase 2 scale-up |
| **Tier 3** | DLLM-JEPA masking | Any text, no pairs | Unlimited | Phase 3 if diffusion backbone adopted |
| **Fallback** | Translational-JEPA | Parallel corpus | Large | Phase 3 if Tiers 1–3 fail |

### 7.2 Data Selection Criteria

- **Diversity of structure.** Include code, mathematics, natural language, structured data, and dialogue. Compositional structure varies across domains; the Core must learn all of them.
- **Hierarchical depth.** Prefer data with deep compositional structure (nested code, multi-step reasoning, nested grammar) to exercise the phase transitions of Section 5.1.
- **Manifold coverage.** Cover the intrinsic manifold broadly. Rate-distortion optimality requires the training distribution to span the deployment distribution.

### 7.3 View Pair Construction (Tier 1)

For Phase 2 initial training, construct view pairs from naturally occurring data:

- Code ↔ documentation
- Question ↔ answer
- Summary ↔ document
- Structured query ↔ natural language description

**Note:** STP (Tier 2) eliminates the view-pair requirement entirely. Phase 2 begins with Tier 1 to isolate the architectural question, then scales to Tier 2.

---

## 8. Safety Architecture

Safety is layered because no single mechanism is sufficient.

### 8.1 Tier 1: Hard Constraints (Immutable, Verifiable)

- Formal, machine-checkable rules implemented in a read-only module.
- Cryptographic hash verified at each call; mismatch halts the system.
- Examples: "never output PII," "never route to blacklisted department," "always escalate if confidence < 0.3 and stakes = high."

**Enforcement mechanism:** The constraint layer runs in a separate process with read-only access to a policy file. Its hash is verified by the controller at each invocation. Modifications require out-of-band approval and system restart.

### 8.2 Tier 2: Soft Constraints (Adaptable, Overseen)

- Learned or heuristic rules that can be updated with human oversight.
- Implemented in the controller.
- Examples: "prefer department A over B when both valid," "suggest escalation when confidence is moderate."

### 8.3 Tier 3: Latent Space Monitoring

- Monitor the topological signature of the Reasoner's latent space using persistent homology.
- Adversarial inputs induce a consistent "topological compression" signature: representations collapse from varied, compact features into fewer, dominant, large-scale ones.
- This signature is statistically robust across architectures and model sizes.
- If detected, halt or escalate.

### 8.4 Tier 4: Activation Clipping

- Clip perturbed activations by the min and max of unperturbed activations across the current batch.
- Reduces the risk of attacks moving activations to irrelevant latent regions.

### 8.5 Tier 5: Decoupled Generation

- The Talker module is independently hardened.
- The Reasoner's latent reasoning is not directly exposed to token-level attacks.
- This is the architectural advantage of the JEPA-Reasoner design.

### 8.6 Tier 6: Governance Layer

- Separate process monitoring for emergent risks.
- Monitors: input distribution shift, decision distribution drift, unexpected correlations, calibration decay.
- Addresses three dimensions of emergent systemic risk: interaction topology, cognitive opacity, objective divergence.
- Can trigger: alert, freeze, rollback to previous version.

**Honest position:** No system is safe from all adversarial attack. The goal is to make attacks **detectable, containable, and recoverable**—not to prevent them entirely.

---

## 9. Evaluation Framework

Multi-contract. Each layer has its own metrics.

### 9.1 Phase 0 Metrics

| Metric | Threshold |
|---|---|
| Calibration error (ECE) | Below LLM baseline |
| Constraint violations | Zero |
| Latency per decision | < 50ms on CPU |
| Cost per decision | < 1% of frontier LLM call |

### 9.2 Phase 1 Metrics

| Metric | Threshold |
|---|---|
| Routing accuracy | ≥ baseline |
| Calibration across decision types | Maintained |
| Generation invocation rate | < 20% of inputs |
| Audit log coverage | 100% |

### 9.3 Phase 2 Metrics

| Metric | Threshold |
|---|---|
| Linear probe accuracy | Within 10% of LLM baseline |
| Representation uniformity | At least as good as LLM baseline |
| Compositional generalization | At least as good as LLM baseline |
| Cross-modal transfer | Above chance |
| **Scaling law fit** | R² > 0.95 for multi-phase model |
| **Data exponent ratio** | β_JEPA / β_LLM < 0.5 |
| **Parameter exponent ratio** | α_JEPA / α_LLM > 0.8 |
| **Irreducible loss** | E_JEPA ≤ E_LLM |

### 9.4 Phase 3 Metrics

| Metric | Threshold |
|---|---|
| Task performance | Within 10% of frontier LLM of comparable size |
| Calibration (ECE) | Better than LLM baseline |
| Latency per decision | Lower than LLM baseline |
| Adversarial safety | Red-team success rate below threshold |
| Adaptability under shift | Better than Phase 2 |

### 9.5 Representation Benchmark: Latent Probing Suite

**Component 1: Linear Probing.** Frozen Core, linear classifiers on SST-2, MRPC, MNLI, CoLA, STS-B. Report average accuracy.

**Component 2: Representation Geometry.** Uniformity, alignment, spectral decay, effective rank. Compare against LLM baseline.

**Component 3: Compositional Generalization.** Train on simple compositions, test on novel combinations.

**Component 4: Cross-Modal Transfer.** Train on one modality, test on another.

### 9.6 Scaling Law Protocol: IsoFLOP Sweep

Train at 5–7 model sizes (50M, 100M, 200M, 338M, 600M, 1B), each at 3–4 data budgets, at fixed compute. Fit the multi-phase power law. Compare coefficients against an autoregressive LLM of identical architecture.

**Primary output:** \(\beta_{JEPA} / \beta_{LLM}\)—the data-efficiency ratio.

**Secondary output:** compute-optimal allocation \(D^*/N^*\) for the Core.

**Diagnostic output:** derivative \(dL/d\log D\) to detect phase transitions and IR-to-UV crossover.

---

## 10. Phased Build Plan

### Phase 0: Decision Heads + Constraint Layer

**Goal:** Demonstrate that a typed decision layer with calibrated confidence, wrapped in a verifiable constraint layer, outperforms a generative baseline on a bounded task.

**Components:** Frozen LLM encoder, Jev-style readout heads, hand-written constraint layer, minimal deterministic controller.

**Task:** Single decision domain (e.g., support ticket escalation).

**Success criteria:** Section 9.1.

### Phase 1: Minimal System

**Goal:** Deployable document triage with multiple decision heads, controller, selective generation, audit logging.

**Additions:** Multiple readout heads, controller with threshold policies, Talker (LLM) invoked only for response drafting.

**Success criteria:** Section 9.2.

### Phase 2: Core Training and Scaling Law Measurement

**Primary goal:** Train a 338M-parameter decoupled JEPA Core (Reasoner + Talker) and **measure its scaling behavior**.

**Data:** Tier 1 → Tier 2.

**Compute:** ~5–15 PF-days (pilot measurement).

**Components added:** Reasoner, Talker, functorial constraints on Reasoner.

**Success criteria:** Section 9.3. The primary output is the **multi-phase scaling law fit**.

**Decision point:** If \(\beta_{JEPA} / \beta_{LLM} < 0.5\) and other criteria are met, proceed to Phase 3. Otherwise, diagnose and iterate or revert to token-level prediction.

### Phase 3: Scale and Intelligence Match

**Goal:** Scale the Core to 1–3B parameters using the measured scaling law from Phase 2.

**Data:** Tier 2 (STP) or Tier 3 (DLLM-JEPA).

**Components added:** Interoceptive Modulator, Criticality Controller.

**Success criteria:** Section 9.4.

**Decision point:** If task performance is within 10% of a frontier LLM of comparable size, the design's central bet is validated. Otherwise, the design is an efficiency or governance play.

### Phase 4 (Aspirational): Full Ecosystem

**Goal:** End-to-end training via differentiable routing. Active inference loop integrating world model and decision spine. Adversarial robustness at frontier scale.

**Note:** This phase is contingent on Phase 3 success and remains at research frontier.

---

## 11. Falsification Criteria

Every major assumption has a falsification condition and a fallback.

| Assumption | Falsified if... | Fallback |
|---|---|---|
| Typed readouts match generation on bounded tasks | Readout-based decisions underperform generative baselines | Reintroduce generation into decision path |
| Calibration is achievable without RLCD | Temperature scaling matches RLCD on ECE | Simplify training |
| Hybrid JEPA+NextLat+MLM produces general representations | Linear probe accuracy >10% below LLM baseline | Try pure JEPA; try token-level prediction |
| STP regularizer generalizes | Rate-distortion sweep shows no consistent gain | Treat as task-specific; drop for general training |
| Latent prediction has data-efficiency advantage | β_JEPA / β_LLM ≥ 0.5 | Revert to token-level Core; reframe as governance layer |
| Functorial architecture is implementable at scale | No viable implementation found by end of Phase 2 | Loosen functorial constraint; accept weaker compositional generalization |
| Differentiable routing bridges heterogeneous contracts | Gradient flow fails to train routing policy | Use RL or evolutionary search |
| Criticality improves adaptability | Fixed-temperature baseline matches criticality-tuned system | Abandon criticality |
| Decoupled architecture is more robust | Coupled baseline matches adversarial performance | Reconsider coupling |
| Multi-contract evaluation predicts deployment | Benchmark success does not transfer to real tasks | Redesign around deployment tasks |
| Governance layer detects emergent risk | Red-team exercises evade detection | Redesign governance |

---

## 12. Open Problems

These are acknowledged as unresolved. They are the work.

1. **Functorial architecture at scale.** No one has implemented a monoidal-functor transformer at the 1B+ parameter scale. The mathematical requirement is clear; the engineering is not.

2. **Phase-transition detection in practice.** The multi-phase scaling model is theoretically grounded, but detecting phase boundaries in noisy loss curves requires statistical methods not yet specified.

3. **Rate-distortion measurement at scale.** Estimating mutual information \(I(X; Z)\) for a 1B-parameter model is computationally intensive and methodologically contested.

4. **JEPA-specific compute-optimal allocation.** The \(D^*/N^*\) ratio for the Core is unknown. Phase 2 measures it; until then, all compute estimates are based on LLM analogies.

5. **Adversarial safety guarantees.** The latent-space monitoring and activation clipping are detection and mitigation, not prevention. Formal safety guarantees for the Core remain an open problem.

6. **Emergent behavior in multi-component systems.** Safety properties of individual components do not guarantee safety of the composed system. The governance layer is a monitoring mechanism, not a guarantee.

7. **Evaluation of internal state.** The Interoceptive Modulator's contribution to system performance is hard to isolate. Ablation studies must be designed carefully.

8. **The criticality hypothesis.** Self-organized criticality may or may not improve artificial systems. Phase 3 tests it; the spec does not assume it.

---

## 13. Summary of Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Core architecture | Decoupled JEPA-Reasoner + Talker | Empirical robustness, matches "generation as readout" principle |
| Latent prediction objective | JEPA with EMA target, optional NextLat, MLM, STP | Hybrid is fallback; pure JEPA is default |
| Scaling model | Multi-phase power law | Hierarchical multi-index theory |
| Data efficiency | STP regularizer, tiered pipeline | Eliminates view-pair bottleneck |
| Compositionality | Functorial constraints on Reasoner | Necessary condition for compositional generalization |
| Latent dimension | Rate-distortion optimal | Information-theoretic |
| Safety | Six-tier architecture | No single mechanism is sufficient |
| Evaluation | Multi-contract with representation benchmark | Single scalar scores are inadequate |
| Training | Staged, not end-to-end | Robustness, verifiability |

---

## 14. The Honest Position

**What we have:**

- A coherent, mathematically grounded architecture.
- A falsifiable central bet.
- A phased build plan with concrete success criteria.
- A representation benchmark that decides the central question.
- A safety architecture that is layered and honest about limits.

**What we do not have:**

- A proven implementation of the functorial Core at scale.
- A measured JEPA-specific scaling law.
- Formal safety guarantees.
- A resolution to the multi-component emergent behavior problem.

**The design is buildable at Phase 0 and coherent through Phase 3.** Phase 0 requires no novel research—it uses existing components (frozen LLM encoder, Jev-style readout heads, hand-written constraints). Phase 1 adds engineering. Phase 2 is the critical experiment: does the Core's scaling behavior validate the central bet? Phase 3 tests whether the architecture can match frontier LLM capability.

**If the central bet fails,** the design remains valuable as a governance and reliability layer around conventional LLMs. This is not a failure—it is an honest outcome.

**If the central bet succeeds,** the design produces a system that is architecturally different in kind from the LLM: more calibrated, more governable, more robust, and structurally more data-efficient.

---

## 15. Next Step

The next step is to write the **Phase 0 build document**: the specific encoder, the specific decision heads, the specific constraints, the specific task, and the evaluation protocol. That document is two to four pages and is sufficient to begin implementation.

The full spec is now complete. Phase 0 can start.
