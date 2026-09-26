# Reference Implementations

Python reference implementations for the Lattice specification. They are organized by spec section, use PyTorch, and are designed to be runnable. Where the spec involves unproven research (functorial Core, criticality controller), the code is structured to make the hypothesis testable rather than assumed.

## File Mapping

| Spec section | Implementation |
|---|---|
| §3.1–3.2 Core | `reference/jepa_core.py` |
| §3.3 Readout heads | `reference/readout_heads.py` |
| §3.4 Controller | `reference/controller.py` |
| §4 Interfaces | `reference/interfaces.py` |
| §5.1 Scaling law | `reference/scaling_law.py` |
| §5.6 STP | `reference/losses.stp_regularizer` |
| §6 Training objective | `reference/losses.py`, `reference/training.py` |
| §8.1 Immutable constraints | `reference/constraint_layer.py` |
| §9.1 Calibration | `reference/calibration.py` |
| §9.5 Representation benchmark | `reference/representation_bench.py` |
| §9.6 IsoFLOP sweep | `reference/scaling_law.py` |
| §10 Phase 0 | `reference/demo_phase0.py` |

## What Is Deliberately Not Implemented

Three items in the spec are left as stubs because they are research problems, not engineering problems, and fabricating an implementation would obscure that:

1. **The functorial composition operator.** `reference/losses.functorial_consistency_loss` accepts a `compose_fn` but does not provide a default. There is no established implementation of a monoidal-functor transformer at scale. This is the single largest engineering open problem in the spec.

2. **The criticality controller.** No implementation is provided. The spec treats criticality as a hypothesis to test against a fixed-temperature baseline; writing code before the test is designed would prejudge the outcome.

3. **RLCD proper.** `reference/calibration.rlcd_loss` is a differentiable surrogate. The real RLCD optimizes calibration under a reward signal, which requires a training loop not specified here.

Everything else runs. `reference/demo_phase0.py` executes end-to-end and is the starting point for Phase 0 implementation.
