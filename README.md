# The Lattice

A layered, decision-centric AI system architecture.

- [SPEC.md](./SPEC.md) — design specification v1.0
- [PLAN.md](./PLAN.md) — reconciled implementation-grade phase plan
- [PLAN_REVISION.md](./PLAN_REVISION.md) — verification-driven training revisions
- [lattice/](./lattice/) — unified implementation package
  - components: interfaces, readout heads, constraint layer, controller, calibration
  - core: JEPA Core, autoregressive baseline, masking, objectives
  - training: data loaders, trainer, paired runner, scaling analysis, probes
  - phases: Phase 0/1/1.5/2 entry points
  - experiments: functoriality ablation, STP validation
