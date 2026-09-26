# Reference Training Implementation

This package encodes the **revised** training plan from the verification: functoriality is a soft ablation, STP is per-task, the scaling analysis is exploratory, paired JEPA/AR runs are first-class, toy-scale falsification gates Phase 2, and calibration is a training objective from Phase 0.

## Package layout

```
lattice/
  __init__.py
  config.py                 # dataclasses for all training configuration
  data.py                   # tokenized dataset, view-pair construction, tiered loading
  models.py                 # JEPA Core and AR baseline, matched backbone
  objectives.py             # combined loss with per-term switches
  calibration.py            # RLCD surrogate and post-hoc temperature
  trainer.py                # staged training loop
  paired.py                 # matched-compute JEPA vs AR runner
  scaling.py                # single-phase vs multi-phase fitting + model choice
  phases/
    __init__.py
    phase0_readouts.py
    phase1_triage.py
    phase1_5_toy_falsification.py
    phase2_core_training.py
  experiments/
    __init__.py
    functorial_ablation.py
    stp_validation.py
```

## Deliberately stubbed

Three items raise `NotImplementedError` rather than being faked:

1. **Phase 1.5 dataloader plumbing.** Needs a tokenized corpus or view-pair corpus.
2. **STP validation across tasks.** The STP paper reports one dataset; the script states the protocol.
3. **The functoriality implementation.** `MonoidalCompose` is a learned bilinear form, not a true monoidal functor.

Everything else runs given data and compute.
