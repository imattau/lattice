'''Phase 2: Core training and scaling law measurement.

This module is a placeholder. The full implementation uses trainer.py
and paired.py to run the IsoFLOP sweep, then scaling.py to fit and
select the scaling model.
'''
from __future__ import annotations


def run_phase2():
    raise NotImplementedError(
        'Phase 2 is implemented by configuring TrainConfig, running '
        'trainer.train and paired.run_paired across model sizes and data '
        'budgets, and fitting scaling models with scaling.choose_scaling_model.'
    )
