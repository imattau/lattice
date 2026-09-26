'''STP per-task validation. Verification made this a gate.

The 16x data-efficiency result was measured on NL-RX-SYNTH. This script
reproduces it and then runs STP on a second task. If the second task
does not show a gain, STP is treated as task-specific.
'''
from __future__ import annotations
from dataclasses import dataclass

from config import TrainConfig


@dataclass
class STPResult:
    task: str
    data_efficiency_ratio: float
    reproduction_of_original: bool


def validate_stp(cfg: TrainConfig, task: str, dataloader,
                 device: str = 'cpu') -> STPResult:
    '''Train with and without STP at multiple data budgets. Fit data
    exponent for each and report the ratio. The NL-RX-SYNTH task should
    reproduce the 16x result; other tasks should be compared fresh.'''
    raise NotImplementedError(
        'Requires a task-specific loader and matched single-task runs. '
        'The protocol is: for D in {D1, D2, D3, D4}, train with and '
        'without STP; fit single_phase power law in scaling.py; report '
        'beta_without / beta_with as the data-efficiency ratio.'
    )
