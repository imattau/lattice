'''Functoriality ablation. Verification made this a soft flag.

Trains two Cores that differ only in whether the functorial consistency
loss is active. Compares on compositional generalization benchmarks.
'''
from __future__ import annotations
import copy

from config import TrainConfig
from trainer import train


def run_ablation(base_cfg: TrainConfig, dataloader, device='cpu'):
    cfg_on = copy.deepcopy(base_cfg)
    cfg_on.enable_functorial = True
    cfg_off = copy.deepcopy(base_cfg)
    cfg_off.enable_functorial = False

    from models import JEPACore
    core_on = JEPACore(cfg_on.model)
    core_off = JEPACore(cfg_off.model)

    out_on = train(core_on, dataloader, cfg_on, device)
    out_off = train(core_off, dataloader, cfg_off, device)

    return {
        'with_functorial': out_on,
        'without_functorial': out_off,
        'note': (
            'Compare on compositional generalization. If no measurable '
            'gain, drop the functorial constraint from the design.'
        ),
    }
