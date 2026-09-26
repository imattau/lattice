'''Phase 1.5: Toy-scale falsification.

The verification inserted this phase between Phase 1 and Phase 2. Its
job is to answer, at 10M-50M parameters, whether a JEPA Core develops
representations competitive with an AR baseline on the Latent Probing
Suite. If it does not, large-scale training will not save the design.
'''
from __future__ import annotations
import argparse

import torch

from lattice.config import (
    DataTier, LossWeights, ModelConfig, Objective, TrainConfig,
)
from lattice.data import build_dataloader
from lattice.paired import run_paired
from lattice.scaling import choose_scaling_model
from lattice.representation_probe import LinearProbeSuite


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=200_000_000)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_reasoner_blocks=args.blocks,
        num_talker_blocks=2,
        max_seq_len=512,
    )
    jepa_cfg = TrainConfig(
        objective=Objective.JEPA_NEXTLAT_MLM,
        data_tier=DataTier.TIER1_NATURAL_PAIRS,
        seq_len=512,
        total_steps=args.tokens // (16 * 512),
        model=model_cfg,
        loss_weights=LossWeights(),
    )
    ar_cfg = TrainConfig(
        objective=Objective.JEPA_ONLY,   # unused for AR
        data_tier=DataTier.TIER1_NATURAL_PAIRS,
        seq_len=512,
        total_steps=jepa_cfg.total_steps,
        model=model_cfg,
    )

    # Loader construction is caller-specific; this is a stub.
    raise NotImplementedError(
        'Provide a dataloader via build_dataloader with a tokenized '
        'corpus or view-pair corpus. See data.py.'
    )

    result = run_paired(model_cfg, jepa_cfg, ar_cfg, dataloader,
                        device=args.device)

    probe = LinearProbeSuite(
        feature_dim=model_cfg.hidden_dim,
        tasks=['sst2', 'mrpc', 'mnli', 'cola', 'stsb'],
    )
    # Evaluate both models on the Latent Probing Suite here.
    # Success criteria: JEPA probe accuracy within 10% of AR baseline.
