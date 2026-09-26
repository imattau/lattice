'''Training configuration. All knobs in one place, serializable to JSON.'''
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


class Objective(str, Enum):
    JEPA_ONLY = 'jepa_only'
    JEPA_NEXTLAT = 'jepa_nextlat'
    JEPA_NEXTLAT_MLM = 'jepa_nextlat_mlm'
    JEPA_NEXTLAT_MLM_STP = 'jepa_nextlat_mlm_stp'
    JEPA_NEXTLAT_MLM_STP_FUNCTORIAL = 'jepa_nextlat_mlm_stp_functorial'


class DataTier(str, Enum):
    TIER1_NATURAL_PAIRS = 'tier1'
    TIER2_STP = 'tier2'
    TIER3_DLLM_MASKING = 'tier3'


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_dim: int = 960
    num_reasoner_blocks: int = 16
    num_talker_blocks: int = 4
    num_heads: int = 16
    max_seq_len: int = 1024
    dropout: float = 0.0


@dataclass
class JEPAConfig:
    ema_decay_start: float = 0.996
    ema_decay_end: float = 1.0          # linear schedule over training
    mask_scale_min: float = 0.15
    mask_scale_max: float = 0.20
    mask_aspect_min: float = 0.75
    mask_aspect_max: float = 1.50
    num_mask_blocks: int = 4


@dataclass
class LossWeights:
    alpha_nextlat: float = 0.1
    beta_mlm: float = 0.1
    gamma_stp: float = 0.01
    delta_functorial: float = 0.01


@dataclass
class TrainConfig:
    objective: Objective = Objective.JEPA_ONLY
    data_tier: DataTier = DataTier.TIER1_NATURAL_PAIRS
    batch_size: int = 16
    seq_len: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    total_steps: int = 10000
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000
    output_dir: str = './runs/default'
    model: ModelConfig = field(default_factory=ModelConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    # Functoriality is a flag, not a hard requirement. Verification made
    # this explicit.
    enable_functorial: bool = False
    # STP is validated per-task. This flag is the gate.
    enable_stp: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.to_json())
