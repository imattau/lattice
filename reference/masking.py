"""Multi-block masking for JEPA. Spec §3.1.

Block scale sampled uniformly from [0.15, 0.20]; aspect ratio from
[0.75, 1.50]. Adapted from vision JEPA to 1-D sequence.
"""
from __future__ import annotations
import math
import random

import torch


def sample_block(
    seq_len: int,
    scale_min: float = 0.15,
    scale_max: float = 0.20,
    aspect_min: float = 0.75,
    aspect_max: float = 1.50,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Sample one contiguous block (start, length)."""
    rng = rng or random
    scale = rng.uniform(scale_min, scale_max)
    aspect = rng.uniform(aspect_min, aspect_max)
    # For 1-D, block length ~ sqrt(scale * seq_len^2 * aspect)
    length = int(round(math.sqrt(scale * seq_len * seq_len * aspect)))
    length = max(1, min(seq_len, length))
    start = rng.randint(0, seq_len - length)
    return start, length


def multi_block_mask(
    batch_size: int,
    seq_len: int,
    num_blocks: int = 4,
    rng: random.Random | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a boolean mask [B, seq_len]; True = masked (target)."""
    rng = rng or random
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for b in range(batch_size):
        for _ in range(num_blocks):
            start, length = sample_block(seq_len, rng=rng)
            mask[b, start:start + length] = True
    return mask
