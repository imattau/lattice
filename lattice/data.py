'''Data loading. Tiered: natural pairs, STP-ready monolingual, DLLM masking.'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional
import random

import torch
from torch.utils.data import Dataset

from lattice.config import DataTier


@dataclass
class ViewPair:
    '''Two semantically-equivalent views of the same content.'''
    view_a: list[int]
    view_b: list[int]


class TokenizedCorpus(Dataset):
    '''A minimal tokenized corpus. Pack sequences to seq_len.'''

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = self.seq_len + 1 - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad, dtype=torch.long)])
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
            'attention_mask': torch.ones(self.seq_len, dtype=torch.bool),
        }


class ViewPairCorpus(Dataset):
    '''Tier 1 corpus: naturally paired views (code<->doc, Q<->A).'''

    def __init__(self, pairs: list[ViewPair], seq_len: int):
        self.pairs = pairs
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            'view_a': self._pad(pair.view_a),
            'view_b': self._pad(pair.view_b),
        }

    def _pad(self, ids: list[int]) -> torch.Tensor:
        ids = ids[: self.seq_len]
        pad = self.seq_len - len(ids)
        return torch.tensor(ids + [0] * pad, dtype=torch.long)


def build_dataloader(tier: DataTier, source: Iterable,
                     batch_size: int, seq_len: int,
                     shuffle: bool = True, num_workers: int = 0):
    from torch.utils.data import DataLoader
    if tier == DataTier.TIER1_NATURAL_PAIRS:
        ds = ViewPairCorpus(list(source), seq_len)
    else:
        # Tier 2 and 3 use flat token streams. Tier 3 additionally uses
        # DLLM double-masking at the model level, not the data level.
        ds = TokenizedCorpus(list(source), seq_len)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True,
    )
