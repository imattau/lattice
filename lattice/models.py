'''JEPA Core and autoregressive baseline with matched backbone size.

Both models share the Reasoner backbone. The AR baseline adds an LM head
on top; the JEPA Core adds an EMA target encoder, a predictor, and a
Talker. This lets paired runs compare objectives rather than capacity.
'''
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Reasoner(nn.Module):
    '''Shared predictive representation. Same module for both models.'''

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_reasoner_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)[None].expand(B, T)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        key_padding = ~attention_mask if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding)
        return self.norm(x)

    @torch.no_grad()
    def ema_update_from(self, source: nn.Module, decay: float) -> None:
        for tp, sp in zip(self.parameters(), source.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


class Predictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Talker(nn.Module):
    '''Token reconstruction head. Reads from Reasoner latents only.'''

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_talker_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


class JEPACore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.reasoner = Reasoner(cfg)
        self.target_reasoner = Reasoner(cfg)
        self.target_reasoner.load_state_dict(self.reasoner.state_dict())
        for p in self.target_reasoner.parameters():
            p.requires_grad = False
        self.predictor = Predictor(cfg.hidden_dim)
        self.talker = Talker(cfg)

    def context_encode(self, input_ids, attention_mask=None):
        return self.reasoner(input_ids, attention_mask)

    @torch.no_grad()
    def target_encode(self, input_ids, attention_mask=None):
        return self.target_reasoner(input_ids, attention_mask)

    def update_target(self, decay: float) -> None:
        self.target_reasoner.ema_update_from(self.reasoner, decay)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.talker(latent)


class ARBaseline(nn.Module):
    '''Standard autoregressive LM with the same Reasoner backbone.'''

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.reasoner = Reasoner(cfg)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        return self.lm_head(self.reasoner(input_ids, attention_mask))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def estimate_flops_per_step(model: nn.Module, cfg: ModelConfig,
                            batch_size: int, seq_len: int,
                            backward_passes: int = 1) -> float:
    '''Rough FLOPs estimate for the Reasoner backbone forward + backward.

    Uses the standard 6 * N_params * N_tokens approximation, scaled by
    the number of forward+backward passes actually performed. This is
    sufficient for the paired-run matched compute comparison.
    '''
    n_params = count_parameters(model, trainable_only=True)
    n_tokens = batch_size * seq_len
    return 6.0 * n_params * n_tokens * (1 + backward_passes)
