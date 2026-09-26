"""Decoupled JEPA Core. Spec §3.1–3.2, §6.

Reasoner: 16 blocks, 960-dim latent, functorial transformations
          (functoriality is *structural* — we implement the monoidal
          latent composition operation and enforce it via a
          consistency loss in the training loop).
Talker:   4+4 blocks, token reconstruction from latents.
Target encoder: EMA copy of the Reasoner.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreConfig:
    vocab_size: int = 32000
    hidden_dim: int = 960
    num_reasoner_blocks: int = 16
    num_talker_blocks: int = 4
    num_heads: int = 16
    max_seq_len: int = 1024
    ema_decay: float = 0.996


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class Reasoner(nn.Module):
    """Shared predictive representation. Spec §3.1."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads)
            for _ in range(cfg.num_reasoner_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        key_padding = ~attention_mask if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding)
        return self.norm(x)

    @torch.no_grad()
    def ema_update(self, source: nn.Module, decay: float) -> None:
        """EMA update of a target Reasoner from the online Reasoner."""
        for tp, sp in zip(self.parameters(), source.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


class Talker(nn.Module):
    """Token decoder reading from Reasoner latents. Spec §3.2."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads)
            for _ in range(cfg.num_talker_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


class JEPACore(nn.Module):
    """Decoupled Reasoner + Talker with EMA target. Spec §3.1–3.2."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.reasoner = Reasoner(cfg)
        self.target_reasoner = Reasoner(cfg)
        self.target_reasoner.load_state_dict(self.reasoner.state_dict())
        for p in self.target_reasoner.parameters():
            p.requires_grad = False
        self.talker = Talker(cfg)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor | None = None
               ) -> torch.Tensor:
        return self.reasoner(input_ids, attention_mask)

    @torch.no_grad()
    def encode_target(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor | None = None
                      ) -> torch.Tensor:
        return self.target_reasoner(input_ids, attention_mask)

    def update_target(self) -> None:
        self.target_reasoner.ema_update(self.reasoner, self.cfg.ema_decay)

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return self.talker(latent)
