'''Minimal transformer shared between the AR baseline and the JEPA
context/target encoders for the Phase 1.5 toy-scale falsification (PLAN.md
"Phase 1.5"). Both variants use the exact same `TinyTransformer` backbone
(same depth, width, heads, context length) so that any measured difference
is attributable to the training objective, not capacity.
'''
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
               key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    '''Token + positional embedding, N pre-norm transformer blocks. Used
    bidirectionally (causal=False) as the JEPA context/target encoder, and
    causally (causal=True) as the AR baseline's backbone.
    '''

    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 6,
                n_heads: int = 8, max_len: int = 192):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, causal: bool = False
               ) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=tokens.device),
                diagonal=1,
            )
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.ln_f(x)


class ARLanguageModel(nn.Module):
    '''AR baseline: TinyTransformer + causal mask + tied LM head.'''

    def __init__(self, backbone: TinyTransformer):
        super().__init__()
        self.backbone = backbone
        vocab_size = backbone.token_emb.num_embeddings
        self.lm_head = nn.Linear(backbone.dim, vocab_size, bias=False)
        self.lm_head.weight = backbone.token_emb.weight  # tied embeddings

    def forward(self, tokens: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(tokens, causal=True)
        return self.lm_head(h), h

    def num_params(self) -> int:
        # lm_head shares weights with token_emb; don't double-count.
        return self.backbone.num_params()


class Predictor(nn.Module):
    '''2-layer MLP, GELU (spec: Predictor).'''

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ema_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    '''In-place EMA update of `target` params toward `source` params.'''
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.mul_(tau).add_(ps.detach(), alpha=1 - tau)


def tau_schedule(step: int, total_steps: int, start: float = 0.996,
                 end: float = 1.0) -> float:
    '''Linear EMA schedule, 0.996 -> 1.0 over training (spec: I-JEPA schedule).'''
    if total_steps <= 1:
        return end
    frac = min(1.0, step / (total_steps - 1))
    return start + (end - start) * frac
