'''Vector Symbolic Architecture (VSA) binding primitives, and a small
neural module that gives a TinyTransformer explicit access to them.

Motivation: every Core arm tested in this project (AR, JEPA, contrastive,
AR+JEPA-auxiliary) failed identically at the two-hop indirection task --
either flooring at chance under a frozen probe (zero pretraining exposure)
or overfitting to memorization under fine-tuning
(PHASE1_5_FINETUNE_INDIRECTION_RESULTS.md: both AR and AR+JEPA-aux hit
1.000 train accuracy and ~0 test accuracy on 192 examples). That failure
mode -- "bind A to B, later retrieve B given A" -- is exactly what
Holographic Reduced Representations (Plate, 1995) and related VSA schemes
were built to do as an explicit, non-learned *operator*, not something a
generic attention/MLP stack has to discover from scratch via curve-fitting
on a handful of examples.

This module implements the Fourier-domain HRR ("FHRR") variant: binding is
circular convolution, computed as elementwise complex multiplication in
the Fourier domain; unbinding is exact (not approximate) when the "key"
vector's Fourier spectrum has unit magnitude at every frequency (a
"phasor" vector) -- `_unit_spectrum` enforces this, so unbinding recovers
the bound value with zero self-noise, unlike the "clean-up memory + noisy
correlation" version of HRR used when keys aren't constrained this way.
'''
from __future__ import annotations
import torch
import torch.nn as nn

from lattice.tiny_transformer import TinyTransformer


def _unit_spectrum(x: torch.Tensor) -> torch.Tensor:
    '''Projects `x` onto the manifold of vectors whose (real) FFT has unit
    magnitude at every frequency -- an FHRR "phasor" vector. Binding with a
    unit-spectrum key is then invertible by construction (dividing by a
    unit-magnitude complex number is the same as multiplying by its
    conjugate).'''
    X = torch.fft.rfft(x, dim=-1)
    X_unit = X / X.abs().clamp(min=1e-8)
    return torch.fft.irfft(X_unit, n=x.shape[-1], dim=-1)


def circular_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''bind(a, b) = circular convolution of a and b, via the Fourier
    convolution theorem: FFT(a*b_conv) = FFT(a) . FFT(b) (elementwise).'''
    A = torch.fft.rfft(a, dim=-1)
    B = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1)


def circular_unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    '''Recovers `b` from `bound = circular_bind(key, b)`, exactly, when
    `key` has unit-magnitude Fourier spectrum (division by a unit-magnitude
    complex number = multiplication by its conjugate, so this is the exact
    inverse of `circular_bind` in that case, not an approximation).'''
    C = torch.fft.rfft(bound, dim=-1)
    K = torch.fft.rfft(key, dim=-1)
    K_inv = torch.conj(K) / (K.abs().clamp(min=1e-8) ** 2)
    return torch.fft.irfft(C * K_inv, n=bound.shape[-1], dim=-1)


class VSAMemoryModule(nn.Module):
    '''Gives a sequence encoder an explicit associative-memory pathway:
    at every position, project a (key, value, query) triple; bind each
    position's key to its value with unit-spectrum keys (so binding is
    exactly invertible); sum all bindings into one fixed-size memory vector
    per example (a superposed HRR memory); then, at every position, use
    that position's own query to unbind against the shared memory and
    retrieve whatever was bound there. This is a *mechanism* the model can
    learn to use for binding/retrieval -- it doesn't hand-solve the task,
    the key/value/query projections and the decision of which positions to
    write/read from are still learned end-to-end.
    '''

    def __init__(self, dim: int):
        super().__init__()
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        '''hidden: [B, T, D] -> retrieved: [B, T, D] (added residually to
        `hidden` by the caller, e.g. `VSAEncoder`).'''
        keys = _unit_spectrum(self.key_proj(hidden))
        values = self.value_proj(hidden)
        queries = _unit_spectrum(self.query_proj(hidden))

        bound = circular_bind(keys, values)          # [B, T, D]
        memory = bound.sum(dim=1, keepdim=True)       # [B, 1, D]
        retrieved = circular_unbind(memory.expand_as(queries), queries)
        return self.out_proj(retrieved)


class VSAEncoder(nn.Module):
    '''A TinyTransformer backbone (bidirectional, matched depth/width/heads
    to the other Phase 1.5 arms) plus a `VSAMemoryModule` combined
    residually after the final transformer layer. Same pooled-feature
    interface as the other arms (`forward(tokens) -> [B, D]`) so it drops
    into the existing probe/fine-tune harness unchanged.
    '''

    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                n_heads: int, max_len: int):
        super().__init__()
        self.backbone = TinyTransformer(vocab_size, dim, n_layers, n_heads,
                                        max_len)
        self.memory = VSAMemoryModule(dim)
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(tokens, causal=False)   # [B, T, D]
        augmented = hidden + self.memory(hidden)        # [B, T, D]
        return augmented.mean(dim=1)                    # [B, D]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BidirectionalPooledBackbone(nn.Module):
    '''Thin wrapper giving a plain TinyTransformer the same
    `forward(tokens) -> [B, D]` pooled interface as `VSAEncoder`, so the
    two can be trained and evaluated by one shared, generic loop -- the
    fair baseline for the VSA-augmented arm, matched in every way except
    the added memory module.
    '''

    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                n_heads: int, max_len: int):
        super().__init__()
        self.backbone = TinyTransformer(vocab_size, dim, n_layers, n_heads,
                                        max_len)
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens, causal=False).mean(dim=1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
