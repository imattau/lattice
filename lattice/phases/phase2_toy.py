'''Toy-scale analog of PLAN.md "Phase 2: Core Training and Scaling Law
Measurement".

The plan's actual Phase 2 spec (5-7 model sizes from 50M-600M params, 3-4
data budgets each up to 128B tokens, 10-30 PF-days of compute) is not
executable in this environment by many orders of magnitude -- see
PHASE2_TOY_RESULTS.md for the explicit scope-reduction accounting, same
discipline as every other deviation in this project.

What this module does implement, at toy scale, is the plan's actual
*method*: Stage 2a's paired IsoFLOP sweep (train a "Core" and a matched AR
baseline at several model sizes and data budgets, recording final loss)
and Stage 2b's exploratory scaling-law fit (L(N, D) = E + A/N^alpha +
B/D^beta, extracting the ratios the plan uses as its Phase 2 decision-gate
metrics: beta_ratio, alpha_ratio, E_ratio).

Per the Phase 1.5 pilot's findings, the "Core" arm here is
AR+JEPA-auxiliary, not pure latent-primary JEPA: pure JEPA had no
surviving compositional-generalization win after the COGS-replication and
held-out-layer-selection follow-ups
(PHASE1_5_COGS_REPLICATION_RESULTS.md), while AR+JEPA-auxiliary was the
one latent-touching arm with a genuine, non-cherry-picked positive
result. Testing the plan's central bet at this stage means testing
whether folding the JEPA objective in as an auxiliary term buys anything
in a scaling-law sense over plain AR -- not resurrecting the latent-primary
framing the toy evidence argued against.
'''
from __future__ import annotations
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

from lattice.phases.phase1_5_toy import (
    ToyConfig as _JepaAuxConfig, build_ar_jepa, train_ar, train_ar_jepa,
)
from lattice.tiny_transformer import ARLanguageModel, TinyTransformer
from lattice.toy_corpus import (
    CharTokenizer, build_pretrain_corpus, gen_code_snippet, gen_json_record,
    load_banking77_texts, load_triage_texts,
)


@dataclass
class SizeConfig:
    name: str
    dim: int
    n_layers: int
    n_heads: int


# Plan spec: 5-7 model sizes, 50M-600M params. This is 3 sizes spanning
# roughly 50K-3M params (backbone-only, excluding tied embeddings) -- a
# toy-of-the-toy in both count and scale, same honest-reduction pattern as
# Phase 1.5's own model-size accounting.
SIZES: list[SizeConfig] = [
    SizeConfig('small', dim=64, n_layers=2, n_heads=4),
    SizeConfig('medium', dim=128, n_layers=4, n_heads=4),
    SizeConfig('large', dim=256, n_layers=6, n_heads=8),
]

# Plan spec: 3-4 data budgets per size, up to 128B tokens. This is 3
# budgets expressed as epoch counts over the same fixed toy corpus (no
# fresh tokens at higher budgets -- an explicit deviation from the plan's
# "more data" framing to "more passes over the same small corpus", flagged
# honestly in PHASE2_TOY_RESULTS.md) rather than 2B/8B/32B/128B tokens.
DATA_BUDGET_EPOCHS: list[int] = [5, 10, 20]

MAX_LEN = 128
BATCH_SIZE = 32
LR = 3e-4


def build_validation_corpus(seed: int = 999, n_code: int = 500,
                            n_json: int = 500, n_banking: int = 500,
                            n_triage: int = 500,
                            train_n_banking: int = 1000,
                            train_n_triage: int = 1000) -> list[str]:
    '''A genuinely held-out corpus for measuring generalization loss.

    `build_pretrain_corpus` always reads BANKING77/triage text starting
    from row 0 regardless of seed -- a different seed alone does NOT
    produce disjoint real-text documents, only a different code/JSON
    synthetic mix and a different shuffle order. This builds the
    validation slice explicitly offset past the training corpus's real-text
    rows (`train_n_banking`/`train_n_triage`, matching whatever was passed
    to `build_pretrain_corpus` for training) so BANKING77/triage documents
    never appear in both splits, and uses a disjoint seed for the code/JSON
    generators so those don't repeat either.
    '''
    rng = random.Random(seed)
    code = [gen_code_snippet(rng) for _ in range(n_code)]
    struct = [gen_json_record(rng) for _ in range(n_json)]
    banking = load_banking77_texts(offset=train_n_banking, limit=n_banking)
    triage_all = load_triage_texts(limit=train_n_triage + n_triage)
    triage = triage_all[train_n_triage:]
    corpus = code + struct + banking + triage
    rng.shuffle(corpus)
    return corpus


def compute_proxy(num_params: int, tokens_seen: int) -> float:
    '''C ~ N * D (drops the constant ~6x factor from the standard
    transformer FLOPs-per-token approximation; only relative compute
    across cells in the sweep matters here, not an absolute FLOP count).
    '''
    return float(num_params) * float(tokens_seen)


def _held_out_ce_loss(model: torch.nn.Module, tokens: torch.Tensor,
                      is_ar_jepa: bool, causal: bool, device: str,
                      batch_size: int = 256) -> float:
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            if is_ar_jepa:
                logits, _ = model.ar_model(batch)
            else:
                logits, _ = model(batch)
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets.reshape(-1), ignore_index=CharTokenizer.PAD,
                reduction='sum',
            )
            valid = (targets != CharTokenizer.PAD).sum().item()
            total_loss += loss.item()
            total_count += valid
    return total_loss / max(1, total_count)


def run_isoflop_cell(
    size: SizeConfig, epochs: int, train_tokens: torch.Tensor,
    val_tokens: torch.Tensor, vocab_size: int, device: str, seed: int = 0,
) -> dict:
    '''Trains one AR baseline and one AR+JEPA-auxiliary Core at a given
    (model size, data budget) cell, matched architecture and budget, and
    returns both arms' held-out cross-entropy loss plus the bookkeeping
    (param count, tokens seen, compute proxy) a scaling-law fit needs.
    '''
    torch.manual_seed(seed)
    config = _JepaAuxConfig(dim=size.dim, n_layers=size.n_layers,
                            n_heads=size.n_heads, max_len=MAX_LEN,
                            batch_size=BATCH_SIZE, epochs=epochs, lr=LR,
                            seed=seed)
    tokens_seen = epochs * train_tokens.size(0) * MAX_LEN

    ar_backbone = TinyTransformer(vocab_size, size.dim, size.n_layers,
                                  size.n_heads, MAX_LEN)
    ar_model = ARLanguageModel(ar_backbone)
    train_ar(ar_model, train_tokens, config, device)
    ar_val_loss = _held_out_ce_loss(ar_model, val_tokens, is_ar_jepa=False,
                                    causal=True, device=device)

    ar_jepa = build_ar_jepa(vocab_size, config)
    train_ar_jepa(ar_jepa, train_tokens, config, device)
    ar_jepa_val_loss = _held_out_ce_loss(ar_jepa, val_tokens, is_ar_jepa=True,
                                         causal=True, device=device)

    return {
        'size': size.name,
        'dim': size.dim,
        'n_layers': size.n_layers,
        'epochs': epochs,
        'tokens_seen': tokens_seen,
        'ar': {
            'num_params': ar_model.num_params(),
            'val_ce_loss': ar_val_loss,
            'compute_proxy': compute_proxy(ar_model.num_params(), tokens_seen),
        },
        'ar_jepa_aux': {
            'num_params': ar_jepa.num_params(),
            'val_ce_loss': ar_jepa_val_loss,
            'compute_proxy': compute_proxy(ar_jepa.num_params(), tokens_seen),
        },
    }


def _scaling_law(ND: np.ndarray, E: float, A: float, alpha: float,
                 B: float, beta: float) -> np.ndarray:
    n, d = ND
    return E + A / (n ** alpha) + B / (d ** beta)


def fit_scaling_law(records: list[dict], arm: str) -> dict:
    '''Fits L(N, D) = E + A/N^alpha + B/D^beta (plan's Stage 2b candidate
    model) via nonlinear least squares over the IsoFLOP sweep's cells for
    one arm. Returns the fitted parameters plus R^2. With only
    len(SIZES) x len(DATA_BUDGET_EPOCHS) points and one seed, this is
    illustrative, not a statistically powered fit -- report accordingly.
    '''
    n = np.array([r[arm]['num_params'] for r in records], dtype=np.float64)
    d = np.array([r['tokens_seen'] for r in records], dtype=np.float64)
    loss = np.array([r[arm]['val_ce_loss'] for r in records], dtype=np.float64)

    e0 = float(loss.min()) * 0.5
    p0 = [e0, 1.0, 0.3, 1.0, 0.3]
    bounds = ([0, 0, 0.01, 0, 0.01], [loss.min(), np.inf, 3.0, np.inf, 3.0])
    try:
        popt, _ = curve_fit(_scaling_law, (n, d), loss, p0=p0, bounds=bounds,
                            maxfev=20000)
    except RuntimeError:
        return {'fit_failed': True}

    pred = _scaling_law((n, d), *popt)
    ss_res = float(((loss - pred) ** 2).sum())
    ss_tot = float(((loss - loss.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    E, A, alpha, B, beta = (float(x) for x in popt)
    return {'E': E, 'A': A, 'alpha': alpha, 'B': B, 'beta': beta, 'r2': r2,
           'fit_failed': False}
