'''Run the toy-scale Phase 2 analog (PLAN.md "Phase 2: Core Training and
Scaling Law Measurement", scaled down -- see PHASE2_TOY_RESULTS.md).

Trains AR and AR+JEPA-auxiliary (the Phase 1.5 pilot's one arm with a
surviving, non-cherry-picked positive result -- not pure latent-primary
JEPA, which had no surviving compositional win after
PHASE1_5_COGS_REPLICATION_RESULTS.md) at every (model size, data budget)
cell in a small IsoFLOP-style grid, records held-out cross-entropy loss
for both, then fits the plan's Stage 2b candidate scaling law
L(N, D) = E + A/N^alpha + B/D^beta to each arm's sweep and reports the
plan's actual decision-gate ratios: beta_ratio (data-efficiency),
alpha_ratio (parameter-scaling), E_ratio (irreducible loss).
'''
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from lattice.phases.phase2_toy import (
    DATA_BUDGET_EPOCHS, MAX_LEN, SIZES, build_validation_corpus,
    fit_scaling_law, run_isoflop_cell,
)
from lattice.toy_corpus import CharTokenizer, build_pretrain_corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train_docs_per_source', type=int, default=1000)
    p.add_argument('--n_val_docs_per_source', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default='PHASE2_TOY_RESULTS.json')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('Building train/validation corpora (disjoint real-text rows) ...')
    train_texts = build_pretrain_corpus(
        seed=args.seed, n_code=args.n_train_docs_per_source,
        n_json=args.n_train_docs_per_source,
        n_banking=args.n_train_docs_per_source,
        n_triage=args.n_train_docs_per_source)
    val_texts = build_validation_corpus(
        seed=args.seed + 999, n_code=args.n_val_docs_per_source,
        n_json=args.n_val_docs_per_source,
        n_banking=args.n_val_docs_per_source,
        n_triage=args.n_val_docs_per_source,
        train_n_banking=args.n_train_docs_per_source,
        train_n_triage=args.n_train_docs_per_source)
    tokenizer = CharTokenizer.fit(train_texts + val_texts)
    train_tokens = tokenizer.encode_batch(train_texts, MAX_LEN)
    val_tokens = tokenizer.encode_batch(val_texts, MAX_LEN)
    print(f'  {len(train_texts)} train docs, {len(val_texts)} val docs, '
         f'vocab_size={tokenizer.vocab_size}')

    records = []
    t_start = time.time()
    for size in SIZES:
        for epochs in DATA_BUDGET_EPOCHS:
            t0 = time.time()
            cell = run_isoflop_cell(size, epochs, train_tokens, val_tokens,
                                    tokenizer.vocab_size, device,
                                    seed=args.seed)
            wall = time.time() - t0
            cell['wall_seconds'] = wall
            records.append(cell)
            print(f'[{size.name:6s} epochs={epochs:3d}] '
                 f'ar_val_ce={cell["ar"]["val_ce_loss"]:.4f} '
                 f'(N={cell["ar"]["num_params"]:,})  '
                 f'ar_jepa_aux_val_ce={cell["ar_jepa_aux"]["val_ce_loss"]:.4f} '
                 f'(N={cell["ar_jepa_aux"]["num_params"]:,})  '
                 f'wall={wall:.1f}s')

    print(f'\nSweep complete in {time.time() - t_start:.1f}s. Fitting '
         f'scaling laws ...')
    ar_fit = fit_scaling_law(records, arm='ar')
    aux_fit = fit_scaling_law(records, arm='ar_jepa_aux')

    result = {
        'sizes': [{'name': s.name, 'dim': s.dim, 'n_layers': s.n_layers,
                  'n_heads': s.n_heads} for s in SIZES],
        'data_budget_epochs': DATA_BUDGET_EPOCHS,
        'max_len': MAX_LEN,
        'records': records,
        'scaling_fit': {'ar': ar_fit, 'ar_jepa_aux': aux_fit},
    }

    if not ar_fit['fit_failed'] and not aux_fit['fit_failed']:
        # E (irreducible loss) is only reported as a ratio if both fits
        # place it at a non-degenerate value -- with this few (N, D) points
        # spanning this narrow a range, curve_fit can and does drive E to
        # numerically-zero for both arms, at which point their ratio is a
        # meaningless division of two near-zero floats, not a real signal.
        e_floor = 1e-3
        e_identified = ar_fit['E'] > e_floor and aux_fit['E'] > e_floor
        result['ratios'] = {
            'beta_ratio_ar_jepa_aux_over_ar': aux_fit['beta'] / ar_fit['beta'],
            'alpha_ratio_ar_jepa_aux_over_ar': aux_fit['alpha'] / ar_fit['alpha'],
            'E_ratio_ar_jepa_aux_over_ar': (
                aux_fit['E'] / ar_fit['E'] if e_identified else None),
            'E_identified': e_identified,
        }
        print('\n=== Plan decision-gate metrics (AR+JEPA-aux vs. AR) ===')
        print(f'  beta ratio (want < 0.5 for a win): '
             f'{result["ratios"]["beta_ratio_ar_jepa_aux_over_ar"]:.3f}')
        print(f'  alpha ratio (want > 0.8 for a win): '
             f'{result["ratios"]["alpha_ratio_ar_jepa_aux_over_ar"]:.3f}')
        if e_identified:
            print(f'  E ratio (want <= 1.0 for a win): '
                 f'{result["ratios"]["E_ratio_ar_jepa_aux_over_ar"]:.3f}')
        else:
            print(f'  E ratio: NOT IDENTIFIED -- both fits placed E at '
                 f'~0 (AR={ar_fit["E"]:.2e}, aux={aux_fit["E"]:.2e}), an '
                 f'artifact of too few (N,D) points over too narrow a '
                 f'range for this term to be distinguishable from zero, '
                 f'not a real irreducible-loss estimate for either arm.')
        print(f'  AR fit:          E={ar_fit["E"]:.3e} A={ar_fit["A"]:.3f} '
             f'alpha={ar_fit["alpha"]:.3f} B={ar_fit["B"]:.3f} '
             f'beta={ar_fit["beta"]:.3f} R2={ar_fit["r2"]:.3f}')
        print(f'  AR+JEPA-aux fit: E={aux_fit["E"]:.3e} A={aux_fit["A"]:.3f} '
             f'alpha={aux_fit["alpha"]:.3f} B={aux_fit["B"]:.3f} '
             f'beta={aux_fit["beta"]:.3f} R2={aux_fit["r2"]:.3f}')
    else:
        print('\nScaling-law fit failed for at least one arm -- see '
             '"scaling_fit" in the output JSON.')

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
