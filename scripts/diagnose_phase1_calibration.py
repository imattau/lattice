'''Decompose the Phase 1 routing-head ECE (0.595 @ 93% accuracy) before
prescribing a fix. Reuses cached features from scripts/train_phase1.py so
training is exactly reproduced (same seed, same hyperparameters).

Three diagnostics:
  1. Confidence histogram + reliability (accuracy per confidence bucket) for
     the routing head on test.jsonl — distinguishes overconfidence from
     underconfidence.
  2. ECE split by hard_case is-not-None vs None within test.jsonl — checks
     whether adversarial/hard-case documents are driving the number or
     whether it's already bad on clean data.
  3. Per-head ECE is already reported separately by evaluate_heads (routing/
     urgency/escalation/safety are never pooled into one global number), so
     this script just re-prints that breakdown for the record instead of
     re-deriving it.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.calibration import expected_calibration_error
from lattice.interfaces import DecisionType
from lattice.phases.phase1_triage import (
    build_bank, evaluate_heads, labels_from_docs, load_jsonl, train_bank,
)


def confidence_histogram(probs: torch.Tensor, labels: torch.Tensor,
                         n_bins: int = 10) -> list[dict]:
    conf, pred = probs.max(dim=-1)
    acc = pred.eq(labels)
    edges = torch.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        n = int(in_bin.sum())
        rows.append({
            'bucket': f'({lo:.1f}, {hi:.1f}]',
            'n': n,
            'frac_of_test': n / len(conf),
            'mean_confidence': float(conf[in_bin].mean()) if n else None,
            'accuracy': float(acc[in_bin].float().mean()) if n else None,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/triage')
    p.add_argument('--cache_dir', default='data/triage/features_cache')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hidden_dim', type=int, default=384)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    cache_dir = Path(args.cache_dir)
    train_docs = load_jsonl(data_dir / 'train.jsonl')
    test_docs = load_jsonl(data_dir / 'test.jsonl')
    train_features = torch.load(cache_dir / 'train.pt')
    test_features = torch.load(cache_dir / 'test.pt')
    train_labels = labels_from_docs(train_docs)
    test_labels = labels_from_docs(test_docs)

    bank = build_bank(hidden_dim=train_features.size(-1))
    train_bank(bank, train_features, train_labels, epochs=args.epochs,
              lr=args.lr, beta_rlcd=args.beta)
    bank.eval()

    print('=== Diagnostic 3: per-head ECE (already isolated, not pooled) ===')
    print(json.dumps(evaluate_heads(bank, test_features, test_labels),
                     indent=2))

    with torch.no_grad():
        outputs = bank(test_features)
    routing_probs = outputs[DecisionType.ROUTING].distribution
    routing_labels = test_labels[DecisionType.ROUTING]

    print('\n=== Diagnostic 1: routing confidence histogram + reliability ===')
    for row in confidence_histogram(routing_probs, routing_labels):
        print(f"  {row['bucket']:>12}  n={row['n']:5d} "
              f"({row['frac_of_test']*100:5.1f}%)  "
              f"mean_conf={row['mean_confidence']}  "
              f"acc={row['accuracy']}")

    print('\n=== Diagnostic 2: ECE split by hard_case vs clean (test.jsonl) ===')
    is_hard = torch.tensor([d['hard_case'] is not None for d in test_docs])
    n_hard = int(is_hard.sum())
    print(f'  hard_case docs: {n_hard} ({n_hard / len(test_docs) * 100:.1f}% '
          f'of test)')
    for name, mask in [('clean', ~is_hard), ('hard_case', is_hard)]:
        probs_m = routing_probs[mask]
        labels_m = routing_labels[mask]
        ece_m = expected_calibration_error(probs_m, labels_m)
        acc_m = (probs_m.argmax(-1) == labels_m).float().mean().item()
        mean_conf_m = probs_m.max(dim=-1).values.mean().item()
        print(f'  {name:>10}: n={int(mask.sum())}  acc={acc_m:.3f}  '
              f'mean_conf={mean_conf_m:.3f}  ece={ece_m:.3f}')


if __name__ == '__main__':
    main()
