'''Run the Phase 1 pipeline: encode the synthetic triage corpus with the
Phase-0-selected frozen encoder, train the readout bank (Stage 1a), tune
the controller's escalation threshold (Stage 1b), fit per-head post-hoc
temperature (Stage 1c), and evaluate on test/gold/redteam splits.

Generate the corpus first with scripts/generate_triage_corpus.py.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.encoders import FrozenEncoder
from lattice.phases.phase1_triage import (
    build_bank, build_constraints, evaluate_heads, evaluate_triage,
    fit_head_temperatures, labels_from_docs, load_jsonl, train_bank,
    tune_threshold, TriageConfig,
)


def _encode_split(encoder: FrozenEncoder, docs: list[dict],
                  cache_dir: Path, name: str) -> torch.Tensor:
    cache_path = cache_dir / f'{name}.pt'
    if cache_path.exists():
        return torch.load(cache_path)
    features = encoder.encode([d['text'] for d in docs])
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    return features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/triage')
    p.add_argument('--encoder', default='BAAI/bge-small-en-v1.5',
                    help='Phase 0 decision: latency-gated frozen encoder')
    p.add_argument('--cache_dir', default='data/triage/features_cache')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='PHASE1_TRAINING_RESULTS.json')
    args = p.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    splits = {
        name: load_jsonl(data_dir / f'{name}.jsonl')
        for name in ('train', 'val', 'test', 'gold_validation', 'redteam')
    }

    print(f'Encoding splits with {args.encoder} ...')
    encoder = FrozenEncoder(args.encoder)
    cache_dir = Path(args.cache_dir)
    features = {
        name: _encode_split(encoder, docs, cache_dir, name)
        for name, docs in splits.items()
    }
    labels = {
        name: labels_from_docs(docs) for name, docs in splits.items()
    }

    print(f'Training readout bank (Stage 1a) for {args.epochs} epochs ...')
    bank = build_bank(encoder.hidden_dim)
    history = train_bank(bank, features['train'], labels['train'],
                         epochs=args.epochs, lr=args.lr, beta_rlcd=args.beta)

    print('Tuning escalation threshold on val (Stage 1b) ...')
    constraints = build_constraints()
    best_threshold = tune_threshold(bank, features['val'], splits['val'],
                                    constraints)
    config = TriageConfig(escalation_confidence_threshold=best_threshold)

    print('Fitting per-head post-hoc temperature on val (Stage 1c) ...')
    head_temps = fit_head_temperatures(bank, features['val'], labels['val'])

    print('Evaluating heads on test ...')
    head_metrics = {
        name: evaluate_heads(bank, features[name], labels[name])
        for name in ('test', 'gold_validation')
    }

    print('Evaluating end-to-end controller on test and redteam ...')
    triage_metrics = {
        name: evaluate_triage(bank, features[name], splits[name], config,
                              constraints)
        for name in ('test', 'gold_validation', 'redteam')
    }

    result = {
        'encoder': args.encoder,
        'epochs': args.epochs,
        'final_train_loss': history[-1],
        'escalation_confidence_threshold': best_threshold,
        'post_hoc_head_temperatures': head_temps,
        'policy_hash': constraints.policy_hash,
        'head_metrics': head_metrics,
        'triage_metrics': triage_metrics,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
