'''Three-encoder comparison on Phase 0.

Answers the blocking question for Phase 1: is the 0.62 accuracy
encoder-limited or head-limited? We hold the readout head, splits, and
calibration fixed and vary only the frozen encoder. Feature tensors are
cached to disk so re-runs are cheap.
'''
from __future__ import annotations
import argparse

import torch
import torch.nn.functional as F

from lattice.calibration import fit_temperature
from lattice.phases.phase0 import (
    _evaluate, _measure_latency, _split_indices, encode_cached,
    load_banking77, train_readout,
)

DEFAULT_ENCODERS = [
    'distilbert-base-uncased',        # baseline that produced 0.62
    'microsoft/deberta-v3-base',      # stronger frozen encoder
    'Qwen/Qwen3-0.6B',                # small LM repurposed as encoder
]


def class_prior_brier(labels: torch.Tensor, num_classes: int) -> float:
    prior = torch.bincount(labels, minlength=num_classes).float()
    prior = prior / prior.sum()
    onehot = F.one_hot(labels, num_classes).float()
    return float(((prior - onehot) ** 2).sum(dim=-1).mean())


def run_one(encoder: str, texts: list[str], labels_all: torch.Tensor,
            num_classes: int, split, device: str, epochs: int,
            lr: float, beta: float) -> dict:
    feats = encode_cached(encoder, texts, device=device)
    train_idx, cal_idx, test_idx = split
    Xtr, ytr = feats[train_idx], labels_all[train_idx]
    Xcal, ycal = feats[cal_idx], labels_all[cal_idx]
    Xte, yte = feats[test_idx], labels_all[test_idx]

    head = train_readout(Xtr, ytr, num_classes, epochs=epochs, lr=lr, beta=beta)
    head.to(device)
    with torch.no_grad():
        cal_logits = head.logit_head(Xcal.to(device))
    temperature = fit_temperature(cal_logits, ycal.to(device))
    metrics = _evaluate(head, Xte.to(device), yte.to(device), num_classes,
                        temperature=temperature)
    metrics['latency_ms'] = _measure_latency(head, Xte, n=200)
    metrics['dim'] = feats.size(-1)
    metrics['temperature'] = temperature
    return metrics


def main():
    p = argparse.ArgumentParser(description='Phase 0 encoder comparison')
    p.add_argument('--encoders', nargs='+', default=DEFAULT_ENCODERS)
    p.add_argument('--data', default='data/banking77.csv')
    p.add_argument('--device', default=None)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels, label_names = load_banking77(args.data)
    num_classes = len(label_names)
    labels_all = torch.tensor(labels, dtype=torch.long)
    split = _split_indices(len(texts), 0.70, 0.15, args.seed)
    prior_brier = class_prior_brier(labels_all[split[2]], num_classes)

    header = (f'{"encoder":<32}{"dim":>6}{"acc":>8}{"ece_raw":>9}'
              f'{"ece_cal":>9}{"brier":>8}{"bss":>8}{"ms":>8}')
    print(header)
    print('-' * len(header))
    rows = []
    for enc in args.encoders:
        m = run_one(enc, texts, labels_all, num_classes, split, device,
                    args.epochs, args.lr, args.beta)
        bss = 1 - m['brier'] / prior_brier  # Brier skill vs class prior
        rows.append((enc, m, bss))
        print(f'{enc:<32}{m["dim"]:>6}{m["accuracy"]:>8.3f}'
              f'{m["ece_raw"]:>9.3f}{m["ece_cal"]:>9.3f}{m["brier"]:>8.3f}'
              f'{bss:>8.3f}{m["latency_ms"]:>8.3f}')

    print('\n(bss = Brier skill score vs class-prior baseline; '
          f'prior Brier = {prior_brier:.3f})')
    best = max(rows, key=lambda r: r[1]['accuracy'])
    print(f'Best accuracy: {best[0]} @ {best[1]["accuracy"]:.3f}')


if __name__ == '__main__':
    main()
