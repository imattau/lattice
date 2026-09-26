'''Does an explicit VSA binding/retrieval mechanism help on the exact
failure mode every other Core arm hit in this project?

PHASE1_5_FINETUNE_INDIRECTION_RESULTS.md: fine-tuning AR and AR+JEPA-aux
directly on the deep (two-hop indirection) task's 192 training examples
drove both to perfect (1.000) training accuracy and chance-or-below test
accuracy (0.000, 0.016) -- textbook memorization, identical for both
objectives. That result motivated exploring architectures with an
explicit binding primitive rather than another loss-function variant on
the same plain Transformer (see the "explore alternatives" / "go far
afield" discussion this script follows from).

This trains, from scratch (no pretraining -- neither arm has ever seen
anything resembling this text, so pretraining exposure isn't the
variable under test here), a `VSAEncoder` (TinyTransformer + an explicit
Holographic-Reduced-Representation bind/unbind memory module,
lattice/vsa.py) and a size-matched plain `BidirectionalPooledBackbone`,
both directly on the indirection task's train split, and compares test
accuracy -- the same train/test overfitting-gap framing as the earlier
fine-tuned-probe experiment.

IMPORTANT ASYMMETRY: VSAEncoder has more trainable parameters than the
plain backbone (the memory module's key/value/query/out projections add
capacity) -- verified by
tests/test_vsa.py::test_vsa_encoder_has_more_params_than_matched_plain_backbone.
Any advantage for VSAEncoder could in principle come from extra capacity
rather than the binding mechanism specifically; report both models'
parameter counts alongside their results so this isn't glossed over.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_deep_composition_task, finetune_pooled_model,
)
from lattice.toy_corpus import CharTokenizer, build_pretrain_corpus
from lattice.vsa import BidirectionalPooledBackbone, VSAEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--style', choices=['symbolic', 'natural'], default='natural')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default='VSA_INDIRECTION_RESULTS.json')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # A tokenizer needs *some* corpus to build its char vocabulary from;
    # reuse the standard toy pretrain corpus purely as a source of
    # characters (not for pretraining -- both models below are trained
    # from scratch, directly on the indirection task only).
    vocab_source = build_pretrain_corpus(seed=args.seed, n_code=200,
                                         n_json=200, n_banking=200,
                                         n_triage=200)
    tokenizer = CharTokenizer.fit(vocab_source)
    split = build_deep_composition_task(tokenizer, args.max_len,
                                        n_per_combo=args.n_per_combo,
                                        seed=args.seed, style=args.style)
    print(f'Deep composition task ({args.style}): '
         f'{split.train_tokens.size(0)} train, '
         f'{split.test_tokens.size(0)} test examples.')
    print(f'Training from scratch for {args.epochs} epochs, lr={args.lr}, '
         f'on {device}.\n')

    results = {}
    for name, build_fn in [
        ('vsa_encoder', lambda: VSAEncoder(
            tokenizer.vocab_size, args.dim, args.n_layers, args.n_heads,
            args.max_len)),
        ('plain_backbone', lambda: BidirectionalPooledBackbone(
            tokenizer.vocab_size, args.dim, args.n_layers, args.n_heads,
            args.max_len)),
    ]:
        torch.manual_seed(args.seed)
        model = build_fn()
        print(f'=== {name} ({model.num_params():,} params) ===')
        result = finetune_pooled_model(model, split, epochs=args.epochs,
                                       lr=args.lr, device=device)
        results[name] = result
        print(f'  train: color={result["train"]["color_accuracy"]:.3f} '
             f'shape={result["train"]["shape_accuracy"]:.3f} '
             f'joint={result["train"]["joint_accuracy"]:.3f}')
        print(f'  test:  color={result["test"]["color_accuracy"]:.3f} '
             f'shape={result["test"]["shape_accuracy"]:.3f} '
             f'joint={result["test"]["joint_accuracy"]:.3f}')
        print(f'  loss: {result["loss_history_every_20"][0]:.3f} -> '
             f'{result["loss_history_every_20"][-1]:.3f}\n')

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'Wrote {args.output}')

    print('\n=== Summary (test joint accuracy; chance = 0.0625) ===')
    for name, result in results.items():
        gap = result['train']['joint_accuracy'] - result['test']['joint_accuracy']
        print(f'  {name:16s} params={result["num_params"]:>8,}  '
             f'train={result["train"]["joint_accuracy"]:.3f}  '
             f'test={result["test"]["joint_accuracy"]:.3f}  '
             f'train-test gap={gap:+.3f}')


if __name__ == '__main__':
    main()
