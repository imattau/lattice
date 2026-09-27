'''Does an explicit VSA binding/retrieval mechanism help on a
compositional-generalization task -- either the indirection task every
other Core arm failed at, or the COGS-style task where real, non-floor
signal exists?

PHASE1_5_FINETUNE_INDIRECTION_RESULTS.md: fine-tuning AR and AR+JEPA-aux
directly on the deep (two-hop indirection) task's 192 training examples
drove both to perfect (1.000) training accuracy and chance-or-below test
accuracy -- textbook memorization. VSA_INDIRECTION_RESULTS.md found the
same failure mode for a VSA-augmented encoder on that task. Since the
indirection task gives *every* architecture zero pretraining exposure to
its structure (a confound documented in
PHASE1_5_DEEP_COMPOSITION_RESULTS.md), `--task cogs`/`cogs_v2` reruns the
same from-scratch comparison on the COGS-style department/urgency (or
JSON name/status) task instead, where frozen-probe evaluation of
*pretrained* encoders found real, well-above-chance signal
(PHASE1_5_COGS_STYLE_RESULTS.md). This trains from scratch here too (no
separate pretraining phase for either task) -- the point of `--task cogs`
is to test the same VSA-vs-plain comparison, same from-scratch protocol,
on a task with a real underlying pattern to find, not to reproduce the
earlier frozen-probe pretrained-checkpoint result.

This trains, from scratch, a `VSAEncoder` (TinyTransformer + an explicit
Holographic-Reduced-Representation bind/unbind memory module,
lattice/vsa.py) and a size-matched plain `BidirectionalPooledBackbone`,
both directly on the chosen task's train split, and compares test
accuracy -- the same train/test overfitting-gap framing as the earlier
fine-tuned-probe experiments.

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
    build_cogs_style_task, build_cogs_style_task_v2,
    build_deep_composition_task, finetune_pooled_model,
)
from lattice.toy_corpus import CharTokenizer, build_pretrain_corpus
from lattice.vsa import BidirectionalPooledBackbone, VSAEncoder


def _build_split(task: str, tokenizer, args):
    if task == 'indirection':
        return build_deep_composition_task(
            tokenizer, args.max_len, n_per_combo=args.n_per_combo,
            seed=args.seed, style=args.style)
    if task == 'cogs':
        return build_cogs_style_task(tokenizer, args.max_len,
                                     n_per_combo=args.n_per_combo,
                                     seed=args.seed)
    if task == 'cogs_v2':
        return build_cogs_style_task_v2(tokenizer, args.max_len,
                                        n_per_combo=args.n_per_combo,
                                        seed=args.seed)
    raise ValueError(f'unknown task: {task}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['indirection', 'cogs', 'cogs_v2'],
                   default='cogs')
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--style', choices=['symbolic', 'natural'], default='natural',
                   help='only used for --task indirection')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--max_len', type=int, default=192)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = f'VSA_{args.task.upper()}_RESULTS.json'

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # A tokenizer needs *some* corpus to build its char vocabulary from;
    # reuse the standard toy pretrain corpus purely as a source of
    # characters (not for pretraining -- both models below are trained
    # from scratch, directly on the chosen task only).
    vocab_source = build_pretrain_corpus(seed=args.seed, n_code=200,
                                         n_json=200, n_banking=200,
                                         n_triage=200)
    tokenizer = CharTokenizer.fit(vocab_source)
    split = _build_split(args.task, tokenizer, args)
    print(f'Task: {args.task}  {split.train_tokens.size(0)} train, '
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
