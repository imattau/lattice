'''Fine-tuned (not frozen) evaluation of AR+JEPA-auxiliary on the deep
(indirection) compositional task -- the "what comes after" item from the
COGS-replication follow-up: given that pure JEPA has no surviving
compositional-generalization win (PHASE1_5_COGS_REPLICATION_RESULTS.md),
and AR+JEPA-auxiliary is the one latent-touching arm with a genuine,
non-cherry-picked positive result, it's the more interesting arm to ask
"how learnable is the harder indirection task from these features, given
direct gradient signal, rather than only a frozen linear probe."

The indirection task floored at chance for every arm under a frozen probe,
in both symbolic and natural-language register
(PHASE1_5_DEEP_COMPOSITION_RESULTS.md) -- that result could not distinguish
"the representation doesn't linearly encode the answer" from "the model
cannot learn this relationship at all." Fine-tuning answers the second,
different question directly: unfreeze the encoder, train it jointly with
a small classification head on the task's own (tiny) training set, and see
whether direct gradient signal succeeds where a frozen probe could not.

Runs on AR+JEPA-auxiliary (as requested) and plain AR (as a reference
point -- a single arm's fine-tuned number is hard to interpret alone).
Reuses the checkpoints already trained in PHASE1_5_RESULTS.pt; the encoder
copy used for fine-tuning is unfrozen and trained fresh on the composition
task only (the loaded checkpoint itself is never mutated -- see
test_finetune_composition_accuracy_does_not_mutate_input_backbone).
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_deep_composition_task, finetune_composition_accuracy,
    load_arm_backbone,
)
from lattice.toy_corpus import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--style', choices=['symbolic', 'natural'], default='natural')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default='PHASE1_5_FINETUNE_INDIRECTION_RESULTS.json')
    args = p.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    split = build_deep_composition_task(tokenizer, config['max_len'],
                                        n_per_combo=args.n_per_combo,
                                        seed=args.seed, style=args.style)
    print(f'Deep composition task ({args.style}): '
         f'{split.train_tokens.size(0)} train, '
         f'{split.test_tokens.size(0)} test examples.\n'
         f'Fine-tuning for {args.epochs} epochs, lr={args.lr} on {device} '
         f'(full-batch AdamW, encoder unfrozen).')

    arms = [
        ('ar_jepa_aux', 'ar_jepa_backbone', True),
        ('ar', 'ar_backbone', True),
    ]

    results = {}
    for name, key, causal in arms:
        print(f'\n=== Fine-tuning {name} ===')
        backbone = load_arm_backbone(ckpt, key, config, tokenizer.vocab_size)
        result = finetune_composition_accuracy(
            backbone, split, causal=causal, epochs=args.epochs, lr=args.lr,
            device=device)
        results[name] = result
        print(f'  train: color={result["train"]["color_accuracy"]:.3f} '
             f'shape={result["train"]["shape_accuracy"]:.3f} '
             f'joint={result["train"]["joint_accuracy"]:.3f}')
        print(f'  test:  color={result["test"]["color_accuracy"]:.3f} '
             f'shape={result["test"]["shape_accuracy"]:.3f} '
             f'joint={result["test"]["joint_accuracy"]:.3f}')
        print(f'  loss: {result["loss_history_every_20"][0]:.3f} -> '
             f'{result["loss_history_every_20"][-1]:.3f}')

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')

    print('\n=== Summary (test joint accuracy; chance = 0.0625) ===')
    for name, result in results.items():
        gap = result['train']['joint_accuracy'] - result['test']['joint_accuracy']
        print(f'  {name:14s} train={result["train"]["joint_accuracy"]:.3f}  '
             f'test={result["test"]["joint_accuracy"]:.3f}  '
             f'train-test gap={gap:+.3f}')


if __name__ == '__main__':
    main()
