'''The experiment the readout-pyramid follow-up said matters most: re-run
compositional generalization on a task where no single token predicts the
answer, instead of the original task where color/shape words are literal
substrings of the input (solvable from layer 1 by every arm, including AR
at ceiling -- closer to lexical detection than composition).

`build_deep_composition_task` (lattice/phases/phase1_5_toy.py) uses
two-hop variable binding: every color and every shape word appears in
every example regardless of the answer, via a full link->value mapping
table; the correct answer requires resolving which link "x"/"y" points to
and then which value that link maps to *in this specific example*. A
bag-of-words read cannot shortcut this.

Reuses the four checkpoints already trained in PHASE1_5_RESULTS.pt (no
retraining) and probes every layer, exactly like the shallow-task
readout-pyramid probe, so the two are directly comparable.

`--style symbolic` (original) produces dense "x=p y=m p=red..." assignment
syntax; `--style natural` (default) produces ordinary declarative English
("p is red. q is blue. ... x is p. y is m.") with the same anti-shortcut
property, to separate "can the objective do two-hop binding" from "was
this exact syntax ever seen during pretraining" -- the confound the
symbolic-style run's floor result could not rule out.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_deep_composition_task, composition_accuracy, layerwise_pooled,
    load_arm_backbone,
)
from lattice.toy_corpus import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--style', choices=['symbolic', 'natural'], default='natural')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.output is None:
        suffix = '' if args.style == 'symbolic' else '_NATURAL'
        args.output = f'PHASE1_5_DEEP_COMPOSITION{suffix}_RESULTS.json'

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    split = build_deep_composition_task(tokenizer, config['max_len'],
                                        n_per_combo=args.n_per_combo,
                                        seed=args.seed, style=args.style)
    print(f'Deep composition task ({args.style}): '
         f'{split.train_tokens.size(0)} train, '
         f'{split.test_tokens.size(0)} test examples.')

    arms = [
        ('ar', 'ar_backbone', True),
        ('jepa', 'jepa_context_encoder', False),
        ('contrastive', 'contrastive_encoder', False),
        ('ar_jepa_aux', 'ar_jepa_backbone', True),
    ]

    results = {}
    for name, key, causal in arms:
        model = load_arm_backbone(ckpt, key, config, tokenizer.vocab_size)
        print(f'Probing {name} ({"causal" if causal else "bidirectional"} '
             f'read) across {config["n_layers"]} layers ...')

        train_layers = layerwise_pooled(model, split.train_tokens, causal)
        test_layers = layerwise_pooled(model, split.test_tokens, causal)

        per_layer = []
        for li in range(config['n_layers']):
            acc = composition_accuracy(train_layers[li], split, test_layers[li],
                                       epochs=300)
            per_layer.append(acc)
            print(f'  layer {li+1}/{config["n_layers"]}: '
                 f'joint={acc["joint_accuracy"]:.3f} '
                 f'(color={acc["color_accuracy"]:.3f}, '
                 f'shape={acc["shape_accuracy"]:.3f})')
        results[name] = {'causal_probe': causal, 'per_layer': per_layer}

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')

    print('\nSummary (deep composition joint accuracy): top vs. best layer')
    for name, data in results.items():
        joints = [layer['joint_accuracy'] for layer in data['per_layer']]
        top = joints[-1]
        best_idx = max(range(len(joints)), key=lambda i: joints[i])
        print(f'  {name:14s} top={top:.3f}  '
             f'best=layer_{best_idx+1}={joints[best_idx]:.3f}  '
             f'delta={joints[best_idx]-top:+.3f}')


if __name__ == '__main__':
    main()
