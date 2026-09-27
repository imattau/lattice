'''COGS-style compositional generalization probe (primitives in the
pretraining corpus, only the combination held out) -- the experiment
recommended to run before either retraining on indirection-style text or
fine-tuning: it cleanly separates "did the model learn to combine known
primitives in a new way" from "did the model ever see this kind of thing
at all," which the deep (indirection) task's floor result could not do in
either the symbolic or natural-language register (see
PHASE1_5_DEEP_COMPOSITION_RESULTS.md).

`build_cogs_style_task` (lattice/phases/phase1_5_toy.py) uses department
names and urgency phrases that are verifiably present, verbatim, in the
actual Phase 1 triage documents sampled into the pretraining corpus for
these four checkpoints -- not asserted, checked
(tests/test_phase1_5_toy.py::test_cogs_task_primitives_are_present_in_the_actual_pretraining_corpus).
Only the specific (department, urgency-phrase) *pairing*, in a sentence
frame that itself never appears verbatim in pretraining, is held out.

Reuses the four checkpoints already trained in PHASE1_5_RESULTS.pt (no
retraining) and probes every layer, same protocol as the other
compositional-generalization follow-ups.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_cogs_style_task, composition_accuracy, layerwise_pooled,
    load_arm_backbone,
)
from lattice.toy_corpus import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='PHASE1_5_COGS_STYLE_RESULTS.json')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    split = build_cogs_style_task(tokenizer, config['max_len'],
                                  n_per_combo=args.n_per_combo,
                                  seed=args.seed)
    print(f'COGS-style task: {split.train_tokens.size(0)} train, '
         f'{split.test_tokens.size(0)} test examples (primitives seen '
         f'during pretraining; only the pairing is held out).')

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
                 f'(dept={acc["color_accuracy"]:.3f}, '
                 f'urgency={acc["shape_accuracy"]:.3f})')
        results[name] = {'causal_probe': causal, 'per_layer': per_layer}

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')

    print('\nSummary (COGS-style joint accuracy): top vs. best layer')
    for name, data in results.items():
        joints = [layer['joint_accuracy'] for layer in data['per_layer']]
        top = joints[-1]
        best_idx = max(range(len(joints)), key=lambda i: joints[i])
        print(f'  {name:14s} top={top:.3f}  '
             f'best=layer_{best_idx+1}={joints[best_idx]:.3f}  '
             f'delta={joints[best_idx]-top:+.3f}')


if __name__ == '__main__':
    main()
