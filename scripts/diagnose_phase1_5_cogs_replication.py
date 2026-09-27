'''Two cheap follow-ups to the COGS-style result (JEPA's best layer beat
AR's, 0.922 vs 0.859, on department/urgency recombination):

1. **Replication** on a second COGS-style task with different primitives
   from a different pretraining-corpus tier (JSON `name`/`status` values,
   vs. the first task's triage-derived department/urgency values). If
   JEPA > AR again, the finding isn't specific to one construction.

2. **Held-out layer selection.** "Best layer" in every prior follow-up was
   selected by scanning all layers *on the task's own test set* -- a mild
   selection-bias risk (the max over several noisy layers is upward-biased
   for whichever arm you apply it to, but if the sweep interacts with the
   task this could still favor one arm unfairly). This picks each arm's
   layer using a *different* task's per-layer accuracy (BANKING77 intent,
   already probed in PHASE1_5_LAYER_PYRAMID_RESULTS.json) as a proxy
   validation signal, then evaluates the COGS-style tasks at that
   independently-chosen layer -- removing the "the layer was chosen using
   the same numbers being reported" objection.

Reuses the four checkpoints already trained in PHASE1_5_RESULTS.pt (no
retraining).
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_cogs_style_task, build_cogs_style_task_v2, composition_accuracy,
    layerwise_pooled, linear_probe_accuracy, load_arm_backbone,
)
from lattice.toy_corpus import CharTokenizer, load_banking77_labeled


def _label_to_index(labels):
    uniq = sorted(set(labels))
    idx = {v: i for i, v in enumerate(uniq)}
    return torch.tensor([idx[v] for v in labels]), len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--n_per_combo', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='PHASE1_5_COGS_REPLICATION_RESULTS.json')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    n_layers = config['n_layers']

    task1 = build_cogs_style_task(tokenizer, config['max_len'],
                                  n_per_combo=args.n_per_combo, seed=args.seed)
    task2 = build_cogs_style_task_v2(tokenizer, config['max_len'],
                                     n_per_combo=args.n_per_combo,
                                     seed=args.seed)

    banking_texts, banking_labels_raw = load_banking77_labeled(offset=4000,
                                                                limit=3000)
    banking_labels, n_banking_classes = _label_to_index(banking_labels_raw)
    banking_tokens = tokenizer.encode_batch(banking_texts, config['max_len'])
    n_banking_train = int(0.8 * len(banking_texts))

    arms = [
        ('ar', 'ar_backbone', True),
        ('jepa', 'jepa_context_encoder', False),
        ('contrastive', 'contrastive_encoder', False),
        ('ar_jepa_aux', 'ar_jepa_backbone', True),
    ]

    results = {}
    for name, key, causal in arms:
        model = load_arm_backbone(ckpt, key, config, tokenizer.vocab_size)
        print(f'\n=== {name} ({"causal" if causal else "bidirectional"} read) ===')

        t1_train = layerwise_pooled(model, task1.train_tokens, causal)
        t1_test = layerwise_pooled(model, task1.test_tokens, causal)
        t2_train = layerwise_pooled(model, task2.train_tokens, causal)
        t2_test = layerwise_pooled(model, task2.test_tokens, causal)
        bank_layers = layerwise_pooled(model, banking_tokens, causal)

        t1_joint, t2_joint, bank_acc = [], [], []
        for li in range(n_layers):
            t1_joint.append(composition_accuracy(
                t1_train[li], task1, t1_test[li], epochs=300)['joint_accuracy'])
            t2_joint.append(composition_accuracy(
                t2_train[li], task2, t2_test[li], epochs=300)['joint_accuracy'])
            bank_acc.append(linear_probe_accuracy(
                bank_layers[li][:n_banking_train],
                banking_labels[:n_banking_train],
                bank_layers[li][n_banking_train:],
                banking_labels[n_banking_train:], n_banking_classes))
            print(f'  layer {li+1}/{n_layers}: task1_joint={t1_joint[-1]:.3f} '
                 f'task2_joint={t2_joint[-1]:.3f}  banking77={bank_acc[-1]:.3f}')

        # "Best own layer" (selection-biased: chosen using the task's own
        # test-set numbers) vs. "held-out-selected layer" (chosen using
        # BANKING77's per-layer numbers, a different task, then applied to
        # the COGS tasks -- removes the same-numbers-twice objection).
        t1_best_idx = max(range(n_layers), key=lambda i: t1_joint[i])
        t2_best_idx = max(range(n_layers), key=lambda i: t2_joint[i])
        bank_best_idx = max(range(n_layers), key=lambda i: bank_acc[i])

        results[name] = {
            'task1_department_urgency': {
                'top_layer': t1_joint[-1],
                'own_best_layer': t1_best_idx + 1,
                'own_best_value': t1_joint[t1_best_idx],
                'heldout_selected_layer': bank_best_idx + 1,
                'heldout_selected_value': t1_joint[bank_best_idx],
                'per_layer': t1_joint,
            },
            'task2_json_name_status': {
                'top_layer': t2_joint[-1],
                'own_best_layer': t2_best_idx + 1,
                'own_best_value': t2_joint[t2_best_idx],
                'heldout_selected_layer': bank_best_idx + 1,
                'heldout_selected_value': t2_joint[bank_best_idx],
                'per_layer': t2_joint,
            },
            'banking77_per_layer': bank_acc,
        }
        print(f'  task1: top={t1_joint[-1]:.3f}  '
             f'own_best=layer_{t1_best_idx+1}={t1_joint[t1_best_idx]:.3f}  '
             f'heldout_selected(layer_{bank_best_idx+1})={t1_joint[bank_best_idx]:.3f}')
        print(f'  task2: top={t2_joint[-1]:.3f}  '
             f'own_best=layer_{t2_best_idx+1}={t2_joint[t2_best_idx]:.3f}  '
             f'heldout_selected(layer_{bank_best_idx+1})={t2_joint[bank_best_idx]:.3f}')

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')

    print('\n=== Summary ===')
    for task_key, label in [('task1_department_urgency', 'Task 1 (dept/urgency)'),
                            ('task2_json_name_status', 'Task 2 (json name/status)')]:
        print(f'\n{label}:')
        for name in results:
            d = results[name][task_key]
            print(f'  {name:14s} top={d["top_layer"]:.3f}  '
                 f'own_best={d["own_best_value"]:.3f}  '
                 f'heldout_selected={d["heldout_selected_value"]:.3f}')


if __name__ == '__main__':
    main()
