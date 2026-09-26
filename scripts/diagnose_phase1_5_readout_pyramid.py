'''Test B from the encoder/decoder-split follow-up: does concatenating
features from multiple depths (a readout pyramid, standard in vision since
FPN) beat reading from any single layer -- including the best single layer
found by the earlier per-layer probe?

Reuses the four checkpoints already trained in PHASE1_5_RESULTS.pt (no
retraining). For each arm, compares three readouts on the same probe,
computed in the same run so the comparison isn't confounded by probe-init
noise across separate script invocations:
  - top layer only (what every PHASE1_5_RESULTS.md number used)
  - best single layer (found by scanning all layers)
  - pyramid: layer 1 + a middle layer + the top layer, concatenated

Test A from the same follow-up (moving the Talker's generation read-point)
is NOT run here: this toy harness never built a Talker/decoder component
(Phase 1.5 only trains encoders + linear probes, no generation path), so
there is nothing to move the read-point of yet. Building a toy
reconstruction decoder to test that would be a separate, larger piece of
work than this script, not a "one-line change on existing checkpoints" in
this codebase specifically -- noted here rather than silently skipped.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_composition_task, composition_accuracy, layerwise_pooled,
    linear_probe_accuracy, load_arm_backbone,
)
from lattice.toy_corpus import (
    CharTokenizer, load_banking77_labeled, load_triage_labeled,
)


def _label_to_index(labels):
    uniq = sorted(set(labels))
    idx = {v: i for i, v in enumerate(uniq)}
    return torch.tensor([idx[v] for v in labels]), len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--output', default='PHASE1_5_READOUT_PYRAMID_RESULTS.json')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    n_layers = config['n_layers']
    mid = n_layers // 2

    split = build_composition_task(tokenizer, config['max_len'])
    banking_texts, banking_labels_raw = load_banking77_labeled(offset=4000,
                                                                limit=3000)
    banking_labels, n_banking_classes = _label_to_index(banking_labels_raw)
    banking_tokens = tokenizer.encode_batch(banking_texts, config['max_len'])
    n_banking_train = int(0.8 * len(banking_texts))

    triage_texts, triage_labels_raw = load_triage_labeled(limit=3000)
    triage_labels = torch.tensor(triage_labels_raw)
    triage_tokens = tokenizer.encode_batch(triage_texts, config['max_len'])
    n_triage_train = int(0.8 * len(triage_texts))

    arms = [
        ('ar', 'ar_backbone', True),
        ('jepa', 'jepa_context_encoder', False),
        ('contrastive', 'contrastive_encoder', False),
        ('ar_jepa_aux', 'ar_jepa_backbone', True),
    ]

    results = {}
    for name, key, causal in arms:
        model = load_arm_backbone(ckpt, key, config, tokenizer.vocab_size)
        print(f'Probing {name} ({"causal" if causal else "bidirectional"} read) ...')

        train_layers = layerwise_pooled(model, split.train_tokens, causal)
        test_layers = layerwise_pooled(model, split.test_tokens, causal)
        banking_layers = layerwise_pooled(model, banking_tokens, causal)
        triage_layers = layerwise_pooled(model, triage_tokens, causal)

        def pyramid(layers, indices):
            return torch.cat([layers[i] for i in indices], dim=-1)

        pyramid_idx = sorted({0, mid, n_layers - 1})

        # Composition task: top-only, every single layer (to find "best"),
        # and the pyramid, all computed in this one run.
        comp_per_layer = [
            composition_accuracy(train_layers[li], split, test_layers[li],
                                 epochs=300)['joint_accuracy']
            for li in range(n_layers)
        ]
        comp_pyramid = composition_accuracy(
            pyramid(train_layers, pyramid_idx), split,
            pyramid(test_layers, pyramid_idx), epochs=300)['joint_accuracy']

        banking_per_layer = [
            linear_probe_accuracy(
                banking_layers[li][:n_banking_train],
                banking_labels[:n_banking_train],
                banking_layers[li][n_banking_train:],
                banking_labels[n_banking_train:], n_banking_classes)
            for li in range(n_layers)
        ]
        banking_pyramid = linear_probe_accuracy(
            pyramid(banking_layers, pyramid_idx)[:n_banking_train],
            banking_labels[:n_banking_train],
            pyramid(banking_layers, pyramid_idx)[n_banking_train:],
            banking_labels[n_banking_train:], n_banking_classes)

        triage_per_layer = [
            linear_probe_accuracy(
                triage_layers[li][:n_triage_train],
                triage_labels[:n_triage_train],
                triage_layers[li][n_triage_train:],
                triage_labels[n_triage_train:], 10)
            for li in range(n_layers)
        ]
        triage_pyramid = linear_probe_accuracy(
            pyramid(triage_layers, pyramid_idx)[:n_triage_train],
            triage_labels[:n_triage_train],
            pyramid(triage_layers, pyramid_idx)[n_triage_train:],
            triage_labels[n_triage_train:], 10)

        def summarize(per_layer, pyramid_val, label):
            top = per_layer[-1]
            best_idx = max(range(len(per_layer)), key=lambda i: per_layer[i])
            print(f'  {label:16s} top={top:.3f}  '
                 f'best=layer_{best_idx+1}={per_layer[best_idx]:.3f}  '
                 f'pyramid(layers {[i+1 for i in pyramid_idx]})={pyramid_val:.3f}')
            return {'top': top, 'best_layer': best_idx + 1,
                    'best': per_layer[best_idx], 'pyramid': pyramid_val,
                    'per_layer': per_layer}

        results[name] = {
            'pyramid_layers': [i + 1 for i in pyramid_idx],
            'composition_joint_accuracy': summarize(
                comp_per_layer, comp_pyramid, 'composition'),
            'banking77_intent_accuracy': summarize(
                banking_per_layer, banking_pyramid, 'banking77'),
            'triage_department_accuracy': summarize(
                triage_per_layer, triage_pyramid, 'triage'),
        }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
