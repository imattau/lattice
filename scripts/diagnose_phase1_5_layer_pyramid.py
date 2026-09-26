'''Readout-pyramid probe: does the toy Phase 1.5 compositional-generalization
gap reflect a missing signal, or a signal sitting at the wrong depth?

BERT-style models are known to stratify hierarchically (POS low, parsing
mid, semantics high) without explicit layer-wise supervision, and the top
layer is specialized for the training objective rather than necessarily
being the best layer for a downstream probe. The Phase 1.5 evaluation so
far has only ever probed the *top* layer's pooled output. This script
reuses the checkpoints already trained in PHASE1_5_RESULTS.pt (no
retraining) and trains the same compositional-generalization probe
(lattice.phases.phase1_5_toy.composition_accuracy) on pooled features from
every layer of each of the four arms.

Reading the result:
  - If a middle layer clearly beats the top layer for JEPA/contrastive/
    AR+JEPA-aux: the compositional signal exists but the readout was
    asking the wrong depth -- a much cheaper fix (a readout pyramid,
    concatenating features from multiple depths) than a training-objective
    or architecture change.
  - If all layers are roughly tied (near the top-layer number already
    reported in PHASE1_5_RESULTS.md): the signal isn't sitting anywhere in
    the stack at this scale, and the gap is upstream of the readout.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    build_composition_task, composition_accuracy, linear_probe_accuracy,
)
from lattice.tiny_transformer import TinyTransformer
from lattice.toy_corpus import (
    CharTokenizer, load_banking77_labeled, load_triage_labeled,
)


def _load_backbone(ckpt: dict, key: str, config: dict, vocab_size: int
                   ) -> TinyTransformer:
    model = TinyTransformer(vocab_size, config['dim'], config['n_layers'],
                            config['n_heads'], config['max_len'])
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def _layerwise_pooled(model: TinyTransformer, tokens: torch.Tensor,
                      causal: bool, batch_size: int = 256) -> list[torch.Tensor]:
    '''Returns a list of [N, D] pooled features, one per layer.'''
    n_layers = len(model.blocks)
    per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size]
            hiddens = model.forward_layers(batch, causal=causal)
            for li, h in enumerate(hiddens):
                per_layer[li].append(h.mean(dim=1))
    return [torch.cat(chunks, dim=0) for chunks in per_layer]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='PHASE1_5_RESULTS.pt')
    p.add_argument('--output', default='PHASE1_5_LAYER_PYRAMID_RESULTS.json')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
    split = build_composition_task(tokenizer, config['max_len'])

    def _label_to_index(labels):
        uniq = sorted(set(labels))
        idx = {v: i for i, v in enumerate(uniq)}
        return torch.tensor([idx[v] for v in labels]), len(idx)

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
        ('ar', 'ar_backbone', False),
        ('jepa', 'jepa_context_encoder', False),
        ('contrastive', 'contrastive_encoder', False),
        ('ar_jepa_aux', 'ar_jepa_backbone', False),
    ]
    # AR-family arms were trained and evaluated causally elsewhere in
    # PHASE1_5_RESULTS.md; probe them the same way here for consistency,
    # and also non-causally so a middle-layer bidirectional read is on the
    # table for every arm, not just the two that were bidirectional to
    # start with.
    causal_variants = {'ar': True, 'ar_jepa_aux': True}

    results = {}
    for name, key, _ in arms:
        model = _load_backbone(ckpt, key, config, tokenizer.vocab_size)
        causal = causal_variants.get(name, False)
        print(f'Probing {name} ({"causal" if causal else "bidirectional"} '
             f'read) across {config["n_layers"]} layers ...')

        train_layers = _layerwise_pooled(model, split.train_tokens, causal)
        test_layers = _layerwise_pooled(model, split.test_tokens, causal)
        banking_layers = _layerwise_pooled(model, banking_tokens, causal)
        triage_layers = _layerwise_pooled(model, triage_tokens, causal)

        per_layer = []
        for li in range(config['n_layers']):
            comp_acc = composition_accuracy(train_layers[li], split,
                                            test_layers[li], epochs=300)
            banking_acc = linear_probe_accuracy(
                banking_layers[li][:n_banking_train],
                banking_labels[:n_banking_train],
                banking_layers[li][n_banking_train:],
                banking_labels[n_banking_train:], n_banking_classes)
            triage_acc = linear_probe_accuracy(
                triage_layers[li][:n_triage_train],
                triage_labels[:n_triage_train],
                triage_layers[li][n_triage_train:],
                triage_labels[n_triage_train:], 10)
            per_layer.append({
                'composition_joint_accuracy': comp_acc['joint_accuracy'],
                'composition_color_accuracy': comp_acc['color_accuracy'],
                'composition_shape_accuracy': comp_acc['shape_accuracy'],
                'banking77_intent_accuracy': banking_acc,
                'triage_department_accuracy': triage_acc,
            })
            print(f'  layer {li+1}/{config["n_layers"]}: '
                 f'composition_joint={comp_acc["joint_accuracy"]:.3f}  '
                 f'banking77={banking_acc:.3f}  triage={triage_acc:.3f}')
        results[name] = {'causal_probe': causal, 'per_layer': per_layer}

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.output}')

    for metric in ('composition_joint_accuracy', 'banking77_intent_accuracy',
                  'triage_department_accuracy'):
        print(f'\nSummary ({metric}): top layer vs. best layer')
        for name, data in results.items():
            vals = [layer[metric] for layer in data['per_layer']]
            top = vals[-1]
            best_idx = max(range(len(vals)), key=lambda i: vals[i])
            print(f'  {name:14s} top={top:.3f}  '
                 f'best=layer_{best_idx+1}={vals[best_idx]:.3f}  '
                 f'delta={vals[best_idx]-top:+.3f}')


if __name__ == '__main__':
    main()
