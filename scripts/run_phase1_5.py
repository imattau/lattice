'''Run the Phase 1.5 toy-scale JEPA-vs-AR-vs-contrastive comparison
(PLAN.md "Phase 1.5"). See PHASE1_5_RESULTS.md for the honest scope
reduction from the plan's 10M-50M-param / 100M-1B-token / 1-3 PF-day spec
to what a single interactive session on one consumer GPU can actually run.

The contrastive arm (InfoNCE on augmented-view positive pairs) is a third,
matched-architecture/data/budget point that distinguishes two hypotheses
the two-arm comparison alone cannot: if contrastive beats AR where JEPA
lost, the problem is JEPA's specific objective, not "latent representation
learning" in general; if contrastive also loses, the gap is more likely
about token-level vs. latent-level training for language compositionality
at this scale, independent of which latent objective is used.
'''
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from lattice.phases.phase1_5_toy import (
    ToyConfig, build_composition_task, build_contrastive, build_jepa,
    composition_accuracy, effective_rank, encode_pooled_ar,
    encode_pooled_contrastive, encode_pooled_jepa, linear_probe_accuracy,
    train_ar, train_contrastive, train_jepa, uniformity,
)
from lattice.tiny_transformer import ARLanguageModel, TinyTransformer
from lattice.toy_corpus import (
    CharTokenizer, build_pretrain_corpus, load_banking77_labeled,
    load_triage_labeled,
)


def _label_to_index(labels: list) -> tuple[torch.Tensor, dict]:
    uniq = sorted(set(labels))
    idx = {v: i for i, v in enumerate(uniq)}
    return torch.tensor([idx[v] for v in labels]), idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--max_len', type=int, default=192)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default='PHASE1_5_RESULTS.json')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    config = ToyConfig(dim=args.dim, n_layers=args.n_layers,
                       n_heads=args.n_heads, max_len=args.max_len,
                       batch_size=args.batch_size, epochs=args.epochs,
                       lr=args.lr, seed=args.seed)

    print('Building offline toy corpus (code/json synthetic + '
         'BANKING77 + triage text) ...')
    pretrain_texts = build_pretrain_corpus(seed=args.seed)
    tokenizer = CharTokenizer.fit(pretrain_texts)
    pretrain_tokens = tokenizer.encode_batch(pretrain_texts, config.max_len)
    print(f'  {len(pretrain_texts)} documents, vocab_size={tokenizer.vocab_size}, '
         f'total chars fed to LM ~= {pretrain_tokens.numel()}')

    print(f'\nTraining AR baseline on {device} ...')
    ar_backbone = TinyTransformer(tokenizer.vocab_size, config.dim,
                                  config.n_layers, config.n_heads,
                                  config.max_len)
    ar_model = ARLanguageModel(ar_backbone)
    t0 = time.time()
    ar_history = train_ar(ar_model, pretrain_tokens, config, device)
    ar_seconds = time.time() - t0
    print(f'  final loss={ar_history[-1]:.4f}  '
         f'params={ar_model.num_params():,}  wall={ar_seconds:.1f}s')

    print(f'\nTraining JEPA (context+target encoder, predictor) on {device} ...')
    jepa = build_jepa(tokenizer.vocab_size, config)
    t0 = time.time()
    jepa_history = train_jepa(jepa, pretrain_tokens, config, device)
    jepa_seconds = time.time() - t0
    print(f'  final loss={jepa_history[-1]:.4f}  '
         f'trained_params={jepa.num_params():,}  wall={jepa_seconds:.1f}s')

    print(f'\nTraining contrastive (InfoNCE, augmented-view pairs) on '
         f'{device} ...')
    contrastive = build_contrastive(tokenizer.vocab_size, config)
    t0 = time.time()
    contrastive_history = train_contrastive(contrastive, pretrain_tokens,
                                            config, device)
    contrastive_seconds = time.time() - t0
    print(f'  final loss={contrastive_history[-1]:.4f}  '
         f'trained_params={contrastive.num_params():,}  '
         f'wall={contrastive_seconds:.1f}s')

    arms = [
        ('ar', encode_pooled_ar, ar_model),
        ('jepa', encode_pooled_jepa, jepa),
        ('contrastive', encode_pooled_contrastive, contrastive),
    ]

    print('\nLinear probes: BANKING77 intent (77-way), '
         'Phase 1 triage department (10-way) ...')
    probe_results = {}
    texts, labels = load_banking77_labeled(offset=4000, limit=3000)
    label_idx, label_map = _label_to_index(labels)
    n_train = int(0.8 * len(texts))
    tokens = tokenizer.encode_batch(texts, config.max_len)
    for model_name, encode_fn, model_obj in arms:
        feats = encode_fn(model_obj, tokens, device)
        acc = linear_probe_accuracy(
            feats[:n_train], label_idx[:n_train],
            feats[n_train:], label_idx[n_train:], len(label_map),
        )
        probe_results.setdefault('banking77_intent', {})[model_name] = acc

    texts, labels = load_triage_labeled(limit=3000)
    tokens = tokenizer.encode_batch(texts, config.max_len)
    labels_t = torch.tensor(labels)
    n_train = int(0.8 * len(texts))
    for model_name, encode_fn, model_obj in arms:
        feats = encode_fn(model_obj, tokens, device)
        acc = linear_probe_accuracy(
            feats[:n_train], labels_t[:n_train],
            feats[n_train:], labels_t[n_train:], 10,
        )
        probe_results.setdefault('triage_department', {})[model_name] = acc

    print('Compositional generalization (held-out color/shape pairings) ...')
    comp_results = {}
    split = build_composition_task(tokenizer, config.max_len)
    for model_name, encode_fn, model_obj in arms:
        train_feats = encode_fn(model_obj, split.train_tokens, device)
        test_feats = encode_fn(model_obj, split.test_tokens, device)
        comp_results[model_name] = composition_accuracy(
            train_feats, split, test_feats, epochs=300)

    print('Representation geometry (uniformity, effective rank) ...')
    geometry = {}
    probe_tokens_all = tokenizer.encode_batch(
        load_banking77_labeled(offset=4000, limit=1000)[0], config.max_len)
    for model_name, encode_fn, model_obj in arms:
        feats = encode_fn(model_obj, probe_tokens_all, device)
        geometry[model_name] = {
            'uniformity': uniformity(feats),
            'effective_rank': effective_rank(feats),
            'dim': feats.size(-1),
        }

    result = {
        'config': vars(args),
        'device': device,
        'vocab_size': tokenizer.vocab_size,
        'pretrain_corpus_size': len(pretrain_texts),
        'ar': {
            'num_params': ar_model.num_params(),
            'final_loss': ar_history[-1],
            'loss_history_every_10': ar_history[::10],
            'wall_seconds': ar_seconds,
        },
        'jepa': {
            'num_trained_params': jepa.num_params(),
            'final_loss': jepa_history[-1],
            'loss_history_every_10': jepa_history[::10],
            'wall_seconds': jepa_seconds,
        },
        'contrastive': {
            'num_trained_params': contrastive.num_params(),
            'final_loss': contrastive_history[-1],
            'loss_history_every_10': contrastive_history[::10],
            'wall_seconds': contrastive_seconds,
        },
        'linear_probe_accuracy': probe_results,
        'composition_task_accuracy': comp_results,
        'representation_geometry': geometry,
    }
    ckpt_path = Path(args.output).with_suffix('.pt')
    torch.save({
        'ar_backbone': ar_model.backbone.state_dict(),
        'jepa_context_encoder': jepa.context_encoder.state_dict(),
        'jepa_predictor': jepa.predictor.state_dict(),
        'contrastive_encoder': contrastive.encoder.state_dict(),
        'contrastive_projector': contrastive.projector.state_dict(),
        'tokenizer_chars': tokenizer.chars,
        'config': vars(args),
    }, ckpt_path)
    print(f'Saved model checkpoints to {ckpt_path} (for follow-up '
         f'diagnostics without retraining)')

    Path(args.output).write_text(json.dumps(result, indent=2))
    print('\n' + json.dumps({
        k: v for k, v in result.items()
        if k in ('linear_probe_accuracy', 'composition_task_accuracy',
                 'representation_geometry')
    }, indent=2))
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
