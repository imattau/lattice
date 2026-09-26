'''Phase 1.5: toy-scale JEPA-vs-AR falsification (PLAN.md "Phase 1.5:
Toy-Scale Falsification").

Trains two architecturally-identical TinyTransformer backbones — one with
next-token cross-entropy (AR baseline), one with a JEPA latent-prediction
objective (context encoder + EMA target encoder + predictor, multi-block
masking, cosine loss) — at matched depth/width/heads/context-length/
optimizer/batch-size/step-count, then compares them on: linear-probe
accuracy on two real offline classification tasks (BANKING77 intent,
Phase 1 triage department routing), a synthetic compositional-
generalization task, and representation geometry (uniformity, effective
rank). See PHASE1_5_RESULTS.md for the honest scope-reduction from the
plan's 10M-50M-parameter / 100M-1B-token / 1-3 PF-day spec.
'''
from __future__ import annotations
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lattice.tiny_transformer import (
    ARLanguageModel, Predictor, TinyTransformer, ema_update, tau_schedule,
)
from lattice.toy_corpus import CharTokenizer


@dataclass
class ToyConfig:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 192
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    mask_scale: tuple[float, float] = (0.15, 0.20)
    ema_start: float = 0.996
    seed: int = 0


def make_block_mask(batch_size: int, seq_len: int,
                    scale_range: tuple[float, float], rng: random.Random
                    ) -> torch.Tensor:
    '''Multi-block masking (spec: V-JEPA-style), one contiguous span per
    example, span length drawn from `scale_range` * seq_len.'''
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for b in range(batch_size):
        span = max(1, int(seq_len * rng.uniform(*scale_range)))
        span = min(span, seq_len)
        start = rng.randint(0, seq_len - span)
        mask[b, start:start + span] = True
    return mask


def train_ar(model: ARLanguageModel, tokens: torch.Tensor, config: ToyConfig,
            device: str) -> list[float]:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    n = tokens.size(0)
    steps_per_epoch = max(1, n // config.batch_size)
    history = []
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size:(i + 1) * config.batch_size]
            batch = tokens[idx].to(device)
            logits, _ = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=CharTokenizer.PAD,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
    return history


@dataclass
class JepaModel:
    context_encoder: TinyTransformer
    target_encoder: TinyTransformer
    predictor: Predictor

    def num_params(self) -> int:
        # target_encoder is a non-trained EMA copy; don't count it as
        # trained capacity, but context_encoder + predictor are trained.
        return self.context_encoder.num_params() + sum(
            p.numel() for p in self.predictor.parameters())


def build_jepa(vocab_size: int, config: ToyConfig) -> JepaModel:
    context_encoder = TinyTransformer(vocab_size, config.dim, config.n_layers,
                                      config.n_heads, config.max_len)
    target_encoder = TinyTransformer(vocab_size, config.dim, config.n_layers,
                                     config.n_heads, config.max_len)
    target_encoder.load_state_dict(context_encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor(config.dim)
    return JepaModel(context_encoder, target_encoder, predictor)


def train_jepa(jepa: JepaModel, tokens: torch.Tensor, config: ToyConfig,
              device: str) -> list[float]:
    jepa.context_encoder.to(device).train()
    jepa.target_encoder.to(device).eval()
    jepa.predictor.to(device).train()
    rng = random.Random(config.seed)
    opt = torch.optim.AdamW(
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters()), lr=config.lr,
    )
    n = tokens.size(0)
    steps_per_epoch = max(1, n // config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    history = []
    step = 0
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size:(i + 1) * config.batch_size]
            batch = tokens[idx].to(device)
            B, T = batch.shape
            block_mask = make_block_mask(B, T, config.mask_scale, rng).to(device)

            context_input = batch.clone()
            context_input[block_mask] = CharTokenizer.MASK

            with torch.no_grad():
                target_h = jepa.target_encoder(batch, causal=False)
            context_h = jepa.context_encoder(context_input, causal=False)
            pred = jepa.predictor(context_h)

            pred_m = pred[block_mask]
            target_m = target_h[block_mask].detach()
            loss = -F.cosine_similarity(pred_m, target_m, dim=-1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            tau = tau_schedule(step, total_steps, start=config.ema_start)
            ema_update(jepa.target_encoder, jepa.context_encoder, tau)

            history.append(float(loss.item()))
            step += 1
    return history


@dataclass
class ContrastiveModel:
    '''SimCLR-style contrastive arm: same encoder architecture as the JEPA
    context encoder, no target network. A projector maps pooled features to
    the space the InfoNCE loss is computed in (SimCLR practice); downstream
    probes use the encoder's raw pooled output, not the projection, so the
    comparison to AR/JEPA pooled features stays apples-to-apples.
    '''
    encoder: TinyTransformer
    projector: Predictor

    def num_params(self) -> int:
        return self.encoder.num_params() + sum(
            p.numel() for p in self.projector.parameters())


def build_contrastive(vocab_size: int, config: ToyConfig) -> ContrastiveModel:
    encoder = TinyTransformer(vocab_size, config.dim, config.n_layers,
                              config.n_heads, config.max_len)
    projector = Predictor(config.dim)
    return ContrastiveModel(encoder, projector)


def make_augmented_view(batch: torch.Tensor, rng: random.Random,
                        crop_range: tuple[float, float] = (0.7, 0.9),
                        mask_rate: float = 0.1) -> torch.Tensor:
    '''Two independent calls on the same batch produce the positive pair
    for InfoNCE: a random contiguous crop (rest set to PAD) plus random
    per-character masking within the kept span. This is the toy-scale
    stand-in for SimCLR-style augmentation (spec calls for "augmentation
    pairs"); there is no vision-style crop/color-jitter equivalent for text,
    so span-crop + token dropout is used instead.
    '''
    B, T = batch.shape
    out = batch.clone()
    for b in range(B):
        crop_len = max(1, min(T, int(T * rng.uniform(*crop_range))))
        start = rng.randint(0, T - crop_len)
        keep = torch.zeros(T, dtype=torch.bool)
        keep[start:start + crop_len] = True
        out[b, ~keep] = CharTokenizer.PAD
        for t in range(start, start + crop_len):
            if rng.random() < mask_rate:
                out[b, t] = CharTokenizer.MASK
    return out


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    '''NT-Xent / InfoNCE: each view's positive is its augmented partner;
    all other views in the batch (both augmentations) are negatives.'''
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float('-inf'))
    targets = torch.cat([
        torch.arange(B, 2 * B), torch.arange(0, B),
    ]).to(z.device)
    return F.cross_entropy(sim, targets)


def train_contrastive(model: ContrastiveModel, tokens: torch.Tensor,
                      config: ToyConfig, device: str) -> list[float]:
    model.encoder.to(device).train()
    model.projector.to(device).train()
    rng = random.Random(config.seed)
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=config.lr,
    )
    n = tokens.size(0)
    steps_per_epoch = max(1, n // config.batch_size)
    history = []
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(steps_per_epoch):
            idx = perm[i * config.batch_size:(i + 1) * config.batch_size]
            batch = tokens[idx]
            view1 = make_augmented_view(batch, rng).to(device)
            view2 = make_augmented_view(batch, rng).to(device)
            h1 = model.encoder(view1, causal=False).mean(dim=1)
            h2 = model.encoder(view2, causal=False).mean(dim=1)
            p1, p2 = model.projector(h1), model.projector(h2)
            loss = info_nce_loss(p1, p2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
    return history


def encode_pooled_contrastive(model: ContrastiveModel, tokens: torch.Tensor,
                              device: str, batch_size: int = 256
                              ) -> torch.Tensor:
    model.encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            h = model.encoder(batch, causal=False)
            feats.append(h.mean(dim=1).cpu())
    return torch.cat(feats, dim=0)


def encode_pooled_ar(model: ARLanguageModel, tokens: torch.Tensor,
                     device: str, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            h = model.backbone(batch, causal=True)
            feats.append(h.mean(dim=1).cpu())
    return torch.cat(feats, dim=0)


def encode_pooled_jepa(jepa: JepaModel, tokens: torch.Tensor, device: str,
                       batch_size: int = 256) -> torch.Tensor:
    jepa.context_encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size].to(device)
            h = jepa.context_encoder(batch, causal=False)
            feats.append(h.mean(dim=1).cpu())
    return torch.cat(feats, dim=0)


# ---- Evaluation: linear probe, composition task, representation geometry --

def linear_probe_accuracy(
    train_features: torch.Tensor, train_labels: torch.Tensor,
    test_features: torch.Tensor, test_labels: torch.Tensor,
    num_classes: int, epochs: int = 200, lr: float = 1e-2,
) -> float:
    '''Train a single linear layer on frozen features; report test accuracy.'''
    dim = train_features.size(-1)
    probe = torch.nn.Linear(dim, num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_x = (train_features - mean) / std
    test_x = (test_features - mean) / std
    for _ in range(epochs):
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(test_x).argmax(dim=-1)
    return (preds == test_labels).float().mean().item()


_COLORS = ['red', 'blue', 'green', 'yellow']
_SHAPES = ['circle', 'square', 'triangle', 'star']
# Held-out *pairings* never seen together at train time. Each individual
# color and shape value still appears in several other training pairings —
# a linear probe cannot predict a class label it never saw during training
# regardless of representation quality, so the joint (color, shape) pair
# cannot be the probe's label space (an earlier version of this function
# did that, and it produced a structural 0% on every held-out pair by
# construction, not a representation finding). Instead, color and shape
# are probed as two independent attributes; success on a held-out pairing
# means correctly reading off *both* attributes for a combination the
# probe's training data never presented together (plan: "train on simple
# compositions ... test on novel combinations").
_HELD_OUT = {('red', 'square'), ('blue', 'circle'), ('green', 'star'),
            ('yellow', 'triangle')}


@dataclass
class CompositionSplit:
    train_tokens: torch.Tensor
    train_color: torch.Tensor
    train_shape: torch.Tensor
    test_tokens: torch.Tensor
    test_color: torch.Tensor
    test_shape: torch.Tensor


def build_composition_task(tokenizer: CharTokenizer, max_len: int
                           ) -> CompositionSplit:
    train_texts, train_color, train_shape = [], [], []
    test_texts, test_color, test_shape = [], [], []
    for ci, color in enumerate(_COLORS):
        for si, shape in enumerate(_SHAPES):
            for article in ('the', 'a', 'this', 'that'):
                text = f'{article} {color} {shape} is on the table'
                if (color, shape) in _HELD_OUT:
                    test_texts.append(text)
                    test_color.append(ci)
                    test_shape.append(si)
                else:
                    train_texts.append(text)
                    train_color.append(ci)
                    train_shape.append(si)
    return CompositionSplit(
        train_tokens=tokenizer.encode_batch(train_texts, max_len),
        train_color=torch.tensor(train_color),
        train_shape=torch.tensor(train_shape),
        test_tokens=tokenizer.encode_batch(test_texts, max_len),
        test_color=torch.tensor(test_color),
        test_shape=torch.tensor(test_shape),
    )


def composition_accuracy(
    train_features: torch.Tensor, split: CompositionSplit,
    test_features: torch.Tensor, epochs: int = 300,
) -> dict:
    '''Two independent 4-way linear probes (color, shape). Joint accuracy
    requires both to be correct on a held-out pairing — the actual
    compositional-generalization signal; the per-attribute accuracies are
    kept for diagnosis (e.g. "color transfers but shape doesn't").
    '''
    color_acc = linear_probe_accuracy(
        train_features, split.train_color, test_features, split.test_color,
        num_classes=len(_COLORS), epochs=epochs)
    shape_acc = linear_probe_accuracy(
        train_features, split.train_shape, test_features, split.test_shape,
        num_classes=len(_SHAPES), epochs=epochs)
    # Joint accuracy needs per-example correctness, not the two independent
    # rates multiplied together -- recompute predictions directly.
    color_probe, shape_probe = _fit_probe(
        train_features, split.train_color, len(_COLORS), epochs), _fit_probe(
        train_features, split.train_shape, len(_SHAPES), epochs)
    with torch.no_grad():
        color_pred = color_probe(test_features).argmax(-1)
        shape_pred = shape_probe(test_features).argmax(-1)
    joint = ((color_pred == split.test_color)
            & (shape_pred == split.test_shape)).float().mean().item()
    return {'color_accuracy': color_acc, 'shape_accuracy': shape_acc,
            'joint_accuracy': joint}


def _fit_probe(features: torch.Tensor, labels: torch.Tensor,
              num_classes: int, epochs: int, lr: float = 1e-2
              ) -> torch.nn.Linear:
    dim = features.size(-1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
    x = (features - mean) / std
    probe = torch.nn.Linear(dim, num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(x), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Wrap so callers can apply the same standardization at test time.
    return _StandardizedProbe(probe, mean, std)


class _StandardizedProbe(torch.nn.Module):
    def __init__(self, probe, mean, std):
        super().__init__()
        self.probe, self.mean, self.std = probe, mean, std

    def forward(self, x):
        return self.probe((x - self.mean) / self.std)


def uniformity(z: torch.Tensor, t: float = 2.0) -> float:
    '''Wang & Isola (2020) uniformity metric on L2-normalized features.'''
    z = F.normalize(z, dim=-1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_dists).mean()).item()


def effective_rank(z: torch.Tensor) -> float:
    '''exp(entropy of normalized singular values) — a scalar proxy for how
    many dimensions the representation actually uses.'''
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / s.sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()
