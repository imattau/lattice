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
    ar_jepa_alpha: float = 0.5
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


@dataclass
class ArJepaModel:
    '''Token-primary, latent-auxiliary arm (the reframe LLM-JEPA and similar
    published work actually use, vs. the spec's original latent-primary
    framing): a standard causal AR model, plus a "NextLat" auxiliary loss —
    predict the EMA target encoder's *next-position* latent from the causal
    hidden state at the current position, exactly analogous to next-token
    prediction but in latent space instead of vocabulary space. No masking
    is needed (unlike the bidirectional JEPA arm) since the causal structure
    already gives a well-defined "next" position to predict.
    '''
    ar_model: ARLanguageModel
    target_encoder: TinyTransformer
    predictor: Predictor

    def num_params(self) -> int:
        return self.ar_model.num_params() + sum(
            p.numel() for p in self.predictor.parameters())


def build_ar_jepa(vocab_size: int, config: ToyConfig) -> ArJepaModel:
    context_backbone = TinyTransformer(vocab_size, config.dim, config.n_layers,
                                       config.n_heads, config.max_len)
    ar_model = ARLanguageModel(context_backbone)
    target_encoder = TinyTransformer(vocab_size, config.dim, config.n_layers,
                                     config.n_heads, config.max_len)
    target_encoder.load_state_dict(context_backbone.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor(config.dim)
    return ArJepaModel(ar_model, target_encoder, predictor)


def train_ar_jepa(model: ArJepaModel, tokens: torch.Tensor, config: ToyConfig,
                  device: str) -> list[dict]:
    '''L = L_AR (next-token cross-entropy) + alpha * L_NextLat (cosine loss
    predicting the EMA target encoder's next-position latent). This is the
    "token-primary, latent-auxiliary" ordering, the reverse of the JEPA
    arm's "latent-primary" objective, at the same architecture/data/budget.
    '''
    model.ar_model.to(device).train()
    model.target_encoder.to(device).eval()
    model.predictor.to(device).train()
    opt = torch.optim.AdamW(
        list(model.ar_model.parameters()) + list(model.predictor.parameters()),
        lr=config.lr,
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

            logits, context_h = model.ar_model(batch)  # causal, full length
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1), ignore_index=CharTokenizer.PAD,
            )

            with torch.no_grad():
                target_h = model.target_encoder(batch, causal=True)
            pred = model.predictor(context_h[:, :-1])
            target_next = target_h[:, 1:].detach()
            valid = (batch[:, 1:] != CharTokenizer.PAD).float()
            cos = F.cosine_similarity(pred, target_next, dim=-1)
            latent_loss = -(cos * valid).sum() / valid.sum().clamp(min=1)

            loss = ce + config.ar_jepa_alpha * latent_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            tau = tau_schedule(step, total_steps, start=config.ema_start)
            ema_update(model.target_encoder, model.ar_model.backbone, tau)

            history.append({'loss': float(loss.item()), 'ce': float(ce.item()),
                            'latent': float(latent_loss.item())})
            step += 1
    return history


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


# ---- Deep (non-lexical) compositional generalization -----------------------
#
# The task above can be solved from layer 1 by every arm (including AR at
# 1.000 joint accuracy) because the color and shape words are literal
# substrings of the input -- a probe only needs to detect "was the substring
# 'red' present", never combine information across positions. That is closer
# to lexical detection than compositional reasoning, and the readout-pyramid
# follow-up (PHASE1_5_LAYER_PYRAMID_RESULTS.md) flagged this as a real
# caveat on how much weight the shallow task's numbers can bear.
#
# This variant removes the shortcut with two-hop variable binding: every
# color and every shape word appears as a substring in *every* example
# regardless of the answer (via a full link->value mapping table), and which
# value is bound to "x" (the color answer) or "y" (the shape answer) is
# randomized per example. A bag-of-words / single-position read cannot solve
# this -- "is `red` present" is true in every example, train and test alike.
# The correct answer requires resolving x -> its link -> the value that link
# maps to *in this specific example*, i.e. combining two separate facts.

_LINKS_COLOR = ['p', 'q', 'r', 's']
_LINKS_SHAPE = ['m', 'n', 'o', 'w']


def _gen_deep_composition_text(rng: random.Random, target_color_idx: int,
                               target_shape_idx: int) -> str:
    color_perm = list(range(len(_COLORS)))
    rng.shuffle(color_perm)
    shape_perm = list(range(len(_SHAPES)))
    rng.shuffle(shape_perm)
    x_link = _LINKS_COLOR[color_perm.index(target_color_idx)]
    y_link = _LINKS_SHAPE[shape_perm.index(target_shape_idx)]
    color_assignments = ' '.join(
        f'{_LINKS_COLOR[i]}={_COLORS[color_perm[i]]}' for i in range(len(_COLORS)))
    shape_assignments = ' '.join(
        f'{_LINKS_SHAPE[i]}={_SHAPES[shape_perm[i]]}' for i in range(len(_SHAPES)))
    return f'x={x_link} y={y_link} {color_assignments} {shape_assignments}'


def _gen_deep_composition_text_natural(rng: random.Random,
                                       target_color_idx: int,
                                       target_shape_idx: int) -> str:
    '''Same two-hop binding, ordinary-English register instead of dense
    assignment syntax ("x=p y=m p=red..."). Every color/shape word still
    appears in every example regardless of the answer -- the anti-shortcut
    property is unchanged -- but the sentence pattern ("p is red.") is
    simple declarative English, much closer to the pretraining corpus's
    register (synthetic code/JSON aside, its natural-language portions --
    BANKING77, Phase 1 triage text -- are ordinary sentences, not
    symbolic notation). This isolates "can the objective do two-hop
    binding" from "was this exact syntax ever seen during pretraining",
    the confound flagged in PHASE1_5_DEEP_COMPOSITION_RESULTS.md.
    '''
    color_perm = list(range(len(_COLORS)))
    rng.shuffle(color_perm)
    shape_perm = list(range(len(_SHAPES)))
    rng.shuffle(shape_perm)
    x_link = _LINKS_COLOR[color_perm.index(target_color_idx)]
    y_link = _LINKS_SHAPE[shape_perm.index(target_shape_idx)]
    color_sentences = ' '.join(
        f'{_LINKS_COLOR[i]} is {_COLORS[color_perm[i]]}.'
        for i in range(len(_COLORS)))
    shape_sentences = ' '.join(
        f'{_LINKS_SHAPE[i]} is {_SHAPES[shape_perm[i]]}.'
        for i in range(len(_SHAPES)))
    return f'{color_sentences} {shape_sentences} x is {x_link}. y is {y_link}.'


def build_deep_composition_task(
    tokenizer: CharTokenizer, max_len: int, n_per_combo: int = 16,
    seed: int = 0, style: str = 'symbolic',
) -> CompositionSplit:
    '''Same held-out (color, shape) pairings as `build_composition_task`
    (so results are comparable), but text requires two-hop variable-binding
    resolution instead of literal substring detection to answer.

    `style`: 'symbolic' (original: "x=p y=m p=red...") or 'natural'
    (ordinary-English sentences: "p is red. ... x is p."), same underlying
    binding task and anti-shortcut property either way.
    '''
    gen_fn = (_gen_deep_composition_text if style == 'symbolic'
             else _gen_deep_composition_text_natural)
    rng = random.Random(seed)
    train_texts, train_color, train_shape = [], [], []
    test_texts, test_color, test_shape = [], [], []
    for ci, color in enumerate(_COLORS):
        for si, shape in enumerate(_SHAPES):
            is_held_out = (color, shape) in _HELD_OUT
            for _ in range(n_per_combo):
                text = gen_fn(rng, ci, si)
                if is_held_out:
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


def load_arm_backbone(ckpt: dict, key: str, config: dict, vocab_size: int
                      ) -> TinyTransformer:
    '''Rebuild a trained backbone from a PHASE1_5_RESULTS.pt-style checkpoint
    without retraining -- used by follow-up diagnostics (e.g. the
    readout-pyramid / layer-probing scripts) that reuse the four arms'
    already-trained weights.'''
    model = TinyTransformer(vocab_size, config['dim'], config['n_layers'],
                            config['n_heads'], config['max_len'])
    model.load_state_dict(ckpt[key])
    model.eval()
    return model


def layerwise_pooled(model: TinyTransformer, tokens: torch.Tensor,
                     causal: bool, batch_size: int = 256
                     ) -> list[torch.Tensor]:
    '''Mean-pooled features at every layer: a list of [N, D] tensors, one
    per transformer block, for probing representation quality at each
    depth instead of only the top layer's output.'''
    n_layers = len(model.blocks)
    per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i:i + batch_size]
            hiddens = model.forward_layers(batch, causal=causal)
            for li, h in enumerate(hiddens):
                per_layer[li].append(h.mean(dim=1))
    return [torch.cat(chunks, dim=0) for chunks in per_layer]


def effective_rank(z: torch.Tensor) -> float:
    '''exp(entropy of normalized singular values) — a scalar proxy for how
    many dimensions the representation actually uses.'''
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / s.sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()
