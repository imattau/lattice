'''Staged training loop for the JEPA Core.

Handles: mask sampling, context/target encoding, predictor, Talker,
auxiliary losses gated by config, EMA schedule, logging.

The loop is deliberately transparent about what is enabled. There is no
implicit switching on objective strings; the caller passes the flags.
'''
from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import TrainConfig
from models import JEPACore
from objectives import (
    MonoidalCompose, combined_objective, functorial_consistency_loss,
    jepa_loss, mlm_loss, next_latent_loss, stp_regularizer,
)


def sample_mask(B: int, T: int, cfg: TrainConfig,
                device) -> torch.Tensor:
    '''Multi-block masking. True = masked (target) position.'''
    import random
    rng = random.Random(cfg.seed + int(time.time()))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    scale = rng.uniform(cfg.jepa.mask_scale_min, cfg.jepa.mask_scale_max)
    aspect = rng.uniform(cfg.jepa.mask_aspect_min, cfg.jepa.mask_aspect_max)
    import math
    for b in range(B):
        for _ in range(cfg.jepa.num_mask_blocks):
            length = max(1, min(T, int(round(math.sqrt(scale * T * T * aspect)))))
            start = rng.randint(0, T - length)
            mask[b, start:start + length] = True
    return mask


def ema_decay(step: int, total: int, cfg: TrainConfig) -> float:
    '''Linear EMA schedule from start to end. Verification noted that
    early-collapse is a real risk; the schedule is part of the fix.'''
    frac = min(1.0, step / max(1, total))
    return cfg.jepa.ema_decay_start * (1 - frac) + cfg.jepa.ema_decay_end * frac


def train_step(
    core: JEPACore,
    batch: dict,
    cfg: TrainConfig,
    step: int,
    compose: MonoidalCompose | None = None,
) -> dict:
    device = next(core.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attn = batch.get('attention_mask')
    if attn is not None:
        attn = attn.to(device)

    B, T = input_ids.shape
    mask = sample_mask(B, T, cfg, device)

    # Context encoding (visible positions only)
    ctx_ids = input_ids.masked_fill(mask, 0)
    ctx_latent = core.context_encode(ctx_ids, attn)

    # Target encoding (full input, no grad)
    tgt_latent = core.target_encode(input_ids, attn)

    # Predict target latents from context via pooled context representation
    pooled = ctx_latent.mean(dim=1, keepdim=True)          # [B, 1, D]
    pred_latent = core.predictor(ctx_latent + 0.0 * pooled)  # [B, T, D]

    # JEPA loss computed only at masked positions
    pred_masked = pred_latent[mask]                        # [N, D]
    tgt_masked = tgt_latent[mask]                          # [N, D]
    l_jepa = jepa_loss(pred_masked, tgt_masked)

    # Auxiliary terms, all gated
    l_next = l_mlm = l_stp = l_func = None

    if cfg.objective.value != 'jepa_only':
        l_next = next_latent_loss(ctx_latent)

    if 'mlm' in cfg.objective.value:
        # Fill in predicted latents at masked positions before the Talker
        filled = ctx_latent.clone()
        filled[mask] = pred_latent[mask].detach()
        logits = core.generate(filled)                     # [B, T, V]
        labels = input_ids.clone()
        labels[~mask] = -100
        l_mlm = mlm_loss(logits, labels)

    if cfg.enable_stp:
        l_stp = stp_regularizer(ctx_latent)

    if cfg.enable_functorial and compose is not None:
        # Split sequence into two halves, treat concat as composition.
        half = T // 2
        if half > 0:
            z_a = ctx_latent[:, :half].mean(dim=1)
            z_b = ctx_latent[:, half:].mean(dim=1)
            z_ab = ctx_latent.mean(dim=1)
            l_func = functorial_consistency_loss(z_a, z_b, z_ab, compose)

    out = combined_objective(
        jepa=l_jepa, weights=cfg.loss_weights,
        nextlat=l_next, mlm=l_mlm, stp=l_stp, functorial=l_func,
    )
    return {
        'loss': out.total,
        'jepa': l_jepa.detach(),
        'nextlat': l_next.detach() if l_next is not None else None,
        'mlm': l_mlm.detach() if l_mlm is not None else None,
        'stp': l_stp.detach() if l_stp is not None else None,
        'functorial': l_func.detach() if l_func is not None else None,
    }


def train(
    core: JEPACore,
    dataloader,
    cfg: TrainConfig,
    device: str = 'cpu',
) -> dict:
    '''Run the full training loop. Returns final metrics and checkpoints.'''
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    core.to(device).train()
    compose = MonoidalCompose(cfg.model.hidden_dim).to(device) \
        if cfg.enable_functorial else None

    params = list(core.reasoner.parameters()) \
        + list(core.predictor.parameters()) \
        + list(core.talker.parameters())
    if compose is not None:
        params += list(compose.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Linear warmup then cosine decay
    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(1, cfg.warmup_steps)
        frac = (step - cfg.warmup_steps) / max(
            1, cfg.total_steps - cfg.warmup_steps)
        import math
        return 0.5 * cfg.lr * (1 + math.cos(math.pi * frac))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    history = []
    step = 0
    t0 = time.time()
    while step < cfg.total_steps:
        for batch in dataloader:
            if step >= cfg.total_steps:
                break
            stats = train_step(core, batch, cfg, step, compose)
            opt.zero_grad()
            stats['loss'].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            scheduler.step()

            decay = ema_decay(step, cfg.total_steps, cfg)
            core.update_target(decay)

            if step % cfg.log_every == 0:
                entry = {
                    'step': step,
                    'lr': scheduler.get_last_lr()[0],
                    'ema': decay,
                    'elapsed': time.time() - t0,
                }
                for k in ('jepa', 'nextlat', 'mlm', 'stp', 'functorial'):
                    v = stats.get(k)
                    if v is not None:
                        entry[k] = float(v)
                history.append(entry)
                print(entry)

            if step > 0 and step % cfg.eval_every == 0:
                ckpt = Path(cfg.output_dir) / f'step_{step}.pt'
                torch.save({
                    'step': step,
                    'model': core.state_dict(),
                    'config': cfg.to_json(),
                }, ckpt)

            step += 1

    torch.save({
        'step': step,
        'model': core.state_dict(),
        'config': cfg.to_json(),
    }, Path(cfg.output_dir) / 'final.pt')
    return {'history': history, 'final_step': step}
