"""Staged training. Spec §5 and §10.

Stage 1: pretrain Reasoner with JEPA (+ optional auxiliaries).
Stage 2: freeze Reasoner, train readout heads.
Stage 3: freeze both, tune controller thresholds on validation.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_core import JEPACore, CoreConfig
from losses import (
    JEPAPredictor, combined_loss, jepa_loss, mlm_loss,
    next_latent_loss, stp_regularizer,
)
from masking import multi_block_mask


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 512
    lr: float = 3e-4
    steps: int = 10000
    num_mask_blocks: int = 4
    warmup_steps: int = 500


def train_reasoner(
    core: JEPACore,
    data_iter,
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Stage 1 — train Reasoner with JEPA objective. Spec §6.5."""
    core.to(device).train()
    predictor = JEPAPredictor(core.cfg.hidden_dim).to(device)
    params = list(core.reasoner.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.05)
    losses: list[float] = []

    for step, batch in enumerate(data_iter):
        if step >= cfg.steps:
            break
        input_ids = batch["input_ids"].to(device)       # [B, T]
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        B, T = input_ids.shape

        mask = multi_block_mask(
            B, T, num_blocks=cfg.num_mask_blocks, device=device,
        )  # True = target

        # Online context encoding
        ctx_ids = input_ids.masked_fill(mask, 0)
        ctx_latent = core.encode(ctx_ids, attn_mask)     # [B, T, D]

        # Target encoding (no grad)
        with torch.no_grad():
            tgt_latent = core.encode_target(input_ids, attn_mask)

        # Predict target latents from context latents at masked positions
        pred_full = predictor(ctx_latent)
        pred_masked = pred_full[mask]                    # [N_masked, D]
        tgt_masked = tgt_latent[mask]                    # [N_masked, D]

        l_jepa = jepa_loss(pred_masked.unsqueeze(0),
                           tgt_masked.unsqueeze(0))

        # Auxiliaries (all optional; spec §6.5)
        l_stp = stp_regularizer(ctx_latent)
        # NextLat: predict next step's latent
        if T > 1:
            l_next = next_latent_loss(ctx_latent[:, :-1], ctx_latent[:, 1:])
        else:
            l_next = torch.zeros((), device=device)

        # MLM: reconstruct masked tokens through the Talker
        talker_logits = core.generate(ctx_latent)        # [B, T, V]
        mlm_labels = input_ids.clone()
        mlm_labels[~mask] = -100
        l_mlm = mlm_loss(talker_logits, mlm_labels)

        loss = combined_loss(
            jepa=l_jepa, next_lat=l_next, mlm=l_mlm, stp=l_stp,
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        core.update_target()
        losses.append(loss.item())

    return losses


def train_readouts(
    core: JEPACore,
    readout_bank: nn.Module,
    data_iter,
    decision_key: str,
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Stage 2 — freeze Reasoner, train readout heads. Spec §5."""
    core.eval()
    for p in core.parameters():
        p.requires_grad = False
    readout_bank.to(device).train()
    opt = torch.optim.AdamW(readout_bank.parameters(), lr=cfg.lr)
    losses: list[float] = []

    for step, batch in enumerate(data_iter):
        if step >= cfg.steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch[decision_key].to(device)
        with torch.no_grad():
            latent = core.encode(input_ids)
            pooled = latent.mean(dim=1)
        readouts = readout_bank(pooled)
        # Use the matching head
        from interfaces import DecisionType
        head_out = readouts[DecisionType(decision_key)]
        loss = F.nll_loss(torch.log(head_out.distribution.clamp(min=1e-8)),
                          labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses
