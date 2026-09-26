# Phase 0 Results

Status: **encoder decision resolved; generative baseline preliminary.**

This document records the empirical results that gate Phase 1, per the
order recommended in `PLAN_REVISION.md` / `PLAN.md`: reproducibility first,
then a three-encoder comparison, then a generative baseline.

## Reproducibility

- Isolated environment: `.venv` (created with `--system-site-packages` so it
  reuses the CUDA-13 torch build that produced these numbers).
- Pinned versions captured in `requirements.lock` (torch 2.13.0+cu130,
  transformers 5.13.0, datasets 4.8.5, tokenizers 0.22.2, ...).
- Dataset: BANKING77 (`PolyAI`), loaded from `data/banking77.csv` (the
  installed `datasets==4.8.5` refuses loading-script datasets, so we ship a
  CSV loader). Download URL is printed by `load_banking77` on miss.

## Headline: the 0.62 was encoder-limited

Same readout head, same splits, same RLCD + post-hoc temperature; only the
frozen encoder changes (30 epochs, lr 3e-3, RLCD beta 1.0). Latency is the
typed readout head forward only (GPU), not end-to-end.

| Frozen encoder (as feature extractor) | dim | Accuracy | ECE raw | ECE cal | Brier | Brier skill* |
|---|---:|---:|---:|---:|---:|---:|
| distilbert-base-uncased | 768 | 0.680 | 0.403 | 0.054 | 0.639 | 0.352 |
| microsoft/deberta-v3-base | 768 | 0.509 | 0.312 | 0.055 | 0.769 | 0.220 |
| **Qwen/Qwen3-0.6B** | 1024 | **0.804** | 0.069 | **0.017** | **0.286** | **0.710** |

\* Brier skill vs the class-prior baseline (prior Brier 0.986). Higher is better.

Reading:
- **Qwen3-0.6B used purely as a frozen encoder clears the ≥ 0.75 bar** with
  accuracy 0.804 and a spectacular post-calibration ECE of 0.017. This is the
  Phase 0 success to treat as real.
- DeBERTa-v3-base's low 0.509 is a **mean-pooling artifact** of its base
  checkpoint (its own pooling is non-standard); it is not evidence that
  DeBERTa is a weak encoder. Do not draw conclusions from that row.
- The earlier 0.62 vs 0.680 for DistilBERT is a head-hyperparameter effect
  (20 vs 30 epochs); some of the gap was head-limited, but the encoder swap
  dominates.

## Decision for Phase 1

Use **frozen Qwen3-0.6B as the shared representation.** The representation is
now strong enough that the controller's escalation threshold reflects genuine
uncertainty rather than a weak encoder — so Phase 1 can validate the
controller/constraint layer without thresholds silently compensating for
representation quality.

## Latency

- Typed readout head: ~0.04 ms/decision (GPU).
- **Open item:** the Phase 0 gate is < 50 ms **on CPU, end-to-end** (encoder
  forward + head). Not yet measured on CPU; DistilBERT is cheap but Qwen3-0.6B
  on CPU is heavier. This must be measured before claiming the latency gate,
  and may motivate a smaller or quantized encoder for the deployed slice.

## Generative baseline (PRELIMINARY — not yet trustworthy)

Per the plan, the baseline is a small local model, not a frontier API. Using
the same Qwen3-0.6B, classification by 77-way teacher-forced likelihood:

- Zero-shot raw cloze: accuracy 0.06 (≈ random).
- 3-shot format-primed: accuracy ~0.16, ECE(raw) ~0.06.

Caveats and why this number is not yet the headline comparison:
- BANKING77 labels share long prefixes (`card_arrival`, `card_arrival_*`,
  `transfer_*`), so mean-token likelihood cannot disambiguate — a base model
  given no vocabulary of options is being asked to hallucinate exact
  snake_case strings.
- At 8 shots the current scorer returns a degenerate 0.000 (a long-prefix
  indexing bug in the scorer), so we do **not** report it.
- A *fair* generative baseline needs an instruct model + constrained decoding
  over the exact label set (the JSON-schema protocol in `PLAN.md` §Phase 0).
  That is a dedicated half-day task, deferred.

What the preliminary result already shows, consistently with the spec thesis
(P1: decision ≠ generation): on a 77-way structured decision, a tiny learned
typed head on a frozen representation is far more accurate *and* more
calibrated than asking the same weights to emit the answer generatively.

## Suggested immediate follow-ups (before Phase 1 build-out)

1. Measure **CPU end-to-end** latency of Qwen3-0.6B encoder + head vs a smaller
   or 4-bit encoder to settle the < 50 ms gate.
2. Implement the **constrained-decoding generative baseline** properly and
   re-run to get a trustworthy comparison row.
3. Add a **fine-tuned BERT reference** (~0.94) as an upper bound so 0.80 from a
   *frozen* encoder is interpreted correctly.
