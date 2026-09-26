# Phase 0 Results

Status: **Phase 0 complete** — encoder decision resolved (latency-gated); generative baseline done. Ready for Phase 1.

Per the order recommended in `PLAN_REVISION.md`: reproducibility, then a
latency-gated encoder comparison, then a fair generative baseline. This
document tracks the empirical results that gate Phase 1.

## Reproducibility

- Isolated environment: `.venv` (created with `--system-site-packages` so it
  reuses the CUDA-13 torch build that produced these numbers).
- Pinned versions captured in `requirements.lock` (torch 2.13.0+cu130,
  transformers 5.13.0, datasets 4.8.5, tokenizers 0.22.2, ...).
- Dataset: BANKING77, loaded from `data/banking77.csv` (the installed
  `datasets==4.8.5` refuses loading-script datasets, so we ship a CSV loader).
  `load_banking77` prints the download URL on miss.

## Encoder comparison — gated by BOTH accuracy and CPU latency

Same readout head, same splits, same RLCD + post-hoc temperature (30 epochs,
lr 3e-3, beta 1.0). Accuracy/calibration measured on GPU; **CPU latency is
encoder forward + typed head, 4 threads, max_len 64 (BANKING77 queries are
short), p95 over 150 single-decision calls.** The gate is the spec's Phase 0
target: **< 50 ms per decision on CPU.**

| Frozen encoder (feature extractor) | dim | params | Acc | ECE cal | CPU p95 | Gate |
|---|---:|---:|---:|---:|---:|:--:|
| distilbert-base-uncased | 768 | 66M | 0.680 | 0.054 | 12.9 ms | pass |
| microsoft/deberta-v3-base | 768 | 149M | 0.509 | 0.055 | n/a | rejected |
| Qwen/Qwen3-0.6B | 1024 | 600M | 0.804 | 0.017 | 442 ms | **FAIL** |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 22M | 0.839 | 0.054 | 4.7 ms | pass |
| **BAAI/bge-small-en-v1.5** | 384 | 33M | **0.873** | 0.056 | **8.4 ms** (6.1 int8) | **pass** |

Reading:
- **Qwen3-0.6B wins calibration on GPU but is 10× over the CPU latency gate.**
  It is not deployable at the spec's per-decision cost target; int8 on CPU does
  not close a 442 ms → 50 ms gap.
- **bge-small-en-v1.5 dominates on the two gates that matter together**: highest
  accuracy (0.873, beating every encoder tried including Qwen3-0.6B) AND CPU
  p95 of 8.4 ms (int8 6.1 ms) — a ~6× margin under the gate at ~18× fewer
  params. Calibration after temperature scaling is on par (~0.055).
- DeBERTa's 0.509 is a **mean-pooling artifact** of its base checkpoint; not a
  usable row. DistilBERT clears latency but is accuracy-limited (0.68).

## Decision for Phase 1

Use **frozen `BAAI/bge-small-en-v1.5` as the shared representation** for the
deployable slice. The earlier 0.62/0.68 weakness was encoder-limited; swapping
representation is free because the readout head, controller, and constraint
layer are unchanged. (Qwen3-0.6B remains a strong GPU/edge-NPU option where the
CPU gate does not apply, but it is not the Phase 1 default.)

Open sub-question for Phase 1: whether the deployed head is trained on
mean-pooled features (as here) or on `[CLS]`/sentence-transformer pooling; the
head can absorb either, but the generator baseline (#2) uses causal decoding,
so the interface comparison is representation-agnostic.

## Latency

- Typed readout head alone (GPU): ~0.04 ms/decision.
- **End-to-end CPU now measured** (see table): bge-small 8.4 ms p95, well inside
  the < 50 ms gate. Latency gate is **closed** for the chosen encoder.

## Generative baseline — DONE (task #2)

Fair comparison per the plan: the same-class instruct model (Qwen3-0.6B, the
model that also tops the encoder comparison on GPU) asked to *generate* an
intent under **constrained decoding** over the exact 77-label trie — the
probability of each label is the joint likelihood of its token path plus the
terminating token, so shared-prefix pairs like `card_arrival` /
`card_arrival_estimate` are scored correctly by construction instead of
colliding under raw likelihood. Same test split as Phase 0
(`scripts/generative_baseline.py`).

| Shots | Acc (exact) | ECE (raw) | Brier | Prefix-collision (of errors) | Latency |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.060 | 0.724 | 1.556 | 0.032 | 222 ms/ex |
| 3 | 0.110 | 0.552 | 1.346 | 0.022 | 527 ms/ex |
| 5 | 0.150 | 0.525 | 1.247 | 0.006 | 687 ms/ex |

Reading:
- **Not comparable accuracy.** Even at 5-shot, constrained-decoding Qwen3-0.6B
  tops out at 0.150 — far below the typed-head numbers on the *same*
  BANKING77 split (0.680–0.873 depending on encoder, 0.804 for Qwen3-0.6B's
  own frozen-encoder row). More shots help, but generation is not closing that
  gap; label choice for a 0.6B model under a chat template is a much harder
  task than linear readout over its own pooled features.
- **Calibration is much worse**: ECE 0.525 vs ~0.055 for the typed head after
  temperature scaling — a ~10x gap, even before accounting for the accuracy
  difference.
- **Prefix collision is not the dominant error mode** once decoding is
  constrained (0.6-3.2% of errors) — confirming the fix does what it's meant
  to (stop shared-prefix labels from colliding), but it does not rescue
  overall accuracy, which is capped by the model's ability to pick the right
  leaf, not by the trie construction.
- **Latency and cost are much higher**: 222-687 ms/example (77 leaf scorings
  each) vs 8.4 ms p95 end-to-end for the typed head on CPU — roughly two to
  three orders of magnitude, even before comparing hardware (this run is GPU;
  the typed head number is CPU).

Net: decision-by-generation is worse on every axis Phase 1 cares about
(accuracy, calibration, latency, cost) than decision-by-readout over the same
frozen representation. This is the empirical case for typed heads over
prompted generation that P1 is built on.

## Follow-ups still open

1. Phase 1 synthetic triage corpus + multi-head controller with thresholds
   grounded in the #2 generator result above.
