# Phase 1.5 Results: Toy-Scale JEPA vs. Autoregressive Falsification

Status: **toy-scale pilot run complete. Directional result: JEPA
underperforms the AR baseline on 2 of 3 representation-quality probes,
outside the plan's 10% tolerance. Scope is far below the plan's spec (see
below) — treat this as a pilot that validates the training/eval pipeline
and produces a caution signal, not as the actual Phase 1.5 decision-gate
run.**

## Objective (PLAN.md "Phase 1.5: Toy-Scale Falsification")

Test the central premise of Phase 2 at the smallest possible scale: does a
JEPA-trained language model develop representations comparable to an
autoregressive model of the same size and compute? No JEPA language model
has been shown to develop general representations at any scale — this is a
premise, not a finding, and the plan calls for testing it cheaply before
any large-scale training.

## Honest scope reduction

The plan specifies 10M-50M-parameter models, 100M-1B training tokens, and a
1-3 PF-day compute budget. None of that is executable in a single
interactive session on one consumer GPU (RTX 5060 Ti, 16GB) — 1 PF-day
alone is roughly 500-1000x more compute than this machine can produce in an
hour. Rather than silently run a smaller experiment and describe it as if
it satisfied the plan (the same principle applied to the Phase 1 corpus
generation and the Phase 1 calibration bug), the reduction is stated
directly:

| Axis | Plan spec | This run | Ratio |
|---|---|---|---:|
| Trained parameters | 10M-50M | 4.8M (AR) / 5.1M (JEPA context+predictor) | ~0.1-0.5x |
| Training tokens | 100M-1B | ~3.07M characters | ~0.003-0.03x |
| Compute budget | 1-3 PF-days | ~13 min wall-clock, one RTX 5060 Ti | several orders of magnitude smaller |
| Data sources | The Stack (code), C4 (NL), JSON/tables | synthetic code/JSON templates + real BANKING77/triage text (all offline, no download) | narrower, more repetitive |
| Downstream probes | SST-2, MRPC, MNLI, CoLA, STS-B | BANKING77 intent (77-way), Phase 1 triage department (10-way), synthetic color/shape composition | different tasks — no internet dataset access assumed |
| Masking/loss-weight grid | pure JEPA vs JEPA+NextLat+MLM | pure JEPA only | grid not run |
| Cross-modal transfer | text -> code and back | not run | open |
| Loss-scaling curve | clean power law across data budgets | single run, no data-budget sweep | not run |

**This is a toy-of-the-toy.** It does not by itself satisfy or fail the
plan's Phase 1.5 decision gate — that requires the real-scale run. What it
does provide: (a) a working, tested JEPA training pipeline (EMA target
encoder, multi-block masking, predictor, cosine loss) at the exact
architecture the plan describes, ready to scale up, and (b) a genuine, if
small, empirical data point that should inform — not be overridden by —
optimism about scaling up.

## Architecture and training

`lattice/tiny_transformer.py`: a single `TinyTransformer` backbone (6
layers, 256-dim, 8 heads, 192-token context) used identically by both
variants — same depth/width/heads/context length, so any measured
difference is attributable to the objective, not capacity (plan's explicit
requirement).

- **AR baseline** (`ARLanguageModel`): the backbone with a causal mask and
  a tied LM head, trained with next-character cross-entropy.
- **JEPA Core** (`phase1_5_toy.JepaModel`): a context encoder (same
  backbone, bidirectional) + an EMA target encoder (exact architectural
  copy, weights updated by EMA only, no gradient) + a 2-layer GELU
  predictor. Multi-block masking (one contiguous span per example, span
  length 15-20% of sequence length) replaces the masked span with a MASK
  token in the context encoder's input; the loss is negative cosine
  similarity between the predictor's output at masked positions and the
  target encoder's (stop-gradient) output at those positions. EMA
  schedule 0.996 -> 1.0, linear (I-JEPA schedule, per spec).
- Both trained with AdamW, same LR (3e-4), same batch size (64), same
  epoch count (20) over the same corpus — matched compute in the sense of
  identical steps and architecture, though JEPA does an extra forward
  pass through the (non-gradient) target encoder each step, so its actual
  FLOP cost per step is higher, consistent with the plan's own compute
  table (JEPA costs more per step than a matched AR baseline).
- Character-level tokenizer (`lattice.toy_corpus.CharTokenizer`, no
  external dependency or download): at this model scale a subword
  vocabulary's embedding table would dominate parameter count, so a
  ~90-symbol character vocabulary keeps nearly all trained capacity in the
  transformer body — closer to the plan's intent of comparing objectives
  at matched *transformer* capacity, at the cost of the tokenizer itself
  being unrealistic for a real LM.
- Corpus (`lattice.toy_corpus.build_pretrain_corpus`): 16,000 documents —
  4,000 each of synthetic code snippets, synthetic JSON records, real
  BANKING77 customer-service queries, and real Phase 1 triage document
  bodies (the two "structured"/"code" sources are template-generated,
  same honest-deviation pattern as `lattice/synthetic_triage.py`; the two
  "natural language" sources are genuine project text, not filler).

## A bug found and fixed before reporting

The first run of the compositional-generalization probe reported **0.0%
accuracy for both models** — assigning each (color, shape) pairing a
single joint class label meant the 4 held-out pairings were classes that
*never appeared during probe training at all*. A linear softmax head
cannot predict a class it never received a gradient signal for, regardless
of representation quality — that is a test-design bug, not a finding
about either model (same "decompose before reporting" discipline as the
Phase 1 calibration bug: an implausible number was investigated before
being written up). Fixed by probing color and shape as two independent
4-way attributes (each value appears in several *other* training pairings,
just never combined with its held-out partner) and reporting joint
accuracy (both attributes correct) as the compositional-generalization
metric — this is what `composition_accuracy` in
`lattice/phases/phase1_5_toy.py` does, and it is covered by
`tests/test_phase1_5_toy.py::test_composition_accuracy_recovers_separable_attributes`,
which fails against the old joint-label formulation.

## Results (seed 0, 20 epochs — `PHASE1_5_RESULTS.json`)

| | AR | JEPA | JEPA/AR | Plan's "within 10%" gate |
|---|---:|---:|---:|:--:|
| Final training loss | 0.402 (CE, nats) | -0.929 (cosine) | n/a (different loss scales) | n/a |
| BANKING77 intent probe (77-way) | 0.085 | 0.058 | 68% | **fail** |
| Triage department probe (10-way) | 1.000 | 0.950 | 95% | pass |
| Composition — color accuracy | 1.000 | 0.750 | 75% | fail |
| Composition — shape accuracy | 1.000 | 0.938 | 94% | pass |
| **Composition — joint accuracy** | **1.000** | **0.688** | **69%** | **fail** |

| Representation geometry | AR | JEPA |
|---|---:|---:|
| Uniformity (Wang & Isola; more negative = more uniform) | -0.013 | -0.628 |
| Effective rank (of 256 dims) | 94.8 | 92.3 |

Reading:

- **Directionally, JEPA underperforms AR on 2 of 3 real probes at this
  scale**, missing the plan's 10%-relative-gap tolerance on BANKING77
  intent (68% of AR) and on compositional joint accuracy (69% of AR). It
  clears the tolerance on the triage department probe (95% of AR) and on
  the composition task's shape sub-accuracy alone (94%), but not on color
  sub-accuracy (75%) or the joint metric that actually tests composition.
  Given the plan's own guidance — "if the Core fails at 10M-50M
  parameters, it will likely fail at 338M" — this is a real caution
  signal, not noise to wave away, even though the run is far below the
  scale that guidance was written for.
- **Do not read this as a decisive falsification.** The corpus is ~1000x
  smaller than spec, heavily templated (code/JSON sources repeat a small
  number of patterns), and no masking-rate/loss-weight grid was run for
  JEPA (the plan explicitly calls for one — "pure JEPA vs JEPA + NextLat +
  MLM" — before drawing conclusions). A single-seed, single-configuration
  toy run is a data point, not the Phase 1.5 decision-gate result.
- **Representation geometry does not track downstream usefulness here,
  which is itself worth flagging.** JEPA's pooled representations are
  properly spread on the hypersphere (uniformity -0.628); AR's are nearly
  collapsed (-0.013, i.e., most document-level pooled vectors are close
  together). By the plan's "representation geometry comparable to AR" eval
  axis alone, JEPA would look fine or even better. But AR still wins every
  downstream probe. The lesson: uniformity is not a reliable stand-in for
  probe accuracy at this scale, so Phase 2's Latent Probing Suite should
  weight the actual linear-probe numbers over geometry metrics when they
  disagree, not treat "comparable geometry" as sufficient evidence on its
  own.
- **Effective rank is similar between the two (94.8 vs 92.3 of 256)** —
  this metric doesn't discriminate between the two objectives at this
  scale.
- **Both models are near-chance on the BANKING77 77-way probe in absolute
  terms** (chance = 1.3%; AR 8.5%, JEPA 5.8%). Neither representation is
  actually good at fine-grained intent classification from a
  character-level, template/business-text-only pretraining corpus — the
  *relative* AR-vs-JEPA gap is the only thing this number is fit to
  support, not either model's absolute quality.

## What this does and doesn't tell us

- **Validated:** the JEPA training machinery (EMA target encoder,
  multi-block masking, predictor, cosine loss, tau schedule) runs, is
  stable (loss decreases smoothly from ~0 to -0.93 cosine similarity), and
  is wired identically enough to the AR baseline that a fair comparison at
  matched architecture is possible. This means Phase 2 doesn't need to
  build this machinery from scratch.
- **Not validated:** that JEPA develops representations comparable to AR
  at *any* scale, including this one. On this run, at this scale, it does
  not, on 2 of 3 tasks. The plan's actual Phase 1.5 decision gate — run at
  the real 10M-50M-parameter / 100M-1B-token scale, with a masking/loss
  grid and the real probe suite — has not been executed and this pilot
  should not be cited as having executed it.

## Follow-ups still open

1. **Run at real Phase 1.5 scale** (10M-50M params, 100M-1B tokens,
   1-3 PF-days) if/when that compute becomes available — this pilot's
   pipeline (`lattice/tiny_transformer.py`,
   `lattice/phases/phase1_5_toy.py`) is architected to scale up directly
   (swap the corpus loader and bump `ToyConfig`), not rewritten.
2. **Masking-rate / loss-weight grid for JEPA** (pure JEPA vs
   JEPA+NextLat vs JEPA+NextLat+MLM) — not run in this pilot; the plan
   requires it before drawing conclusions about the objective itself
   rather than one specific configuration of it.
3. **Cross-modal transfer** (train on text, probe on code or vice versa)
   — not implemented in this pass.
4. **Loss-scaling curve** — this pilot is a single run at one data budget;
   the plan's "both models follow a clean power law" criterion needs a
   sweep across data budgets at fixed model size, not done here.
5. **Investigate the BANKING77 vs. triage-department probe gap.** JEPA is
   close to AR on triage department (10-way, in-domain corpus text used
   in pretraining) but far behind on BANKING77 intent (77-way, harder,
   more classes) — worth checking whether this is a fine-grained-label
   problem or a domain-familiarity effect before scaling up.
6. Model checkpoints from this run are saved to `PHASE1_5_RESULTS.pt`
   (gitignored, like other `*.pt` artifacts) so follow-up diagnostics can
   reuse the trained models without a 13-minute retrain.
