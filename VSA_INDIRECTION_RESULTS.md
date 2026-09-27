# Exploring Alternatives: VSA Binding Mechanism on the Indirection Task

Status: **the explicit binding primitive is correct (verified), but once
training is properly tuned, VSA-augmented and plain encoders fail
identically — both memorize the 192 training examples perfectly and
collapse to chance-or-below on held-out pairings.** No data-efficiency or
inductive-bias advantage was found for this specific integration of a VSA
memory mechanism, on this specific task, at this scale.

## Motivation

Every Core arm tested in this project — AR, JEPA, contrastive,
AR+JEPA-auxiliary — failed identically at the two-hop indirection task.
Under a frozen probe it floored at chance regardless of textual register
(`PHASE1_5_DEEP_COMPOSITION_RESULTS.md`); under fine-tuning, AR and
AR+JEPA-aux both hit perfect training accuracy and chance-or-below test
accuracy on the same 192 examples
(`PHASE1_5_FINETUNE_INDIRECTION_RESULTS.md`) — textbook memorization, not
learning the general "resolve the pointer, then look up its value" rule.

Following a broader "explore alternatives to the standard Transformer
Core" discussion, this tests a genuinely different mechanism rather than
another loss-function variant on the same architecture: **Vector Symbolic
Architecture (VSA) binding** — an explicit, non-learned *operator* for
"bind A to B, retrieve B given A" (Holographic Reduced Representations,
Plate 1995), motivated directly by the fact that this is exactly the
operation the indirection task requires and exactly what every
architecture tried so far has failed to learn implicitly.

## The binding primitive: verified correct before trusting anything built on it

`lattice/vsa.py` implements the Fourier-domain HRR variant: binding is
circular convolution (elementwise complex multiplication in Fourier
space); unbinding is **exact** — not approximate — when the "key" vector's
Fourier spectrum is constrained to unit magnitude at every frequency (a
"phasor" vector). `tests/test_vsa.py` verifies this directly:
`circular_unbind(circular_bind(key, value), key)` recovers `value` to
within 1e-4, for a single pair.

**A real property surfaced while testing, not glossed over**: for multiple
superposed (key, value) pairs summed into one memory vector, retrieval
accuracy does **not** improve with dimension the way classic Gaussian-HRR
theory predicts. Forcing keys to exact unit-magnitude spectrum (required
for exact single-pair unbinding) means each interfering pair's cross-talk
has the *same* L2 norm as its own value — convolving with a unit-spectrum
vector preserves norm exactly (Parseval's theorem), it doesn't average
down like small-amplitude noise would. For `n` superposed pairs, expected
cosine similarity to the correct value sits near `1/sqrt(n)` regardless of
dimension — confirmed empirically flat at ~0.5 for n=4 from dim=256 up to
dim=65536. The mechanism still always retrieves the *correct* item as the
best match (verified), just with more residual cross-talk than a
dimension-scaling argument would suggest. This matters for the indirection
task specifically: it stores up to 8 bindings (4 color-links + 4
shape-links) per example, so raw (untrained) retrieval capacity is
lower than a naive HRR pitch would imply — the model's key/value
projections would need to *learn* better-than-random structure to do
better than this floor, which is itself part of what's being tested.

## Architecture

`VSAEncoder` (`lattice/vsa.py`): the same bidirectional `TinyTransformer`
backbone used throughout this project, plus a `VSAMemoryModule` combined
residually after the final layer — at every position, project a
(key, value, query) triple; bind each position's key (unit-spectrum) to
its value; sum all bindings into one fixed-size memory vector per example;
at every position, unbind that position's own query against the shared
memory and retrieve whatever was stored there. The key/value/query
projections and the decision of *which* positions to write to and read
from are learned end-to-end via backprop through the FFT operations — the
mechanism provides the binding *operator*, not a hand-coded solution.

`BidirectionalPooledBackbone`: the same backbone alone, as the matched
baseline. **Honest asymmetry, verified by
`test_vsa_encoder_has_more_params_than_matched_plain_backbone`**:
`VSAEncoder` has more trainable parameters (130,304 vs. 113,664 at
dim=64/2 layers/4 heads) because the memory module's four projections add
capacity on top of an identical backbone. Any difference found could in
principle come from extra capacity, not the binding mechanism
specifically — worth remembering when reading the result below, which
found no difference to explain either way.

## An optimization issue found and fixed before trusting the comparison

The first attempt (lr=1e-4, matching the earlier fine-tuning experiments)
showed `VSAEncoder` converging far slower than the plain backbone — loss
still at 2.07 after 5,000 epochs, vs. the plain backbone reaching ~0 within
a few hundred. This was diagnosed before drawing any conclusion from it: a
learning-rate sweep (`--lr 3e-4/1e-3/3e-3`, 500 epochs each) found lr=1e-4
too slow, lr=1e-3 converges properly (train loss 0.114-0.236), and
lr=3e-3 diverges outright (loss exploding to 350) — the FFT-heavy pathway
is more learning-rate-sensitive than the plain backbone, a real practical
cost of this design worth noting on its own, separate from whether the
mechanism helps generalization.

## Result (matched lr=1e-3, 500 epochs, natural-register indirection task, 192 train / 64 test)

| Model | Params | Train joint acc. | Test joint acc. | Final train loss |
|---|---:|---:|---:|---:|
| VSA-augmented | 130,304 | 1.000 | 0.031 | 0.114 |
| Plain backbone | 113,664 | 1.000 | 0.016 | 0.001 |

(Chance = 0.0625.) **Once trained at a learning rate where both converge,
both memorize the training set perfectly and both collapse to
chance-or-below on the 64 held-out pairings — the same failure mode, at
statistically indistinguishable magnitude (0.031 vs. 0.016 at n=64 is well
within noise).** The explicit binding operator did not give a measurable
data-efficiency or generalization advantage over the plain backbone on
this task, at this scale.

## Reading

- **The binding primitive being correct doesn't make the whole learning
  problem easy.** Having access to an exact bind/unbind *operator* doesn't
  eliminate the actual hard part: the model still has to *discover*, from
  only 192 examples and zero prior exposure to anything like this text
  structure, which token positions should become keys and which should
  become values (i.e., that the token right after "is" following a
  single-letter subject is the value to bind to that subject's link, and
  the token right after "is" following a link letter is the value to bind
  to that link). That discovery problem is a generic pattern-learning task
  neither architecture is obviously advantaged on — plain self-attention
  can already implement a similar "attend to an earlier occurrence and
  copy what followed it" operation (the well-documented "induction head"
  behavior in Transformers), so an explicit VSA operator doesn't reduce
  the *search* problem as much as the motivating hypothesis assumed.
- **This is one specific integration of VSA, not a general verdict on
  VSA mechanisms.** A single global memory formed by summing bindings from
  every position, read out via one query per position, is a plausible but
  not the only design. Segregating "assignment" bindings from "query"
  positions explicitly (rather than asking the same generic key/value/query
  projections to discover the distinction unsupervised), using multiple
  memory slots instead of one superposed vector, or providing positional
  hints, might behave differently — this experiment doesn't rule those out.
- **The larger-data check is inconclusive, not a further negative.**
  Repeating the comparison with 4x more data (768 train examples) at the
  same 500-epoch budget found the plain backbone still memorizes easily
  (train loss 0.002) but `VSAEncoder` had **not yet converged** at all
  (train loss 2.741, train accuracy only 0.113) — the FFT-heavy pathway
  appears to need more optimization steps as the dataset grows, which is
  itself a real cost, not evidence either way about generalization at that
  data scale. This needs a longer run before it says anything.
- **Consistent with, and now a fifth line of evidence for, the pattern
  across this entire project**: every architecture and objective tried on
  the indirection task specifically — AR, JEPA, contrastive,
  AR+JEPA-auxiliary, and now VSA-augmented — has failed the same way. The
  common factor across all five is the same 192-example, zero-exposure
  setup, which increasingly looks like the load-bearing variable, not the
  architecture or objective on top of it.

## Follow-ups

1. **Re-run the larger-data (768-example) comparison with enough epochs
   for `VSAEncoder` to actually converge** before treating it as
   inconclusive rather than negative.
2. **Try segregating assignment vs. query roles explicitly** (e.g., a
   learned gate or positional signal marking "this is a binding
   statement" vs. "this is a query") instead of asking generic per-position
   projections to discover that distinction from 192 examples.
3. ~~Test the VSA mechanism on the COGS-style task instead of the
   indirection task~~ — **done, see `VSA_COGS_RESULTS.md`. Result: VSA
   doesn't just fail to help, it clearly loses to the plain backbone on
   both COGS-style tasks (0.266 vs 0.406, and 0.250 vs 0.750,
   replicated) — the tie-at-floor on indirection was not "no difference
   detectable," it was masking a real cost that shows up once there's
   signal to lose.**
4. Given five architectures now share this exact failure mode on the same
   task, the more informative next move may be addressing the *data*
   side directly (per `PHASE1_5_DEEP_COMPOSITION_RESULTS.md`'s
   follow-up: give the corpus real exposure to indirection-style
   structure before evaluating any architecture on it) rather than
   continuing to vary the architecture against a task no architecture has
   ever been given a fair chance to learn from.
