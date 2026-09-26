import torch

from lattice.phases.phase1_5_toy import (
    _COGS_DEPARTMENTS, _COGS_HELD_OUT, _COGS_URGENCY, _COLORS, _SHAPES,
    ToyConfig, build_ar_jepa, build_composition_task, build_cogs_style_task,
    build_contrastive, build_deep_composition_task, build_jepa,
    composition_accuracy, effective_rank, encode_pooled_ar,
    encode_pooled_contrastive, encode_pooled_jepa, info_nce_loss,
    linear_probe_accuracy, make_augmented_view, make_block_mask, train_ar,
    train_ar_jepa, train_contrastive, train_jepa, uniformity,
)
from lattice.tiny_transformer import (
    ARLanguageModel, Predictor, TinyTransformer, ema_update, tau_schedule,
)
from lattice.toy_corpus import CharTokenizer, build_pretrain_corpus


def test_char_tokenizer_roundtrip_shape():
    tok = CharTokenizer.fit(['hello world', 'abc'])
    ids = tok.encode('hello', max_len=10)
    assert len(ids) == 10
    assert all(i != CharTokenizer.PAD for i in ids[:5])
    assert all(i == CharTokenizer.PAD for i in ids[5:])


def test_char_tokenizer_unknown_char_maps_to_unk():
    tok = CharTokenizer.fit(['abc'])
    ids = tok.encode('abz', max_len=3)
    assert ids[2] == CharTokenizer.UNK


def test_build_pretrain_corpus_is_deterministic_and_diverse():
    a = build_pretrain_corpus(seed=0, n_code=10, n_json=10, n_banking=10,
                              n_triage=10)
    b = build_pretrain_corpus(seed=0, n_code=10, n_json=10, n_banking=10,
                              n_triage=10)
    assert a == b
    assert len(a) == 40


def test_tiny_transformer_forward_shapes():
    model = TinyTransformer(vocab_size=20, dim=16, n_layers=2, n_heads=2,
                            max_len=8)
    tokens = torch.randint(3, 20, (4, 8))
    out = model(tokens, causal=False)
    assert out.shape == (4, 8, 16)
    out_causal = model(tokens, causal=True)
    assert out_causal.shape == (4, 8, 16)


def test_forward_layers_returns_one_hidden_state_per_block_and_matches_forward():
    model = TinyTransformer(vocab_size=20, dim=16, n_layers=4, n_heads=2,
                            max_len=8)
    tokens = torch.randint(3, 20, (4, 8))
    hiddens = model.forward_layers(tokens, causal=False)
    assert len(hiddens) == 4
    for h in hiddens:
        assert h.shape == (4, 8, 16)
    # Last layer must match plain forward() (which applies ln_f at the top).
    assert torch.allclose(hiddens[-1], model(tokens, causal=False))
    # Earlier layers must differ from the top (not a no-op stack).
    assert not torch.allclose(hiddens[0], hiddens[-1])


def test_ar_model_trains_and_loss_decreases():
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['hello world this is a test of the ar model'])
    tokens = tok.encode_batch(['hello world this is a test'] * 32, max_len=16)
    backbone = TinyTransformer(tok.vocab_size, dim=16, n_layers=2, n_heads=2,
                               max_len=16)
    model = ARLanguageModel(backbone)
    config = ToyConfig(dim=16, n_layers=2, n_heads=2, max_len=16,
                       batch_size=8, epochs=5, lr=1e-2)
    history = train_ar(model, tokens, config, device='cpu')
    assert history[-1] < history[0]


def test_ema_update_moves_target_toward_source():
    a = TinyTransformer(10, dim=8, n_layers=1, n_heads=2, max_len=4)
    b = TinyTransformer(10, dim=8, n_layers=1, n_heads=2, max_len=4)
    before = next(a.parameters()).clone()
    ema_update(a, b, tau=0.0)  # tau=0 -> fully replace with source
    after = next(a.parameters())
    expected = next(b.parameters())
    assert torch.allclose(after, expected)
    assert not torch.allclose(before, after)


def test_tau_schedule_endpoints():
    assert tau_schedule(0, 100, start=0.9, end=1.0) == 0.9
    assert abs(tau_schedule(99, 100, start=0.9, end=1.0) - 1.0) < 1e-6


def test_make_block_mask_respects_scale_bounds():
    import random
    rng = random.Random(0)
    mask = make_block_mask(batch_size=5, seq_len=100,
                           scale_range=(0.15, 0.20), rng=rng)
    counts = mask.sum(dim=1)
    assert (counts >= 14).all() and (counts <= 21).all()


def test_jepa_trains_and_loss_decreases():
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['hello world this is a test of the jepa model'])
    tokens = tok.encode_batch(['hello world this is a test'] * 32, max_len=16)
    config = ToyConfig(dim=16, n_layers=2, n_heads=2, max_len=16,
                       batch_size=8, epochs=5, lr=1e-2)
    jepa = build_jepa(tok.vocab_size, config)
    history = train_jepa(jepa, tokens, config, device='cpu')
    # cosine loss is in [-1, 1]; just check it moves down from its start.
    assert history[-1] < history[0]


def test_linear_probe_learns_separable_classes():
    torch.manual_seed(0)
    n = 200
    dim = 4
    labels = torch.randint(0, 3, (n,))
    features = F_one_hot_features(labels, dim)
    acc = linear_probe_accuracy(features[:150], labels[:150],
                                features[150:], labels[150:], num_classes=3,
                                epochs=100)
    assert acc > 0.9


def F_one_hot_features(labels: torch.Tensor, dim: int) -> torch.Tensor:
    import torch.nn.functional as F
    base = F.one_hot(labels, num_classes=dim).float()
    return base + 0.01 * torch.randn_like(base)


def test_composition_task_holds_out_pairings_but_not_attribute_values():
    tok = CharTokenizer.fit(['the red circle is on the table a blue square '
                             'this green triangle that yellow star'])
    split = build_composition_task(tok, max_len=32)
    assert split.train_tokens.size(0) > 0 and split.test_tokens.size(0) > 0
    # Every held-out attribute value must still appear in train (otherwise
    # a linear probe could never predict it, regardless of representation
    # quality -- this was the bug in an earlier version of this task).
    assert set(split.test_color.tolist()) <= set(split.train_color.tolist())
    assert set(split.test_shape.tolist()) <= set(split.train_shape.tolist())
    # But the exact (color, shape) pairings in test must be absent from train.
    train_pairs = set(zip(split.train_color.tolist(), split.train_shape.tolist()))
    test_pairs = set(zip(split.test_color.tolist(), split.test_shape.tolist()))
    assert train_pairs.isdisjoint(test_pairs)


def test_deep_composition_task_is_deterministic():
    tok = CharTokenizer.fit(['x=p y=m p=red q=blue r=green s=yellow '
                             'm=circle n=square o=triangle w=star'])
    a = build_deep_composition_task(tok, max_len=96, n_per_combo=4, seed=0)
    b = build_deep_composition_task(tok, max_len=96, n_per_combo=4, seed=0)
    assert torch.equal(a.train_tokens, b.train_tokens)
    assert torch.equal(a.train_color, b.train_color)


def test_deep_composition_task_removes_lexical_shortcut():
    '''Every color and shape word must appear as a substring in every
    example, train and test alike -- otherwise a bag-of-words probe could
    solve the task without doing the two-hop binding resolution it's meant
    to require (the flaw the deep task exists to fix in the original
    lexically-transparent composition task).'''
    tok = CharTokenizer.fit(['x=p y=m p=red q=blue r=green s=yellow '
                             'm=circle n=square o=triangle w=star'])
    split = build_deep_composition_task(tok, max_len=96, n_per_combo=4, seed=1)
    # Decode a handful of raw texts back out via the tokenizer's own vocab
    # is unnecessary here -- regenerate directly to inspect text content.
    import random
    from lattice.phases.phase1_5_toy import _gen_deep_composition_text
    rng = random.Random(1)
    for _ in range(20):
        text = _gen_deep_composition_text(rng, 0, 0)
        for color in _COLORS:
            assert color in text
        for shape in _SHAPES:
            assert shape in text


def test_deep_composition_task_holds_out_pairings_but_not_values():
    tok = CharTokenizer.fit(['x=p y=m p=red q=blue r=green s=yellow '
                             'm=circle n=square o=triangle w=star'])
    split = build_deep_composition_task(tok, max_len=96, n_per_combo=4, seed=0)
    assert set(split.test_color.tolist()) <= set(split.train_color.tolist())
    assert set(split.test_shape.tolist()) <= set(split.train_shape.tolist())
    train_pairs = set(zip(split.train_color.tolist(), split.train_shape.tolist()))
    test_pairs = set(zip(split.test_color.tolist(), split.test_shape.tolist()))
    assert train_pairs.isdisjoint(test_pairs)


def test_deep_composition_task_natural_style_removes_lexical_shortcut():
    '''Natural-language register must preserve the anti-shortcut property:
    every color/shape word present in every example regardless of label.'''
    import random
    from lattice.phases.phase1_5_toy import _gen_deep_composition_text_natural
    rng = random.Random(2)
    for _ in range(20):
        text = _gen_deep_composition_text_natural(rng, 1, 2)
        for color in _COLORS:
            assert color in text
        for shape in _SHAPES:
            assert shape in text
        # Ordinary declarative sentences, not dense assignment syntax.
        assert '=' not in text
        assert '.' in text


def test_deep_composition_task_natural_style_fits_toy_context_length():
    tok = CharTokenizer.fit(['p is red. q is blue. r is green. s is yellow. '
                             'm is circle. n is square. o is triangle. '
                             'w is star. x is p. y is m.'])
    split = build_deep_composition_task(tok, max_len=192, n_per_combo=4,
                                        seed=0, style='natural')
    # No example should have been silently truncated by the 192-char
    # context length the toy models were trained with -- if it were, the
    # tokenizer would have dropped the "x is ..." / "y is ..." suffix that
    # carries the answer.
    assert (split.train_tokens != 0).sum(dim=1).max() < 192


def test_deep_composition_task_style_is_deterministic_per_style():
    tok = CharTokenizer.fit(['p is red. q is blue. r is green. s is yellow. '
                             'm is circle. n is square. o is triangle. '
                             'w is star. x is p. y is m.'])
    a = build_deep_composition_task(tok, max_len=192, n_per_combo=4, seed=0,
                                    style='natural')
    b = build_deep_composition_task(tok, max_len=192, n_per_combo=4, seed=0,
                                    style='natural')
    assert torch.equal(a.train_tokens, b.train_tokens)


def test_cogs_task_primitives_are_present_in_the_actual_pretraining_corpus():
    '''The whole point of this task vs. the deep-composition task: its
    vocabulary must be verifiably present in the real pretraining corpus
    these checkpoints were trained on, not merely asserted to be.'''
    corpus_blob = ' '.join(build_pretrain_corpus(seed=0))
    for dept in _COGS_DEPARTMENTS:
        assert dept in corpus_blob
    for phrase in _COGS_URGENCY:
        assert phrase in corpus_blob


def test_cogs_task_holds_out_pairings_but_not_primitives():
    tok = CharTokenizer.fit(['I have a question about billing. please '
                             'escalate right away. route this to support.'])
    split = build_cogs_style_task(tok, max_len=192, n_per_combo=4, seed=0)
    assert set(split.test_color.tolist()) <= set(split.train_color.tolist())
    assert set(split.test_shape.tolist()) <= set(split.train_shape.tolist())
    train_pairs = set(zip(split.train_color.tolist(), split.train_shape.tolist()))
    test_pairs = set(zip(split.test_color.tolist(), split.test_shape.tolist()))
    assert train_pairs.isdisjoint(test_pairs)
    assert test_pairs == _COGS_HELD_OUT


def test_cogs_task_is_deterministic():
    tok = CharTokenizer.fit(['I have a question about billing. please '
                             'escalate right away. route this to support.'])
    a = build_cogs_style_task(tok, max_len=192, n_per_combo=4, seed=0)
    b = build_cogs_style_task(tok, max_len=192, n_per_combo=4, seed=0)
    assert torch.equal(a.train_tokens, b.train_tokens)


def test_composition_accuracy_infers_class_count_from_labels():
    '''composition_accuracy must not hardcode against _COLORS/_SHAPES --
    it needs to work for any two-attribute CompositionSplit, e.g. the COGS
    task's 4 departments x 4 urgency phrases.'''
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['I have a question about billing. please '
                             'escalate right away. route this to support.'])
    split = build_cogs_style_task(tok, max_len=192, n_per_combo=4, seed=0)
    train_features = torch.randn(split.train_tokens.size(0), 8)
    test_features = torch.randn(split.test_tokens.size(0), 8)
    result = composition_accuracy(train_features, split, test_features,
                                  epochs=10)
    assert 0.0 <= result['joint_accuracy'] <= 1.0


def test_composition_accuracy_recovers_separable_attributes():
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['the red circle is on the table a blue square '
                             'this green triangle that yellow star'])
    split = build_composition_task(tok, max_len=32)
    # Features that cleanly encode color/shape as one-hot blocks: a
    # representation with genuine compositional structure should let both
    # attribute probes -- and therefore the joint accuracy -- succeed on
    # held-out pairings.
    def fake_features(colors, shapes):
        c = torch.nn.functional.one_hot(colors, 4).float()
        s = torch.nn.functional.one_hot(shapes, 4).float()
        feats = torch.cat([c, s], dim=-1)
        return feats + 0.01 * torch.randn_like(feats)

    train_feats = fake_features(split.train_color, split.train_shape)
    test_feats = fake_features(split.test_color, split.test_shape)
    result = composition_accuracy(train_feats, split, test_feats, epochs=200)
    assert result['color_accuracy'] > 0.9
    assert result['shape_accuracy'] > 0.9
    assert result['joint_accuracy'] > 0.8


def test_augmented_view_pads_outside_crop_and_preserves_length():
    import random
    rng = random.Random(0)
    batch = torch.randint(3, 20, (4, 20))
    view = make_augmented_view(batch, rng, crop_range=(0.5, 0.5), mask_rate=0.0)
    assert view.shape == batch.shape
    for b in range(4):
        n_pad = (view[b] == 0).sum().item()
        assert n_pad >= 10 - 1  # ~half the sequence padded out (off-by-one ok)


def test_info_nce_loss_lower_for_aligned_positive_pairs():
    torch.manual_seed(0)
    B, D = 8, 16
    base = torch.randn(B, D)
    aligned = base + 0.01 * torch.randn(B, D)
    misaligned = base[torch.randperm(B)] + 0.01 * torch.randn(B, D)
    aligned_loss = info_nce_loss(base, aligned)
    misaligned_loss = info_nce_loss(base, misaligned)
    assert aligned_loss.item() < misaligned_loss.item()


def test_contrastive_trains_and_loss_decreases():
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['hello world this is a test of the contrastive '
                             'model training loop end to end'])
    tokens = tok.encode_batch(['hello world this is a test'] * 32, max_len=16)
    config = ToyConfig(dim=16, n_layers=2, n_heads=2, max_len=16,
                       batch_size=8, epochs=5, lr=1e-2)
    model = build_contrastive(tok.vocab_size, config)
    history = train_contrastive(model, tokens, config, device='cpu')
    assert history[-1] < history[0]
    feats = encode_pooled_contrastive(model, tokens[:8], device='cpu')
    assert feats.shape == (8, config.dim)


def test_ar_jepa_trains_and_both_loss_terms_decrease():
    torch.manual_seed(0)
    tok = CharTokenizer.fit(['hello world this is a test of the ar jepa '
                             'auxiliary loss training loop end to end'])
    tokens = tok.encode_batch(['hello world this is a test'] * 32, max_len=16)
    config = ToyConfig(dim=16, n_layers=2, n_heads=2, max_len=16,
                       batch_size=8, epochs=8, lr=1e-2, ar_jepa_alpha=0.5)
    model = build_ar_jepa(tok.vocab_size, config)
    history = train_ar_jepa(model, tokens, config, device='cpu')
    ce_start = sum(h['ce'] for h in history[:4]) / 4
    ce_end = sum(h['ce'] for h in history[-4:]) / 4
    latent_start = sum(h['latent'] for h in history[:4]) / 4
    latent_end = sum(h['latent'] for h in history[-4:]) / 4
    assert ce_end < ce_start
    assert latent_end < latent_start
    feats = encode_pooled_ar(model.ar_model, tokens[:8], device='cpu')
    assert feats.shape == (8, config.dim)


def test_uniformity_and_effective_rank_run():
    z = torch.randn(50, 8)
    u = uniformity(z)
    r = effective_rank(z)
    assert isinstance(u, float)
    assert 1.0 <= r <= 8.0
