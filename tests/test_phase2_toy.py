import numpy as np
import torch

from lattice.phases.phase2_toy import (
    DATA_BUDGET_EPOCHS, SIZES, build_validation_corpus, compute_proxy,
    fit_scaling_law, run_isoflop_cell, _scaling_law,
)
from lattice.toy_corpus import CharTokenizer, build_pretrain_corpus


def test_validation_corpus_disjoint_from_training_real_text():
    train_corpus = build_pretrain_corpus(seed=0, n_code=20, n_json=20,
                                         n_banking=50, n_triage=50)
    val_corpus = build_validation_corpus(seed=999, n_code=20, n_json=20,
                                         n_banking=20, n_triage=20,
                                         train_n_banking=50, train_n_triage=50)
    assert set(train_corpus).isdisjoint(set(val_corpus))


def test_validation_corpus_is_deterministic():
    a = build_validation_corpus(seed=999, n_code=5, n_json=5, n_banking=5,
                                n_triage=5, train_n_banking=10,
                                train_n_triage=10)
    b = build_validation_corpus(seed=999, n_code=5, n_json=5, n_banking=5,
                                n_triage=5, train_n_banking=10,
                                train_n_triage=10)
    assert a == b


def test_compute_proxy_scales_with_params_and_tokens():
    small = compute_proxy(1000, 10000)
    large = compute_proxy(2000, 10000)
    assert large == 2 * small


def test_run_isoflop_cell_returns_expected_shape():
    torch.manual_seed(0)
    texts = ['hello world this is a small toy sentence for testing'] * 40
    tok = CharTokenizer.fit(texts)
    tokens = tok.encode_batch(texts, max_len=32)
    val_tokens = tok.encode_batch(texts[:10], max_len=32)
    size = SIZES[0]
    # Shrink further for a fast unit test -- override via a tiny SizeConfig.
    from lattice.phases.phase2_toy import SizeConfig
    tiny_size = SizeConfig('tiny', dim=8, n_layers=1, n_heads=2)
    result = run_isoflop_cell(tiny_size, epochs=1, train_tokens=tokens,
                              val_tokens=val_tokens,
                              vocab_size=tok.vocab_size, device='cpu')
    assert result['size'] == 'tiny'
    for arm in ('ar', 'ar_jepa_aux'):
        assert result[arm]['num_params'] > 0
        assert result[arm]['val_ce_loss'] > 0
        assert result[arm]['compute_proxy'] > 0


def test_fit_scaling_law_recovers_known_power_law():
    rng = np.random.default_rng(0)
    n = np.array([1e3, 1e3, 1e3, 1e4, 1e4, 1e4, 1e5, 1e5, 1e5])
    d = np.tile([1e3, 1e4, 1e5], 3)
    true_E, true_A, true_alpha, true_B, true_beta = 0.5, 5.0, 0.3, 3.0, 0.4
    loss = _scaling_law((n, d), true_E, true_A, true_alpha, true_B, true_beta)
    loss = loss + rng.normal(0, 0.001, size=loss.shape)  # tiny noise

    records = [
        {'ar': {'num_params': int(ni), 'val_ce_loss': float(li)},
         'tokens_seen': int(di)}
        for ni, di, li in zip(n, d, loss)
    ]
    fit = fit_scaling_law(records, arm='ar')
    assert not fit['fit_failed']
    assert fit['r2'] > 0.99
    assert abs(fit['alpha'] - true_alpha) < 0.1
    assert abs(fit['beta'] - true_beta) < 0.1


def test_sizes_and_budgets_are_nonempty_and_distinct():
    assert len(SIZES) >= 2
    assert len({s.name for s in SIZES}) == len(SIZES)
    assert len(DATA_BUDGET_EPOCHS) >= 2
    assert len(set(DATA_BUDGET_EPOCHS)) == len(DATA_BUDGET_EPOCHS)
