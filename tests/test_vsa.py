import torch

from lattice.vsa import (
    BidirectionalPooledBackbone, VSAEncoder, VSAMemoryModule, _unit_spectrum,
    circular_bind, circular_unbind,
)


def test_unit_spectrum_has_unit_magnitude_fourier_coefficients():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    u = _unit_spectrum(x)
    U = torch.fft.rfft(u, dim=-1)
    assert torch.allclose(U.abs(), torch.ones_like(U.abs()), atol=1e-4)


def test_unbind_exactly_recovers_value_for_unit_spectrum_key():
    torch.manual_seed(0)
    key = _unit_spectrum(torch.randn(8, 16))
    value = torch.randn(8, 16)
    bound = circular_bind(key, value)
    recovered = circular_unbind(bound, key)
    assert torch.allclose(recovered, value, atol=1e-4)


def test_unbind_recovers_correct_value_from_a_superposed_memory():
    '''The actual regime VSAMemoryModule uses: multiple (key, value) pairs
    bound and *summed* into one memory vector, then unbound by a specific
    key. Note what this test does NOT expect: similarity to the correct
    value approaching 1 as dimension grows. Forcing keys to unit-magnitude
    spectrum (required for *exact* single-pair unbinding, verified above)
    means each interfering pair's cross-talk has the same norm as its own
    value -- convolving with a unit-spectrum vector preserves L2 norm
    exactly (Parseval), it doesn't average down like ordinary
    small-amplitude noise would. For n superposed pairs the expected
    cosine similarity to the correct value is close to 1/sqrt(n)
    regardless of dimension -- confirmed empirically flat at ~0.5 for
    n=4 from dim=256 up to dim=65536. What must still hold, and does: the
    correct item's key always retrieves a result closer to its own value
    than to any other stored value.'''
    torch.manual_seed(0)
    dim = 256
    keys = [_unit_spectrum(torch.randn(1, dim)) for _ in range(4)]
    values = [torch.randn(1, dim) for _ in range(4)]
    memory = sum(circular_bind(k, v) for k, v in zip(keys, values))

    for i in range(4):
        recovered = circular_unbind(memory, keys[i])
        sims = [torch.cosine_similarity(recovered, v).item() for v in values]
        assert sims[i] == max(sims), (
            f'expected key {i} to best-match its own value; got sims {sims}')
        assert sims[i] > 0.3


def test_vsa_memory_module_shapes_and_gradients_flow():
    torch.manual_seed(0)
    module = VSAMemoryModule(dim=16)
    hidden = torch.randn(3, 5, 16, requires_grad=True)
    out = module(hidden)
    assert out.shape == hidden.shape
    out.sum().backward()
    assert hidden.grad is not None
    assert not torch.allclose(hidden.grad, torch.zeros_like(hidden.grad))


def test_vsa_encoder_forward_shape_and_trainable():
    torch.manual_seed(0)
    enc = VSAEncoder(vocab_size=20, dim=16, n_layers=2, n_heads=2, max_len=8)
    tokens = torch.randint(3, 20, (4, 8))
    pooled = enc(tokens)
    assert pooled.shape == (4, 16)
    loss = pooled.sum()
    loss.backward()
    assert all(p.grad is not None for p in enc.parameters())


def test_bidirectional_pooled_backbone_matches_vsa_encoder_interface():
    torch.manual_seed(0)
    backbone = BidirectionalPooledBackbone(vocab_size=20, dim=16, n_layers=2,
                                           n_heads=2, max_len=8)
    tokens = torch.randint(3, 20, (4, 8))
    pooled = backbone(tokens)
    assert pooled.shape == (4, 16)


def test_vsa_encoder_has_more_params_than_matched_plain_backbone():
    '''The memory module adds parameters (key/value/query/out projections)
    on top of the same-size TinyTransformer -- confirms the two arms are
    NOT parameter-matched by construction, a caveat the write-up must
    state rather than silently comparing unequal-capacity models.'''
    torch.manual_seed(0)
    vsa = VSAEncoder(vocab_size=50, dim=32, n_layers=2, n_heads=2, max_len=16)
    plain = BidirectionalPooledBackbone(vocab_size=50, dim=32, n_layers=2,
                                        n_heads=2, max_len=16)
    assert vsa.num_params() > plain.num_params()
