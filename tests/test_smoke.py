'''Smoke tests for core Lattice components.'''
from __future__ import annotations
import torch

from lattice.interfaces import DecisionType, Stakes
from lattice.readout_heads import ReadoutBank, ReadoutHead
from lattice.constraint_layer import ConstraintLayer, default_policy
from lattice.controller import Controller, ControllerConfig
from lattice.calibration import expected_calibration_error, fit_temperature, rlcd_surrogate
from lattice.masking import multi_block_mask
from lattice.models import JEPACore, ARBaseline, ModelConfig as _ModelConfig


def test_readout_head_shapes():
    head = ReadoutHead(hidden_dim=32, num_options=4,
                       decision_type=DecisionType.ROUTING)
    x = torch.randn(8, 32)
    out = head(x)
    assert out.distribution.shape == (8, 4)
    assert torch.allclose(out.distribution.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_readout_bank():
    bank = ReadoutBank(
        hidden_dim=32,
        specs={DecisionType.ROUTING: 4, DecisionType.URGENCY: 3},
    )
    x = torch.randn(8, 32)
    outs = bank(x)
    assert DecisionType.ROUTING in outs
    assert DecisionType.URGENCY in outs


def test_constraint_layer_default_policy():
    layer = ConstraintLayer(default_policy())
    from lattice.interfaces import ProposedAction, DecisionType
    action = ProposedAction(
        action_type='route',
        parameters={'department': 'blacklisted_dept'},
        decision_type=DecisionType.ROUTING,
        distribution=torch.tensor([[0.5, 0.5]]),
        confidence=torch.tensor([0.5]),
        stakes=Stakes.LOW,
    )
    result = layer.evaluate(action)
    assert result.verdict.value == 'reject'


def test_controller_routes_or_escalates():
    layer = ConstraintLayer(default_policy())
    ctrl = Controller(ControllerConfig(), layer)
    from lattice.interfaces import ReadoutOutput
    readout = ReadoutOutput(
        decision_type=DecisionType.ROUTING,
        distribution=torch.tensor([[0.7, 0.3]]),
        confidence=torch.tensor([0.7]),
        temperature=torch.tensor([1.0]),
        ood_score=torch.tensor([-1.0]),
    )
    action, result = ctrl.decide({DecisionType.ROUTING: readout}, stakes=Stakes.LOW)
    assert action.action_type == 'route'
    assert result.verdict.value == 'approve'


def test_calibration_ece():
    probs = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]])
    labels = torch.tensor([0, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_multi_block_mask():
    mask = multi_block_mask(batch_size=2, seq_len=64, num_blocks=2, device='cpu')
    assert mask.shape == (2, 64)
    assert mask.dtype == torch.bool


def test_jepa_core_forward():
    cfg = _ModelConfig(vocab_size=100, hidden_dim=64, num_reasoner_blocks=2,
                       num_talker_blocks=1, num_heads=4, max_seq_len=32)
    core = JEPACore(cfg)
    ids = torch.randint(0, 100, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.bool)
    latent = core.context_encode(ids, attn)
    assert latent.shape == (2, 16, 64)


def test_ar_baseline_forward():
    cfg = _ModelConfig(vocab_size=100, hidden_dim=64, num_reasoner_blocks=2,
                       num_talker_blocks=1, num_heads=4, max_seq_len=32)
    model = ARBaseline(cfg)
    ids = torch.randint(0, 100, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 100)
