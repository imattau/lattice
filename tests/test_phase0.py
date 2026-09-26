'''Unit tests for the Phase 0 pipeline (synthetic, no downloads).'''
from __future__ import annotations

import torch

from lattice.constraint_layer import ConstraintLayer
from lattice.controller import Controller, ControllerConfig
from lattice.interfaces import DecisionType, Stakes
from lattice.phases.phase0 import (
    _build_policy, _evaluate, _measure_latency, train_readout,
)
from lattice.readout_heads import ReadoutHead


def test_train_readout_and_evaluate():
    features = torch.randn(200, 32)
    labels = (features[:, 0] > 0).long()
    head = train_readout(features, labels, num_classes=2, epochs=20, lr=1e-2)
    metrics = _evaluate(head, features, labels, num_classes=2)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['ece_raw'] <= 1.0
    assert 0.0 <= metrics['ece_cal'] <= 1.0
    assert metrics['brier'] >= 0.0


def test_latency_positive():
    head = ReadoutHead(32, 2, DecisionType.ROUTING)
    features = torch.randn(16, 32)
    lat = _measure_latency(head, features, n=10)
    assert lat > 0.0


def test_phase0_constraint_policy():
    policy = _build_policy()
    layer = ConstraintLayer(policy)
    from lattice.interfaces import ProposedAction
    action = ProposedAction(
        action_type='route',
        parameters={'department': 'dept_0'},
        decision_type=DecisionType.ROUTING,
        distribution=torch.tensor([[0.5, 0.5]]),
        confidence=torch.tensor([0.2]),
        stakes=Stakes.HIGH,
    )
    result = layer.evaluate(action)
    assert result.verdict.value == 'reject'


def test_controller_decide():
    policy = _build_policy()
    controller = Controller(ControllerConfig(), ConstraintLayer(policy))
    from lattice.interfaces import ReadoutOutput
    readout = ReadoutOutput(
        decision_type=DecisionType.ROUTING,
        distribution=torch.tensor([[0.8, 0.2]]),
        confidence=torch.tensor([0.8]),
        temperature=torch.tensor([1.0]),
        ood_score=torch.tensor([-1.0]),
    )
    proposal, result = controller.decide(
        {DecisionType.ROUTING: readout}, stakes=Stakes.LOW,
    )
    assert proposal.action_type == 'route'
    assert result.verdict.value == 'approve'
