import torch

from lattice.constraint_layer import ConstraintLayer, triage_policy
from lattice.interfaces import DecisionType, ReadoutOutput, Verdict
from lattice.phases.phase1_triage import (
    HEAD_SPECS, build_bank, decide_triage, evaluate_heads, labels_from_docs,
    train_bank, TriageConfig,
)
from lattice.synthetic_triage import BLACKLISTED_DEPARTMENT, generate_corpus, \
    document_to_dict


def _readout(decision_type, probs) -> ReadoutOutput:
    dist = torch.tensor([probs])
    return ReadoutOutput(
        decision_type=decision_type,
        distribution=dist,
        confidence=dist.max(dim=-1).values,
        temperature=torch.ones(1),
        ood_score=torch.zeros(1),
    )


def _constraints():
    return ConstraintLayer(triage_policy(int(BLACKLISTED_DEPARTMENT)))


def test_never_routes_to_blacklisted_department():
    blacklisted = int(BLACKLISTED_DEPARTMENT)
    routing_probs = [0.0] * 10
    routing_probs[blacklisted] = 1.0
    readouts = {
        DecisionType.ROUTING: _readout(DecisionType.ROUTING, routing_probs),
        DecisionType.URGENCY: _readout(DecisionType.URGENCY, [1.0, 0.0, 0.0]),
        DecisionType.ESCALATION: _readout(DecisionType.ESCALATION, [1.0, 0.0]),
        DecisionType.SAFETY: _readout(DecisionType.SAFETY, [1.0, 0.0]),
    }
    action, verdict = decide_triage(readouts, TriageConfig(), _constraints())
    assert verdict == Verdict.APPROVE
    assert action.action_type == 'escalate'


def test_forces_escalation_on_safety_at_high_stakes():
    readouts = {
        DecisionType.ROUTING: _readout(
            DecisionType.ROUTING, [1.0] + [0.0] * 9),
        DecisionType.URGENCY: _readout(DecisionType.URGENCY, [0.0, 0.0, 1.0]),
        # Soft escalation head wrongly says "do not escalate".
        DecisionType.ESCALATION: _readout(DecisionType.ESCALATION, [1.0, 0.0]),
        DecisionType.SAFETY: _readout(DecisionType.SAFETY, [0.0, 1.0]),
    }
    action, verdict = decide_triage(readouts, TriageConfig(), _constraints())
    assert verdict == Verdict.APPROVE
    assert action.action_type == 'escalate'


def test_clean_high_confidence_routes_normally():
    readouts = {
        DecisionType.ROUTING: _readout(
            DecisionType.ROUTING, [1.0] + [0.0] * 9),
        DecisionType.URGENCY: _readout(DecisionType.URGENCY, [1.0, 0.0, 0.0]),
        DecisionType.ESCALATION: _readout(DecisionType.ESCALATION, [1.0, 0.0]),
        DecisionType.SAFETY: _readout(DecisionType.SAFETY, [1.0, 0.0]),
    }
    action, verdict = decide_triage(readouts, TriageConfig(), _constraints())
    assert verdict == Verdict.APPROVE
    assert action.action_type == 'route'
    assert action.parameters['department'] == 0


def test_bank_trains_and_improves_loss():
    docs = [document_to_dict(d) for d in generate_corpus(200, seed=0)]
    labels = labels_from_docs(docs)
    features = torch.randn(200, 32)
    bank = build_bank(hidden_dim=32)
    history = train_bank(bank, features, labels, epochs=5, lr=1e-2)
    assert len(history) == 5
    assert all(dt in HEAD_SPECS for dt in labels)


def test_evaluate_heads_reports_all_heads():
    docs = [document_to_dict(d) for d in generate_corpus(50, seed=1)]
    labels = labels_from_docs(docs)
    features = torch.randn(50, 16)
    bank = build_bank(hidden_dim=16)
    metrics = evaluate_heads(bank, features, labels)
    assert set(metrics) == {dt.value for dt in HEAD_SPECS}
    for m in metrics.values():
        assert 0.0 <= m['accuracy'] <= 1.0
        assert 0.0 <= m['ece'] <= 1.0
