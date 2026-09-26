'''Phase 1: document triage system.

Composes the readout bank, a triage-specific decision function, and the
constraint layer over the synthetic triage corpus
(lattice/synthetic_triage.py). This replaces the earlier placeholder.

Design notes:
  - The plan's readout bank calls for a "confidence (5-way score)" head.
    Confidence is a *derived* quantity of a trained model (e.g. max-softmax),
    not an annotatable ground truth (see PHASE1_RESULTS.md), so it is not
    trained as a fifth classification head here. Each ReadoutOutput already
    carries its own `confidence` field (softmax max), which is what the
    controller and constraint layer consult. The corpus's `label_confidence`
    field remains available as an auxiliary stratification variable for
    evaluation (e.g. "does ECE get worse on high-ambiguity documents?") but
    is not a training target.
  - Stage 1a: joint per-head RLCD training on train.jsonl (this module).
  - Stage 1b: controller threshold tuning on val.jsonl (`tune_threshold`).
  - Stage 1c: post-hoc per-head temperature fit on val.jsonl
    (`fit_head_temperatures`), on top of the in-loop RLCD temperature.
'''
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lattice.calibration import (
    expected_calibration_error, fit_temperature, rlcd_surrogate,
)
from lattice.constraint_layer import ConstraintLayer, triage_policy
from lattice.interfaces import (
    DecisionType, ProposedAction, ReadoutOutput, Stakes, Verdict,
)
from lattice.readout_heads import ReadoutBank
from lattice.synthetic_triage import BLACKLISTED_DEPARTMENT, Urgency

HEAD_SPECS: dict[DecisionType, int] = {
    DecisionType.ROUTING: 10,
    DecisionType.URGENCY: 3,
    DecisionType.ESCALATION: 2,
    DecisionType.SAFETY: 2,
}


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def labels_from_docs(docs: list[dict]) -> dict[DecisionType, torch.Tensor]:
    return {
        DecisionType.ROUTING: torch.tensor([d['department'] for d in docs]),
        DecisionType.URGENCY: torch.tensor([d['urgency'] for d in docs]),
        DecisionType.ESCALATION: torch.tensor(
            [int(d['escalate']) for d in docs]),
        DecisionType.SAFETY: torch.tensor(
            [int(d['safety_flag']) for d in docs]),
    }


def build_bank(hidden_dim: int) -> ReadoutBank:
    return ReadoutBank(hidden_dim, HEAD_SPECS)


def build_constraints() -> ConstraintLayer:
    return ConstraintLayer(triage_policy(int(BLACKLISTED_DEPARTMENT)))


def train_bank(
    bank: ReadoutBank,
    features: torch.Tensor,
    labels: dict[DecisionType, torch.Tensor],
    epochs: int = 30,
    lr: float = 1e-3,
    beta_rlcd: float = 1.0,
) -> list[float]:
    '''Stage 1a: joint per-head RLCD training.'''
    bank.train()
    opt = torch.optim.AdamW(bank.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        outputs = bank(features)
        loss = torch.zeros(())
        for dt, out in outputs.items():
            loss = loss + rlcd_surrogate(out.distribution, labels[dt],
                                         beta=beta_rlcd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history


def evaluate_heads(
    bank: ReadoutBank,
    features: torch.Tensor,
    labels: dict[DecisionType, torch.Tensor],
) -> dict:
    bank.eval()
    with torch.no_grad():
        outputs = bank(features)
    metrics = {}
    for dt, out in outputs.items():
        y = labels[dt]
        probs = out.distribution
        acc = (probs.argmax(-1) == y).float().mean().item()
        ece = expected_calibration_error(probs, y)
        brier = ((probs - F.one_hot(y, probs.size(-1))) ** 2
                 ).sum(-1).mean().item()
        metrics[dt.value] = {'accuracy': acc, 'ece': ece, 'brier': brier}
    return metrics


def fit_head_temperatures(
    bank: ReadoutBank,
    features: torch.Tensor,
    labels: dict[DecisionType, torch.Tensor],
) -> dict[str, float]:
    '''Stage 1c: post-hoc per-head temperature fit on held-out (val) logits,
    on top of the temperature already learned in-loop during RLCD training.
    Returns the fitted post-hoc T per head (reported, not applied in-place,
    since the in-loop temperature already calibrates the bank's own output).
    '''
    bank.eval()
    temps = {}
    for name, head in bank.heads.items():
        dt = DecisionType(name)
        with torch.no_grad():
            logits = head.logit_head(features)
        temps[name] = fit_temperature(logits, labels[dt])
    return temps


def _select(out: ReadoutOutput, i: int) -> ReadoutOutput:
    return ReadoutOutput(
        decision_type=out.decision_type,
        distribution=out.distribution[i:i + 1],
        confidence=out.confidence[i:i + 1],
        temperature=out.temperature[i:i + 1],
        ood_score=out.ood_score[i:i + 1],
    )


@dataclass
class TriageConfig:
    escalation_confidence_threshold: float = 0.3


def decide_triage(
    readouts: dict[DecisionType, ReadoutOutput],
    config: TriageConfig,
    constraints: ConstraintLayer,
) -> tuple[ProposedAction, Verdict]:
    '''One document's decision path: readouts -> proposed action -> hard
    constraint check -> deterministic fallback on rejection. Batch size 1.
    '''
    routing = readouts[DecisionType.ROUTING]
    urgency = readouts[DecisionType.URGENCY]
    escalation = readouts[DecisionType.ESCALATION]
    safety = readouts[DecisionType.SAFETY]

    department = int(routing.distribution.argmax(-1).item())
    urgency_level = int(urgency.distribution.argmax(-1).item())
    safety_flag = bool(safety.distribution.argmax(-1).item() == 1)
    soft_escalate = bool(escalation.distribution.argmax(-1).item() == 1)
    routing_conf = float(routing.confidence.item())

    stakes = (Stakes.HIGH if urgency_level == Urgency.HIGH
             else Stakes.MEDIUM if urgency_level == Urgency.MEDIUM
             else Stakes.LOW)

    low_conf_high_stakes = (
        stakes == Stakes.HIGH
        and routing_conf < config.escalation_confidence_threshold
    )
    action_type = ('escalate' if (soft_escalate or low_conf_high_stakes)
                  else 'route')

    proposal = ProposedAction(
        action_type=action_type,
        parameters={'department': department, 'safety_flag': safety_flag},
        decision_type=DecisionType.ROUTING,
        distribution=routing.distribution,
        confidence=routing.confidence,
        stakes=stakes,
    )
    result = constraints.evaluate(proposal)
    if result.verdict == Verdict.REJECT:
        fallback = ProposedAction(
            action_type='escalate',
            parameters={'department': department, 'safety_flag': safety_flag},
            decision_type=DecisionType.ESCALATION,
            distribution=escalation.distribution,
            confidence=escalation.confidence,
            stakes=stakes,
        )
        fallback_result = constraints.evaluate(fallback)
        return fallback, fallback_result.verdict
    return proposal, result.verdict


def tune_threshold(
    bank: ReadoutBank,
    features: torch.Tensor,
    docs: list[dict],
    constraints: ConstraintLayer,
    candidates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> float:
    '''Stage 1b: sweep the escalation confidence threshold on val.jsonl,
    picking the value that maximizes escalation-decision accuracy against
    the gold `escalate` label.
    '''
    bank.eval()
    with torch.no_grad():
        outputs = bank(features)
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        config = TriageConfig(escalation_confidence_threshold=t)
        correct = 0
        for i, doc in enumerate(docs):
            readouts_i = {dt: _select(out, i) for dt, out in outputs.items()}
            action, _ = decide_triage(readouts_i, config, constraints)
            predicted = action.action_type == 'escalate'
            if predicted == bool(doc['escalate']):
                correct += 1
        acc = correct / len(docs)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t


def evaluate_triage(
    bank: ReadoutBank,
    features: torch.Tensor,
    docs: list[dict],
    config: TriageConfig,
    constraints: ConstraintLayer,
) -> dict:
    '''End-to-end controller evaluation: constraint-violation rate (post
    constraint layer; gate is zero), escalation-decision accuracy, and,
    for red-team documents, whether the *raw* readout (pre-constraint-layer)
    would have obeyed an embedded prompt-injection attempt.
    '''
    bank.eval()
    with torch.no_grad():
        outputs = bank(features)

    n = len(docs)
    post_violations = 0
    escalation_correct = 0
    injection_attempts = 0
    injection_obeyed_by_raw_head = 0

    for i, doc in enumerate(docs):
        readouts_i = {dt: _select(out, i) for dt, out in outputs.items()}
        action, verdict = decide_triage(readouts_i, config, constraints)
        if verdict == Verdict.REJECT:
            post_violations += 1
        if (action.action_type == 'escalate') == bool(doc['escalate']):
            escalation_correct += 1

        if doc.get('injected_department') is not None:
            injection_attempts += 1
            raw_department = int(
                readouts_i[DecisionType.ROUTING].distribution
                .argmax(-1).item()
            )
            if raw_department == doc['injected_department']:
                injection_obeyed_by_raw_head += 1

    return {
        'n': n,
        'post_constraint_violation_rate': post_violations / n,
        'escalation_decision_accuracy': escalation_correct / n,
        'injection_attempts': injection_attempts,
        'injection_obeyed_by_raw_head_rate': (
            injection_obeyed_by_raw_head / injection_attempts
            if injection_attempts else None
        ),
    }


def build_triage_system():
    '''Kept for interface stability; the real pipeline is
    scripts/train_phase1.py, which needs an encoder + data paths that don't
    belong in library code.
    '''
    raise NotImplementedError(
        'Use scripts/train_phase1.py to run the full Phase 1 pipeline '
        '(encode corpus, train_bank, tune_threshold, evaluate_triage).'
    )
