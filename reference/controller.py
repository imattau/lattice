"""Deterministic controller. Spec §3.4.

Routing, thresholds, escalation. Consumes typed readouts; emits actions.
Never parses prose.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from interfaces import (
    ConstraintResult, DecisionType, ProposedAction,
    ReadoutOutput, Stakes, Verdict,
)
from constraint_layer import ConstraintLayer


@dataclass
class ControllerConfig:
    # Escalation threshold on confidence.
    escalation_confidence_threshold: float = 0.3
    # Routing: if confidence below this, ask for more evidence.
    routing_confidence_threshold: float = 0.5
    # Generation invocation threshold — only generate when confident enough.
    generation_confidence_threshold: float = 0.6


class Controller:
    """Deterministic control path. Spec §3.4, §4.4–4.6."""

    def __init__(self, config: ControllerConfig, constraints: ConstraintLayer):
        self.config = config
        self.constraints = constraints

    def decide(
        self,
        readouts: dict[DecisionType, ReadoutOutput],
        stakes: Stakes,
    ) -> tuple[ProposedAction, ConstraintResult]:
        """Convert typed readouts into a proposed action, then constraint-check."""
        urgency = readouts.get(DecisionType.URGENCY)
        routing = readouts.get(DecisionType.ROUTING)

        action_type = "route"
        parameters: dict = {}

        if routing is not None:
            params = routing.distribution.mean(dim=0)
            department = int(params.argmax().item())
            parameters["department"] = f"dept_{department}"
            routing_conf = float(routing.confidence.mean())
        else:
            routing_conf = 0.0

        # Escalation policy
        conf = float(urgency.confidence.mean()) if urgency else routing_conf
        if (stakes == Stakes.HIGH
                and conf < self.config.escalation_confidence_threshold):
            action_type = "escalate"

        proposal = ProposedAction(
            action_type=action_type,
            parameters=parameters,
            decision_type=DecisionType.ROUTING,
            distribution=routing.distribution if routing else torch.zeros(1),
            confidence=torch.tensor([conf]),
            stakes=stakes,
        )
        result = self.constraints.evaluate(proposal)

        if result.verdict == Verdict.REJECT and action_type != "escalate":
            # Deterministic fallback: escalate on rejection.
            fallback = ProposedAction(
                action_type="escalate",
                parameters={},
                decision_type=DecisionType.ESCALATION,
                distribution=torch.zeros(1),
                confidence=torch.tensor([conf]),
                stakes=stakes,
            )
            fallback_result = self.constraints.evaluate(fallback)
            return fallback, fallback_result

        return proposal, result

    def should_generate(self, readout: ReadoutOutput) -> bool:
        """Spec §9.2 — generation invoked on < 20% of inputs."""
        return float(readout.confidence.mean()) >= \
            self.config.generation_confidence_threshold
