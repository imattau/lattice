'''Constraint layer. Spec §8.1.

Hard constraints are formal, machine-checkable rules in a read-only
module. The policy is hashed at load; a mismatch halts the system.
'''
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Callable

from lattice.interfaces import (
    ConstraintResult, ProposedAction, Stakes, Verdict,
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    predicate: Callable[[ProposedAction], bool]  # True => violation
    explanation: str


class ConstraintLayer:
    '''Immutable constraint layer. Spec §8.1.

    The policy is loaded once and hashed. Any call verifies the hash
    before evaluating rules. Modifications require a fresh load.
    '''

    def __init__(self, rules: list[Rule]):
        self._rules = tuple(rules)
        self._policy_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        # Hash the rule IDs and explanations. Predicates are not
        # serializable, so we hash a canonical description.
        desc = json.dumps(
            [{'id': r.rule_id, 'expl': r.explanation} for r in self._rules],
            sort_keys=True,
        )
        return hashlib.sha256(desc.encode()).hexdigest()

    @property
    def policy_hash(self) -> str:
        return self._policy_hash

    def verify(self, expected_hash: str) -> None:
        if self._policy_hash != expected_hash:
            raise RuntimeError(
                f'Constraint policy hash mismatch: '
                f'{self._policy_hash} != {expected_hash}'
            )

    def evaluate(self, action: ProposedAction) -> ConstraintResult:
        trace = []
        for rule in self._rules:
            violated = rule.predicate(action)
            trace.append({
                'rule_id': rule.rule_id,
                'outcome': 'violation' if violated else 'pass',
                'explanation': rule.explanation,
            })
            if violated:
                return ConstraintResult(
                    verdict=Verdict.REJECT,
                    trace=trace,
                    policy_hash=self._policy_hash,
                )
        return ConstraintResult(
            verdict=Verdict.APPROVE,
            trace=trace,
            policy_hash=self._policy_hash,
        )


# ---- Example rules (spec §8.1) -------------------------------------------

def rule_no_blacklisted_routing(action: ProposedAction) -> bool:
    if action.action_type != 'route':
        return False
    blacklist = {'blacklisted_dept'}
    return action.parameters.get('department') in blacklist


def rule_escalate_on_low_confidence_high_stakes(action: ProposedAction) -> bool:
    # Violation if: high stakes, confidence < 0.3, action is not escalate.
    if action.stakes != Stakes.HIGH:
        return False
    conf = float(action.confidence.mean())
    if conf >= 0.3:
        return False
    return action.action_type != 'escalate'


def default_policy() -> list[Rule]:
    return [
        Rule('no_blacklisted_routing', rule_no_blacklisted_routing,
             'Never route to a blacklisted department.'),
        Rule('escalate_low_conf_high_stakes',
             rule_escalate_on_low_confidence_high_stakes,
             'Escalate when stakes are high and confidence < 0.3.'),
    ]
