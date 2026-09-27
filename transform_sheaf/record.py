"""The transformation record.

For each document, apply every operation. Record the outcome. The
record is not a representation of the document. It is a log of what
happened when the fixed operation library was applied.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from transform_sheaf.operations import (
    OPERATIONS, OPERATION_NAMES, OpResult, Outcome, tokenize,
)


@dataclass
class DocumentRecord:
    text: str
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def feature_vector(self) -> list[float]:
        """One-hot over (operation, outcome) pairs."""
        vec = []
        for op_name in OPERATION_NAMES:
            for outcome in Outcome:
                vec.append(1.0 if self.outcomes.get(op_name) == outcome else 0.0)
        return vec

    def signature(self) -> tuple:
        """A compact signature: which operations apply."""
        return tuple(
            name for name in OPERATION_NAMES
            if self.outcomes.get(name) == Outcome.APPLIES
        )


def apply_operations(text: str) -> DocumentRecord:
    tokens = tokenize(text)
    rec = DocumentRecord(text=text)
    for op in OPERATIONS:
        name = op.__name__.replace("op_", "")
        result: OpResult = op(tokens)
        rec.outcomes[name] = result.outcome
        rec.details[name] = result.detail
    return rec


def apply_operations_to_corpus(texts: list[str]) -> list[DocumentRecord]:
    return [apply_operations(t) for t in texts]
