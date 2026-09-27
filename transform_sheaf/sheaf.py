"""Local-to-global consistency.

A simple sheaf construction: the site is the set of sentences in a
document. Local data at each sentence is the operation-outcome
signature for that sentence. Restriction to a smaller window is
subset. The sheaf condition: the document-level signature must be
consistent with the union of sentence-level signatures.

This is a COARSE approximation. A proper implementation would use a
Grothendieck topology over a finer site. The coarse version is enough
to demonstrate the signal: coherence or its obstruction.
"""
from __future__ import annotations
from dataclasses import dataclass

from transform_sheaf.operations import Outcome
from transform_sheaf.record import DocumentRecord, apply_operations


@dataclass
class CoherenceReport:
    coherent: bool
    document_signature: tuple
    sentence_union: tuple
    obstruction_ops: tuple    # operations where local and global disagree


def split_sentences(text: str) -> list[str]:
    parts = []
    current = []
    for tok in text.replace("?", " ? ").replace(".", " . ").split():
        if tok in {"?", "."}:
            if current:
                parts.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current:
        parts.append(" ".join(current))
    return parts or [text]


def coherence(text: str) -> CoherenceReport:
    """Check whether the document's operation signature glues from
    its sentence-level signatures."""
    doc = apply_operations(text)
    sentences = split_sentences(text)
    sent_records = [apply_operations(s) for s in sentences]

    # Union of operations that apply at the sentence level
    union = set()
    for rec in sent_records:
        for op, outcome in rec.outcomes.items():
            if outcome == Outcome.APPLIES:
                union.add(op)

    doc_applies = set(doc.signature())
    obstruction = doc_applies.symmetric_difference(union)

    return CoherenceReport(
        coherent=len(obstruction) == 0,
        document_signature=tuple(sorted(doc_applies)),
        sentence_union=tuple(sorted(union)),
        obstruction_ops=tuple(sorted(obstruction)),
    )
