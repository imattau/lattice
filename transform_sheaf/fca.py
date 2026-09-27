"""Formal concept features from the transformation record.

The formal context is (documents x operation-outcome) incidence. The
features below are the raw incidence plus a few derived signals:
  - apply count (how many operations applied)
  - block count
  - contradiction count
  - coherence flag from the sheaf check
"""
from __future__ import annotations
import numpy as np

from transform_sheaf.operations import Outcome
from transform_sheaf.record import DocumentRecord, OPERATION_NAMES
from transform_sheaf.sheaf import coherence


def build_feature_matrix(
    records: list[DocumentRecord],
    include_sheaf: bool = True,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    feature_names = []
    for op_name in OPERATION_NAMES:
        for outcome in Outcome:
            feature_names.append(f"{op_name}:{outcome.value}")
    if include_sheaf:
        feature_names.append("coherent")
        feature_names.append("obstruction_count")
        feature_names.append("apply_count")
        feature_names.append("block_count")
        feature_names.append("contradict_count")

    for rec in records:
        vec = []
        for op_name in OPERATION_NAMES:
            for outcome in Outcome:
                vec.append(1.0 if rec.outcomes.get(op_name) == outcome else 0.0)

        if include_sheaf:
            rep = coherence(rec.text)
            vec.append(1.0 if rep.coherent else 0.0)
            vec.append(float(len(rep.obstruction_ops)))
            applies = sum(1 for o in rec.outcomes.values() if o == Outcome.APPLIES)
            blocks = sum(1 for o in rec.outcomes.values() if o == Outcome.BLOCKS)
            contradicts = sum(1 for o in rec.outcomes.values()
                              if o == Outcome.CONTRADICTS)
            vec.append(float(applies))
            vec.append(float(blocks))
            vec.append(float(contradicts))

        rows.append(vec)

    return np.array(rows, dtype=np.float32), feature_names
