from lattice.synthetic_triage import (
    BLACKLISTED_DEPARTMENT, DOC_TYPES, Department, HARD_CASES, Urgency,
    generate_corpus, generate_document,
)
import random


def test_20_doc_types_10_departments():
    assert len(DOC_TYPES) == 20
    assert {d.department for d in DOC_TYPES} == set(Department)
    assert len(set(Department)) == 10


def test_generation_is_deterministic():
    a = generate_corpus(200, seed=42)
    b = generate_corpus(200, seed=42)
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.department for d in a] == [d.department for d in b]


def test_different_seeds_differ():
    a = generate_corpus(50, seed=1)
    b = generate_corpus(50, seed=2)
    assert [d.text for d in a] != [d.text for d in b]


def test_doc_ids_unique():
    docs = generate_corpus(500, seed=0)
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


def test_labels_in_range():
    docs = generate_corpus(1000, seed=0, hard_case_rate=0.3)
    for d in docs:
        assert 0 <= d.department <= 9
        assert d.urgency in (Urgency.LOW, Urgency.MEDIUM, Urgency.HIGH)
        assert isinstance(d.escalate, bool)
        assert isinstance(d.safety_flag, bool)
        assert 0 <= d.label_confidence <= 4
        assert d.hard_case is None or d.hard_case in HARD_CASES


def test_hard_case_rate_matches_target():
    docs = generate_corpus(5000, seed=0, hard_case_rate=0.2)
    rate = sum(d.hard_case is not None for d in docs) / len(docs)
    assert abs(rate - 0.2) < 0.03


def test_injected_instruction_flags_safety():
    rng = random.Random(0)
    found = 0
    for i in range(2000):
        doc = generate_document(rng, f'd{i}', hard_case_rate=1.0)
        if doc.hard_case == 'injected_instruction':
            found += 1
            assert doc.safety_flag is True
            assert doc.injected_department is not None
    assert found > 0


def test_rule_precedence_stacks_triggers_and_forces_escalation():
    rng = random.Random(0)
    found = 0
    for i in range(2000):
        doc = generate_document(rng, f'd{i}', hard_case_rate=1.0)
        if doc.hard_case == 'rule_precedence':
            found += 1
            assert doc.safety_flag is True
            assert doc.urgency == Urgency.HIGH
            assert doc.escalate is True
            assert doc.injected_department == int(BLACKLISTED_DEPARTMENT)
    assert found > 0


def test_department_distribution_roughly_balanced():
    docs = generate_corpus(20000, seed=0)
    counts = [0] * 10
    for d in docs:
        counts[d.department] += 1
    expected = len(docs) / 10
    for c in counts:
        assert abs(c - expected) / expected < 0.25
