from transform_sheaf.sheaf import coherence, split_sentences
from transform_sheaf.record import apply_operations
from transform_sheaf.fca import build_feature_matrix


def test_split_sentences():
    s = split_sentences("i need a refund . my card was charged .")
    assert len(s) == 2


def test_coherence_reports_structure():
    rep = coherence("i need a refund . the app crashes .")
    assert isinstance(rep.coherent, bool)
    assert isinstance(rep.obstruction_ops, tuple)


def test_feature_matrix_shape():
    texts = ["i need a refund", "the app crashes"]
    records = [apply_operations(t) for t in texts]
    X, names = build_feature_matrix(records)
    assert X.shape[0] == 2
    assert X.shape[1] == len(names)
