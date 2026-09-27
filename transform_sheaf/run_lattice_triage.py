"""Run the transformation-sheaf pipeline against the project's own real
Phase 1 triage corpus (data/triage/*.jsonl), not just run_triage.py's small
hand-built toy corpus -- a fairer, more relevant test since this is the
actual domain application the project targets, and it has real linguistic
diversity (20 document types, template-filled bodies, hard cases) that the
36-template toy corpus doesn't.

Compares: transformation-record features (this pipeline's own logistic
head), a bag-of-words baseline (same head), and the calibrated readout-head
accuracy already measured for this exact task in Phase 1
(PHASE1_RESULTS.md) as an external reference point for what a trained
representation achieves on the same 10-way department-routing task.
"""
from __future__ import annotations
import json

import numpy as np

from transform_sheaf.classify import LogisticClassifier
from transform_sheaf.fca import build_feature_matrix
from transform_sheaf.record import apply_operations_to_corpus
from transform_sheaf.sheaf import coherence


def load_split(path: str, limit: int) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            doc = json.loads(line)
            texts.append(doc['text'])
            labels.append(doc['department'])
    return texts, labels


def bag_of_words_features(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            if tok in vocab:
                X[i, vocab[tok]] += 1.0
    return X


def run(n_train: int = 3000, n_test: int = 1000):
    train_texts, train_labels = load_split('data/triage/train.jsonl', n_train)
    test_texts, test_labels = load_split('data/triage/test.jsonl', n_test)
    num_classes = 10

    print(f'Train: {len(train_texts)}  Test: {len(test_texts)}  '
         f'(10-way department routing, real Phase 1 triage corpus)')

    # --- Transformation record features ---
    train_records = apply_operations_to_corpus(train_texts)
    test_records = apply_operations_to_corpus(test_texts)
    Xtr, feature_names = build_feature_matrix(train_records)
    Xte, _ = build_feature_matrix(test_records)

    clf = LogisticClassifier(num_classes=num_classes, num_features=Xtr.shape[1],
                             lr=0.5, epochs=800, l2=1e-3)
    clf.fit(Xtr, np.array(train_labels))
    acc_record = (clf.predict(Xte) == np.array(test_labels)).mean()

    # --- Bag of words baseline (same classifier, same protocol) ---
    vocab: dict[str, int] = {}
    for t in train_texts:
        for tok in t.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    Xtr_bow = bag_of_words_features(train_texts, vocab)
    Xte_bow = bag_of_words_features(test_texts, vocab)
    clf_bow = LogisticClassifier(num_classes=num_classes,
                                 num_features=Xtr_bow.shape[1],
                                 lr=0.5, epochs=800, l2=1e-3)
    clf_bow.fit(Xtr_bow, np.array(train_labels))
    acc_bow = (clf_bow.predict(Xte_bow) == np.array(test_labels)).mean()

    # --- BOW + record combined: the more defensible framing of the
    # hypothesis (structural features ADD to lexical ones, not replace
    # them) ---
    Xtr_combined = np.concatenate([Xtr_bow, Xtr], axis=1)
    Xte_combined = np.concatenate([Xte_bow, Xte], axis=1)
    clf_combined = LogisticClassifier(num_classes=num_classes,
                                      num_features=Xtr_combined.shape[1],
                                      lr=0.5, epochs=800, l2=1e-3)
    clf_combined.fit(Xtr_combined, np.array(train_labels))
    acc_combined = (clf_combined.predict(Xte_combined)
                    == np.array(test_labels)).mean()

    coherent = sum(1 for t in train_texts if coherence(t).coherent)

    print(f'\nTransformation record features ({Xtr.shape[1]}): '
         f'acc = {acc_record:.3f}')
    print(f'Bag of words baseline ({Xtr_bow.shape[1]}):       '
         f'acc = {acc_bow:.3f}')
    print(f'BOW + record combined ({Xtr_combined.shape[1]}):        '
         f'acc = {acc_combined:.3f}')
    print(f'Trained readout head, same task (PHASE1_RESULTS.md,'
         f' post-calibration-fix): acc = 0.997')
    print(f'\nCoherent documents: {coherent} / {len(train_texts)}')


if __name__ == '__main__':
    run()
