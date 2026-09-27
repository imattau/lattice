"""Smallest test of the hypothesis.

Train a decision head on the transformation record. Compare to a
bag-of-words baseline and to a Core-based head (if available).

Uses a synthetic triage corpus with structurally distinct categories.
"""
from __future__ import annotations
import random
from collections import Counter

import numpy as np

from transform_sheaf.record import apply_operations_to_corpus
from transform_sheaf.fca import build_feature_matrix
from transform_sheaf.classify import LogisticClassifier


CATEGORIES = ["billing", "technical", "account", "general"]

# Templates per category. Structure differs; vocabulary overlaps.
TEMPLATES = {
    "billing": [
        "i was charged twice for my subscription",
        "my credit card was charged without authorization",
        "i need a refund for last month",
        "the invoice was sent to the wrong address",
        "i did not receive a receipt for my payment",
        "my payment was declined but the amount was taken",
    ],
    "technical": [
        "the app crashes when i try to log in",
        "the server is down and i cannot access my account",
        "the page does not load",
        "i am unable to upload files",
        "the connection keeps dropping",
        "the error message says the request failed",
    ],
    "account": [
        "i want to change my password",
        "please update my email address",
        "i would like to delete my account",
        "can you merge my two accounts",
        "i need to verify my identity",
        "please close my profile",
    ],
    "general": [
        "what are your business hours",
        "where can i find the user guide",
        "how do i contact support",
        "do you offer a mobile app",
        "is there a free trial",
        "can you explain your pricing",
    ],
}

NOISE_WORDS = ["please", "hello", "thanks", "urgent", "asap", "hi"]


def make_corpus(n_per_class: int = 60, seed: int = 0) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for idx, cat in enumerate(CATEGORIES):
        for _ in range(n_per_class):
            base = rng.choice(TEMPLATES[cat])
            noise = " ".join(rng.sample(NOISE_WORDS, rng.randint(0, 2)))
            text = (noise + " " + base).strip()
            texts.append(text)
            labels.append(idx)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train_test_split(texts, labels, test_frac=0.3, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in test_idx], [labels[i] for i in test_idx],
    )


def bag_of_words_features(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            if tok in vocab:
                X[i, vocab[tok]] += 1.0
    return X


def run():
    texts, labels = make_corpus(n_per_class=80, seed=1)
    tr_texts, tr_labels, te_texts, te_labels = train_test_split(
        texts, labels, test_frac=0.3, seed=1,
    )
    num_classes = len(CATEGORIES)

    # --- Transformation record features ---
    tr_records = apply_operations_to_corpus(tr_texts)
    te_records = apply_operations_to_corpus(te_texts)
    Xtr, feature_names = build_feature_matrix(tr_records)
    Xte, _ = build_feature_matrix(te_records)

    clf = LogisticClassifier(
        num_classes=num_classes, num_features=Xtr.shape[1],
        lr=0.5, epochs=800, l2=1e-3,
    ).fit(Xtr, np.array(tr_labels))
    preds = clf.predict(Xte)
    acc_record = (preds == np.array(te_labels)).mean()

    # --- Bag of words baseline ---
    vocab = {}
    for t in tr_texts:
        for tok in t.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    Xtr_bow = bag_of_words_features(tr_texts, vocab)
    Xte_bow = bag_of_words_features(te_texts, vocab)
    clf_bow = LogisticClassifier(
        num_classes=num_classes, num_features=Xtr_bow.shape[1],
        lr=0.5, epochs=800, l2=1e-3,
    ).fit(Xtr_bow, np.array(tr_labels))
    preds_bow = clf_bow.predict(Xte_bow)
    acc_bow = (preds_bow == np.array(te_labels)).mean()

    # --- Distribution of feature activations ---
    apply_counts = Counter()
    for rec in tr_records:
        for op, outcome in rec.outcomes.items():
            if outcome.value == "applies":
                apply_counts[op] += 1

    print(f"Train size: {len(tr_texts)}, Test size: {len(te_texts)}")
    print(f"Classes: {CATEGORIES}")
    print()
    print(f"Transformation record features ({Xtr.shape[1]}): "
          f"acc = {acc_record:.3f}")
    print(f"Bag of words baseline ({Xtr_bow.shape[1]}):       "
          f"acc = {acc_bow:.3f}")
    print()
    print("Feature activations (training set, 'applies' count):")
    for op, count in apply_counts.most_common():
        print(f"  {op:12s} {count:4d} / {len(tr_records)}")

    # --- Sheaf coherence distribution ---
    from transform_sheaf.sheaf import coherence
    coherent_count = sum(1 for t in tr_texts if coherence(t).coherent)
    print()
    print(f"Coherent documents: {coherent_count} / {len(tr_texts)}")

    return {
        "record_acc": acc_record,
        "bow_acc": acc_bow,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    run()
