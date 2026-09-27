"""Decision head: logistic regression over the record features.

This is the only LEARNED component in the system. Everything upstream
— the operation library, the sheaf check, the FCA features — is fixed.
The decision head is trained on labels, but the features it sees are
not representations; they are a record of what the fixed operations
did when applied.
"""
from __future__ import annotations
import numpy as np


class LogisticClassifier:
    """Simple multinomial logistic regression via numpy.

    Uses full-batch gradient descent. Good enough for the small
    corpora in the toy tests.
    """

    def __init__(self, num_classes: int, num_features: int, lr: float = 0.5,
                 epochs: int = 500, l2: float = 1e-3):
        self.W = np.zeros((num_features, num_classes), dtype=np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        n, d = X.shape
        num_classes = self.W.shape[1]
        Y = np.zeros((n, num_classes), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)
            grad_logits = (probs - Y) / n
            grad_W = X.T @ grad_logits + self.l2 * self.W
            grad_b = grad_logits.sum(axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.W + self.b).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)
