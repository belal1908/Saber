from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from holo.config import MODEL_PATH


@dataclass
class ZoneClassifier:
    classes: list[str]
    coef: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, y: list[str]) -> "ZoneClassifier":
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X, y)
        return cls(classes=list(clf.classes_), coef=clf.coef_, intercept=clf.intercept_)

    def predict(self, features: np.ndarray) -> str:
        logits = self.coef @ features + self.intercept
        return self.classes[int(np.argmax(logits))]

    def save(self, path=MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "classes": self.classes,
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, path=MODEL_PATH) -> "ZoneClassifier":
        data = json.loads(path.read_text())
        return cls(
            classes=data["classes"],
            coef=np.array(data["coef"]),
            intercept=np.array(data["intercept"]),
        )
