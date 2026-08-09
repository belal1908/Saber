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
    mode: str = "passive"  # "passive" (tap detection) or "probe" (active chirp)
    device: int | str | None = None  # input device this model was trained against

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: list[str],
        mode: str = "passive",
        device: int | str | None = None,
    ) -> ZoneClassifier:
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X, y)
        return cls(
            classes=list(clf.classes_),
            coef=clf.coef_,
            intercept=clf.intercept_,
            mode=mode,
            device=device,
        )

    def predict(self, features: np.ndarray) -> str:
        logits = self.coef @ features + self.intercept
        return self.classes[int(np.argmax(logits))]

    def scores(self, features: np.ndarray) -> dict[str, float]:
        """Per-zone confidence (softmax over logits), for diagnosing whether a
        wrong prediction was a close call or the classes aren't separating at all."""
        logits = self.coef @ features + self.intercept
        shifted = logits - np.max(logits)  # numerically stable softmax
        exp = np.exp(shifted)
        probs = exp / exp.sum()
        return dict(zip(self.classes, (float(p) for p in probs)))

    def save(self, path=MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "classes": self.classes,
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept.tolist(),
                    "mode": self.mode,
                    "device": self.device,
                }
            )
        )

    @classmethod
    def load(cls, path=MODEL_PATH) -> ZoneClassifier:
        data = json.loads(path.read_text())
        return cls(
            classes=data["classes"],
            coef=np.array(data["coef"]),
            intercept=np.array(data["intercept"]),
            mode=data.get("mode", "passive"),  # models saved before mode tracking default to passive
            device=data.get("device"),  # models saved before device tracking default to None (OS default)
        )
