"""
IDS predictor.

Loads the trained models and exposes one clean method the rest of the
system uses:

    predict(feature_vector) -> IDSResult

The result contains the binary decision, the predicted attack type,
a confidence value, and whether the unsupervised anomaly model flagged
the sample. Confidence is the model's own predicted probability -- it is
NOT hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import joblib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IDSResult:
    """Structured output of a single IDS prediction."""

    is_attack: bool
    attack_type: str
    confidence: float          # 0..1, model probability of the decision
    is_anomaly: bool           # unsupervised anomaly flag
    binary_attack_prob: float  # probability the binary model assigns to "attack"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IDSPredictor:
    """Wraps the trained scaler + models for inference."""

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = model_dir
        self.scaler = self._load("scaler.joblib")
        self.binary = self._load("binary_rf.joblib")
        self.multiclass = self._load("multiclass_xgb.joblib")
        self.anomaly = self._load("anomaly_iforest.joblib")
        meta = self._load("metadata.joblib")
        self.feature_names: List[str] = [str(x) for x in meta["feature_names"]]
        self.class_names: List[str] = [str(x) for x in meta["class_names"]]
        # Maps XGBoost's internal contiguous class index -> original index.
        self.xgb_new_to_old: Dict[int, int] = {
            int(k): int(v) for k, v in meta.get("xgb_new_to_old", {}).items()
        }
        logger.info("IDSPredictor loaded (%d features, %d classes)",
                    len(self.feature_names), len(self.class_names))

    def _load(self, name: str) -> Any:
        path = os.path.join(self.model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model artifact: {path}. Run Phase 5 training first."
            )
        return joblib.load(path)

    # ------------------------------------------------------------------
    def predict(self, feature_vector: np.ndarray) -> IDSResult:
        """Predict on a single feature vector."""
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs = self.scaler.transform(x)

        # Binary decision + probability.
        b_proba = self.binary.predict_proba(xs)[0]
        attack_prob = float(b_proba[1]) if len(b_proba) > 1 else float(b_proba[0])
        is_attack = attack_prob >= 0.5

        # Multi-class type + confidence (map XGB index back to real class).
        m_proba = self.multiclass.predict_proba(xs)[0]
        m_new_idx = int(np.argmax(m_proba))
        m_old_idx = self.xgb_new_to_old.get(m_new_idx, m_new_idx)
        attack_type = self.class_names[m_old_idx]
        m_conf = float(m_proba[m_new_idx])

        # Anomaly flag (unsupervised).
        is_anomaly = bool(self.anomaly.predict(xs)[0] == -1)

        # If binary says attack but multiclass says 'normal', pick the most
        # probable non-normal class for a coherent report.
        if is_attack and attack_type == "normal":
            order = np.argsort(m_proba)[::-1]
            for new_idx in order:
                old_idx = self.xgb_new_to_old.get(int(new_idx), int(new_idx))
                if self.class_names[old_idx] != "normal":
                    attack_type = self.class_names[old_idx]
                    m_conf = float(m_proba[int(new_idx)])
                    break

        confidence = m_conf if is_attack else float(b_proba[0])

        return IDSResult(
            is_attack=bool(is_attack),
            attack_type=attack_type if is_attack else "normal",
            confidence=round(confidence, 4),
            is_anomaly=is_anomaly,
            binary_attack_prob=round(attack_prob, 4),
        )

    def predict_batch(self, X: np.ndarray) -> List[IDSResult]:
        """Predict on a 2-D array of feature vectors."""
        return [self.predict(row) for row in np.asarray(X)]