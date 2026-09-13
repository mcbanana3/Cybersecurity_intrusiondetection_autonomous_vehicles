"""
Model factory for the IDS.

Centralises the choice of estimators and their hyperparameters so the
training pipeline stays clean and the choices are documented:

    * Binary detector      -> RandomForestClassifier
      Strong, robust baseline for tabular window features; handles
      non-linear boundaries and gives feature importances for free.

    * Multi-class detector -> XGBoost (XGBClassifier)
      Gradient boosting handles class imbalance and subtle differences
      between attack types (e.g. tampering vs charging) well.

    * Anomaly detector     -> IsolationForest
      Unsupervised; trained on NORMAL windows only so it can flag
      previously-unseen ("zero-day") attacks as outliers.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier


def build_binary_model(random_state: int = 42) -> RandomForestClassifier:
    """Return the Random Forest used for binary attack detection."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # counter attack/normal imbalance
        n_jobs=-1,
        random_state=random_state,
    )


def build_multiclass_model(
    num_classes: int, random_state: int = 42
) -> XGBClassifier:
    """Return the XGBoost model used for attack-type classification.

    Args:
        num_classes: Number of target classes (normal + attack types).
        random_state: Seed for reproducibility.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )


def build_anomaly_model(random_state: int = 42) -> IsolationForest:
    """Return the Isolation Forest used for unknown-attack detection."""
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        n_jobs=-1,
        random_state=random_state,
    )