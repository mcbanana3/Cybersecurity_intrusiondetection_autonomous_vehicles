"""
Evaluation helpers for the IDS.

Produces real metrics (no fabricated numbers): accuracy, precision,
recall, F1 (macro + weighted), a confusion matrix, and per-class
support. Used for both the binary and multi-class models.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Compute a dictionary of classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional names for readable confusion-matrix labels.

    Returns:
        Dict with accuracy, macro/weighted precision/recall/F1, the
        confusion matrix (list of lists), and class labels.
    """
    labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    if class_names is not None:
        metrics["class_names"] = [
            class_names[i] if i < len(class_names) else str(i) for i in labels
        ]

    return metrics


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Return F1 score per class name (0.0 for classes never predicted)."""
    scores = f1_score(
        y_true, y_pred, average=None, labels=list(range(len(class_names))),
        zero_division=0,
    )
    return {class_names[i]: float(scores[i]) for i in range(len(class_names))}