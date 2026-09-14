"""
SHAP explainability for the binary IDS.

Computes global feature importance using SHAP on the trained Random
Forest binary detector, so we can explain WHICH features drive attack
detection (e.g. frame_rate_hz, unknown_id_count). Produces a bar plot
and a ranked CSV. Uses a small background/sample size to stay fast.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

FIG_DIR = os.path.join("results", "figures")


def compute_shap_importance(
    model,
    scaler,
    X_test: np.ndarray,
    feature_names: List[str],
    max_samples: int = 150,
) -> Tuple[str, pd.DataFrame]:
    """Compute SHAP global importance for the binary RF model.

    Args:
        model: Trained RandomForestClassifier (binary).
        scaler: Fitted StandardScaler used at training time.
        X_test: Raw (unscaled) test feature matrix.
        feature_names: Ordered feature names.
        max_samples: Cap on samples used (keeps SHAP fast).

    Returns:
        (figure_path, importance_dataframe) where the dataframe has
        columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    import shap

    os.makedirs(FIG_DIR, exist_ok=True)

    # Scale + subsample for speed.
    Xs = scaler.transform(X_test)
    if len(Xs) > max_samples:
        idx = np.random.default_rng(42).choice(len(Xs), max_samples, replace=False)
        Xs = Xs[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    # For binary RF, shap_values may be a list [class0, class1] or a 3-D
    # array. Normalise to the positive-class ('attack') contribution.
    sv = _positive_class_shap(shap_values)

    mean_abs = np.abs(sv).mean(axis=0)
    # Guard: ensure 1-D length matches feature_names.
    mean_abs = np.asarray(mean_abs).ravel()[: len(feature_names)]

    imp = pd.DataFrame(
        {"feature": feature_names[: len(mean_abs)], "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Plot top 15.
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#8e44ad")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("SHAP Global Feature Importance (binary IDS)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "shap_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s (top feature: %s)", path, imp.iloc[0]["feature"])
    return path, imp


def _positive_class_shap(shap_values) -> np.ndarray:
    """Extract the positive-class SHAP matrix across SHAP versions."""
    # Case 1: list of arrays per class -> take class 1.
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1
                          else shap_values[0])
    arr = np.asarray(shap_values)
    # Case 2: 3-D array (samples, features, classes) -> take last class.
    if arr.ndim == 3:
        return arr[:, :, -1]
    # Case 3: already 2-D (samples, features).
    return arr