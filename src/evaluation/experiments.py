"""
Model comparison experiment.

Trains several classic baselines on the SAME scaled split and reports
accuracy / macro-F1 for both the binary and multi-class tasks. Produces
the comparison table reviewers expect. All numbers are real (computed on
the held-out test split).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _binary_models(seed: int) -> Dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced",
                                               random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200,
                                               class_weight="balanced",
                                               n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5,
                                 learning_rate=0.1, eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1,
                                 random_state=seed),
    }


def run_model_comparison(data: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Compare baselines on the binary task (and RF/XGB on multi-class).

    Args:
        data: Loaded dataset dict (from load_dataset).
        seed: Random seed.

    Returns:
        A DataFrame with columns:
        ['model', 'task', 'accuracy', 'f1_macro'].
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(data["X_train"])
    Xte = scaler.transform(data["X_test"])

    yb_tr, yb_te = data["yb_train"], data["yb_test"]
    ym_tr, ym_te = data["ym_train"], data["ym_test"]

    rows: List[Dict[str, Any]] = []

    # ---- Binary task: all baselines ----
    for name, model in _binary_models(seed).items():
        model.fit(Xtr, yb_tr)
        pred = model.predict(Xte)
        rows.append({
            "model": name,
            "task": "binary",
            "accuracy": round(float(accuracy_score(yb_te, pred)), 4),
            "f1_macro": round(float(f1_score(yb_te, pred, average="macro",
                                             zero_division=0)), 4),
        })
        logger.info("[binary] %-18s acc=%.3f f1=%.3f",
                    name, rows[-1]["accuracy"], rows[-1]["f1_macro"])

    # ---- Multi-class task: RF + XGBoost (with contiguous label remap) ----
    present = sorted(set(ym_tr.tolist()))
    old_to_new = {o: n for n, o in enumerate(present)}
    ym_tr_r = np.array([old_to_new[int(v)] for v in ym_tr], dtype=np.int64)

    def _to_new(arr):
        return np.array([old_to_new.get(int(v), -1) for v in arr], dtype=np.int64)

    ym_te_r = _to_new(ym_te)

    multi_models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               class_weight="balanced",
                                               random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=6,
                                 learning_rate=0.1,
                                 objective="multi:softprob",
                                 num_class=len(present),
                                 eval_metric="mlogloss", tree_method="hist",
                                 n_jobs=-1, random_state=seed),
    }
    for name, model in multi_models.items():
        model.fit(Xtr, ym_tr_r)
        pred = model.predict(Xte)
        rows.append({
            "model": name,
            "task": "multiclass",
            "accuracy": round(float(accuracy_score(ym_te_r, pred)), 4),
            "f1_macro": round(float(f1_score(ym_te_r, pred, average="macro",
                                             zero_division=0)), 4),
        })
        logger.info("[multi ] %-18s acc=%.3f f1=%.3f",
                    name, rows[-1]["accuracy"], rows[-1]["f1_macro"])

    return pd.DataFrame(rows)