"""
Phase 11 tests: plots, SHAP, model comparison.

Requires data/dataset.npz + trained models.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pytest

from src.evaluation.experiments import run_model_comparison
from src.evaluation.explain import compute_shap_importance
from src.evaluation.plots import plot_confusion_matrix, plot_roc
from src.features.dataset import load_dataset


@pytest.fixture(scope="module")
def _require_artifacts() -> None:
    if not (os.path.exists("data/dataset.npz")
            and os.path.exists("models/binary_rf.joblib")):
        pytest.skip("dataset/models missing - run run_phase5.py first")


def test_confusion_matrix_plot(_require_artifacts) -> None:
    path = plot_confusion_matrix([[5, 1], [0, 4]], ["normal", "attack"])
    assert os.path.exists(path)


def test_roc_plot(_require_artifacts) -> None:
    data = load_dataset("data/dataset.npz")
    scaler = joblib.load("models/scaler.joblib")
    binary = joblib.load("models/binary_rf.joblib")
    proba = binary.predict_proba(scaler.transform(data["X_test"]))[:, 1]
    path, auc_val = plot_roc(data["yb_test"], proba)
    assert os.path.exists(path)
    assert 0.0 <= auc_val <= 1.0


def test_shap_importance(_require_artifacts) -> None:
    data = load_dataset("data/dataset.npz")
    scaler = joblib.load("models/scaler.joblib")
    binary = joblib.load("models/binary_rf.joblib")
    path, imp = compute_shap_importance(
        binary, scaler, data["X_test"], data["feature_names"], max_samples=60
    )
    assert os.path.exists(path)
    assert not imp.empty
    assert set(["feature", "mean_abs_shap"]).issubset(imp.columns)


def test_model_comparison(_require_artifacts) -> None:
    data = load_dataset("data/dataset.npz")
    comp = run_model_comparison(data, seed=42)
    assert len(comp) >= 4
    assert set(["model", "task", "accuracy", "f1_macro"]).issubset(comp.columns)
    assert comp["accuracy"].between(0, 1).all()