"""
Phase 5 tests: train the IDS and verify models + predictor behave.

These tests train on the real Phase 4 dataset, so they require
data/dataset.npz to exist (run run_phase4.py first). Training is fast
(small dataset).

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.features.dataset import load_dataset
from src.ids.predict import IDSPredictor, IDSResult
from src.ids.train import IDSTrainer
from src.utils.config_loader import load_config

DATASET = "data/dataset.npz"


@pytest.fixture(scope="module")
def trained() -> None:
    """Ensure the dataset exists and models are trained once for the module."""
    if not os.path.exists(DATASET):
        pytest.skip("data/dataset.npz not found - run run_phase4.py first")
    IDSTrainer(load_config()).train(DATASET)


def test_artifacts_created(trained) -> None:
    for name in ["scaler.joblib", "binary_rf.joblib",
                 "multiclass_xgb.joblib", "anomaly_iforest.joblib",
                 "metadata.joblib"]:
        assert os.path.exists(os.path.join("models", name))


def test_binary_accuracy_reasonable(trained) -> None:
    metrics = IDSTrainer(load_config()).train(DATASET).metrics
    assert metrics["binary"]["test"]["accuracy"] >= 0.7


def test_multiclass_accuracy_reasonable(trained) -> None:
    metrics = IDSTrainer(load_config()).train(DATASET).metrics
    assert metrics["multiclass"]["test"]["accuracy"] >= 0.6


def test_predictor_returns_result(trained) -> None:
    data = load_dataset(DATASET)
    predictor = IDSPredictor("models")
    res = predictor.predict(data["X_test"][0])
    assert isinstance(res, IDSResult)
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.is_attack, bool)


def test_predictor_detects_dos(trained) -> None:
    """A DoS test window should be flagged as an attack."""
    data = load_dataset(DATASET)
    class_names = data["class_names"]
    if "dos" not in class_names:
        pytest.skip("no dos class present")
    dos_idx = class_names.index("dos")
    predictor = IDSPredictor("models")
    hits = 0
    total = 0
    for i, y in enumerate(data["ym_test"]):
        if y == dos_idx:
            total += 1
            if predictor.predict(data["X_test"][i]).is_attack:
                hits += 1
    if total == 0:
        pytest.skip("no dos windows in test split")
    assert hits > 0