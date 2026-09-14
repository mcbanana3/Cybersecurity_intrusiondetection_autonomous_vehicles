"""
Phase 10 tests: full end-to-end chain via IntegratedIDS.

Requires the dataset + trained models (run run_phase4.py & run_phase5.py,
or just run_phase10.py once).

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.features.dataset import _NON_FEATURE
from src.integration.integrated_ids import IntegratedIDS, IntegratedVerdict
from src.pipeline import run_pipeline
from src.utils.config_loader import load_config


@pytest.fixture(scope="module")
def _require_models() -> None:
    if not os.path.exists(os.path.join("models", "binary_rf.joblib")):
        pytest.skip("models not trained - run run_phase5.py / run_phase10.py first")


@pytest.fixture(scope="module")
def integrated(_require_models) -> IntegratedIDS:
    return IntegratedIDS(load_config(), "models")


def test_normal_vector_not_attack(integrated) -> None:
    """A clearly-normal feature window should not be flagged."""
    result = run_pipeline()
    features = result.features
    cols = [c for c in features.columns if c not in _NON_FEATURE]
    normal = features[features["label_multi"] == "normal"]
    assert not normal.empty
    # Check the majority of normal windows are NOT flagged (low FPR).
    flagged = 0
    for _, row in normal.iterrows():
        v = integrated.analyze(row[cols].to_numpy(dtype=np.float32))
        if v.is_attack:
            flagged += 1
    assert flagged / len(normal) <= 0.30


def test_each_attack_detected_and_scored(integrated) -> None:
    """Every present attack type must be detected in >=1 window, with risk."""
    result = run_pipeline()
    features = result.features
    cols = [c for c in features.columns if c not in _NON_FEATURE]

    for atk in ["spoofing", "replay", "dos", "injection",
                "tampering", "charging_attack"]:
        sub = features[features["label_multi"] == atk]
        if sub.empty:
            continue
        detected_any = False
        for _, row in sub.iterrows():
            v = integrated.analyze(row[cols].to_numpy(dtype=np.float32))
            if v.is_attack:
                detected_any = True
                assert v.affected_asset is not None
                assert v.risk_score is not None and v.risk_score > 0
                assert v.recommended_action is not None
                break
        assert detected_any, f"{atk} was never detected!"


def test_verdict_demo_string(integrated) -> None:
    """The demo string should contain the required fields for an attack."""
    result = run_pipeline()
    features = result.features
    cols = [c for c in features.columns if c not in _NON_FEATURE]
    attacks = features[features["label_multi"] != "normal"]
    row = attacks.iloc[0]
    v = integrated.analyze(row[cols].to_numpy(dtype=np.float32))
    if v.is_attack:
        s = v.demo_string()
        for field in ["Attack detected", "Attack type", "Confidence",
                      "Affected asset", "Severity", "Risk score",
                      "Recommended action"]:
            assert field in s


def test_dos_maps_to_can_bus(integrated) -> None:
    """A detected DoS window must map to the CAN Bus asset."""
    result = run_pipeline()
    features = result.features
    cols = [c for c in features.columns if c not in _NON_FEATURE]
    dos = features[features["label_multi"] == "dos"]
    if dos.empty:
        pytest.skip("no dos windows")
    for _, row in dos.iterrows():
        v = integrated.analyze(row[cols].to_numpy(dtype=np.float32))
        if v.is_attack and v.attack_type == "dos":
            assert v.affected_asset == "CAN Bus"
            return
    pytest.skip("dos not detected as 'dos' type in any window")