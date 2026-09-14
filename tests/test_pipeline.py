"""
Phase 9 tests: the pipeline orchestrator that powers the dashboard.

Requires trained models (run run_phase5.py first).

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import os

import pytest

from src.pipeline import PipelineResult, run_pipeline


@pytest.fixture(scope="module")
def _require_models() -> None:
    if not os.path.exists(os.path.join("models", "binary_rf.joblib")):
        pytest.skip("models not trained - run run_phase5.py first")


def test_pipeline_runs(_require_models) -> None:
    result = run_pipeline()
    assert isinstance(result, PipelineResult)
    assert len(result.verdicts) > 0
    assert not result.signals.empty
    assert not result.can_attacked.empty


def test_pipeline_detects_some_attack(_require_models) -> None:
    result = run_pipeline()
    assert any(v.is_attack for v in result.verdicts)


def test_pipeline_disable_all_attacks(_require_models) -> None:
    overrides = {atk: {"enabled": False} for atk in
                 ["spoofing", "replay", "dos", "injection",
                  "tampering", "charging_attack"]}
    result = run_pipeline(overrides)
    # With every attack disabled, the CAN trace should be all-normal.
    assert (result.can_attacked["label"] == "normal").all()


def test_pipeline_charging_included(_require_models) -> None:
    result = run_pipeline()
    assert result.charging_total > 0
    assert not result.charging_attacked.empty