"""
Phase 6 tests: asset registry, scenario impact, risk engine, IDS bridge.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tara.assets import AssetRegistry
from src.tara.risk_engine import RiskEngine
from src.tara.scenarios import ScenarioCatalog
from src.utils.config_loader import load_config


def test_asset_registry_loads() -> None:
    reg = AssetRegistry(load_config())
    assert len(reg.all()) >= 5
    assert reg.get("CAN Bus").name == "CAN Bus"


def test_unknown_asset_fallback() -> None:
    reg = AssetRegistry(load_config())
    a = reg.get("Nonexistent Asset")
    assert a.name == "Nonexistent Asset"
    assert a.criticality == "Medium"


def test_impact_aggregation_in_range() -> None:
    cat = ScenarioCatalog(load_config())
    for scenario in cat.all().values():
        agg = cat.aggregate_impact(scenario)
        assert 1.0 <= agg <= 5.0


def test_detection_raises_feasibility() -> None:
    engine = RiskEngine(load_config())
    base = engine.assess_attack_type("tampering", detected=False)
    det = engine.assess_attack_type("tampering", detected=True)
    assert det.feasibility >= base.feasibility
    assert det.risk_score >= base.risk_score


def test_risk_levels_valid() -> None:
    engine = RiskEngine(load_config())
    for ra in engine.static_tara_table(detected=True):
        assert ra.risk_level in {"Low", "Medium", "High", "Critical"}
        assert 1.0 <= ra.risk_score <= 25.0


def test_priority_sorted_descending() -> None:
    engine = RiskEngine(load_config())
    table = engine.static_tara_table(detected=True)
    scores = [ra.risk_score for ra in table]
    assert scores == sorted(scores, reverse=True)


def test_dos_maps_to_can_bus() -> None:
    engine = RiskEngine(load_config())
    ra = engine.assess_attack_type("dos", detected=True)
    assert ra.asset == "CAN Bus"


def test_ids_bridge_with_duck_typed_result() -> None:
    """assess() should work with any object exposing is_attack/attack_type."""
    @dataclass
    class FakeResult:
        is_attack: bool
        attack_type: str

    engine = RiskEngine(load_config())
    ra = engine.assess(FakeResult(is_attack=True, attack_type="spoofing"))
    assert ra is not None and ra.asset == "Powertrain ECU"

    none_ra = engine.assess(FakeResult(is_attack=False, attack_type="normal"))
    assert none_ra is None