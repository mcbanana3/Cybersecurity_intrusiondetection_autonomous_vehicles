"""
Phase 8 tests: charging session model + security monitor integration.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

from src.charging.monitor import ChargingSecurityMonitor
from src.charging.session import CHARGING_COLUMNS, ChargingSession
from src.utils.config_loader import load_config


def test_normal_session_columns_and_plausible() -> None:
    df = ChargingSession(load_config()).generate_normal()
    assert list(df.columns) == CHARGING_COLUMNS
    assert (df["label"] == "normal").all()
    # Normal grid frequency stays within a plausible band.
    assert df["grid_frequency_hz"].min() >= 49.0
    assert df["grid_frequency_hz"].max() <= 51.0


def test_normal_soc_increases() -> None:
    df = ChargingSession(load_config()).generate_normal()
    assert df["soc"].iloc[-1] > df["soc"].iloc[0]


def test_attacked_session_has_attack_samples() -> None:
    df = ChargingSession(load_config()).generate_attacked()
    assert (df["label"] == "attack").sum() > 0
    # During attack, grid frequency drops below the plausible floor.
    assert df["grid_frequency_hz"].min() < 49.0


def test_monitor_detects_charging_attack() -> None:
    cfg = load_config()
    attacked = ChargingSession(cfg).generate_attacked()
    verdicts = ChargingSecurityMonitor(cfg).evaluate(attacked)
    flagged = [v for v in verdicts if v.tampered]
    assert len(flagged) > 0
    v = flagged[0]
    assert v.detected_attack == "charging_attack"
    assert v.affected_asset == "Charging Controller"
    assert v.risk_score is not None and v.risk_score > 0


def test_monitor_does_not_flag_normal() -> None:
    cfg = load_config()
    normal = ChargingSession(cfg).generate_normal()
    verdicts = ChargingSecurityMonitor(cfg).evaluate(normal)
    flagged = [v for v in verdicts if v.tampered]
    assert len(flagged) == 0


def test_secure_control_channel() -> None:
    cfg = load_config()
    monitor = ChargingSecurityMonitor(cfg)
    assert monitor.secure_control_message(soc=50.0, current=80.0) is True