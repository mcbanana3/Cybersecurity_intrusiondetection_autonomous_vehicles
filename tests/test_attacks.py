"""
Phase 3 tests: verify each simulated attack changes the trace correctly.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import pandas as pd

from src.attacks.engine import AttackEngine
from src.generator.can_bus import CANBus, CAN_COLUMNS
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config


def _baseline() -> pd.DataFrame:
    config = load_config()
    signals = SignalGenerator(config).generate()
    return CANBus(config).generate_trace(signals)


def test_engine_produces_attack_frames() -> None:
    """Running the engine must produce at least some attack-labelled frames."""
    config = load_config()
    attacked, events = AttackEngine(config).run(_baseline())
    assert (attacked["label"] == "attack").sum() > 0
    assert len(events) >= 5


def test_columns_preserved() -> None:
    """Attacked trace must keep the canonical CAN columns."""
    config = load_config()
    attacked, _ = AttackEngine(config).run(_baseline())
    assert list(attacked.columns) == CAN_COLUMNS


def test_all_attack_types_present() -> None:
    """Each enabled attack type should appear in attack_type values."""
    config = load_config()
    attacked, _ = AttackEngine(config).run(_baseline())
    types = set(attacked["attack_type"].unique())
    for expected in ["spoofing", "replay", "dos", "injection",
                     "tampering", "charging_attack"]:
        assert expected in types, f"Missing attack type: {expected}"


def test_dos_increases_frame_count() -> None:
    """DoS flooding must increase total frame count vs baseline."""
    config = load_config()
    base = _baseline()
    attacked, _ = AttackEngine(config).run(base)
    assert len(attacked) > len(base)


def test_injection_adds_unknown_id() -> None:
    """Injection must introduce an ID not present in the baseline."""
    config = load_config()
    base = _baseline()
    attacked, _ = AttackEngine(config).run(base)
    base_ids = set(base["arbitration_id"].unique())
    attacked_ids = set(attacked["arbitration_id"].unique())
    assert len(attacked_ids - base_ids) > 0


def test_tampering_keeps_frame_count_for_target() -> None:
    """Tampering corrupts existing frames; it must not add BMS_State frames."""
    config = load_config()
    base = _baseline()
    attacked, _ = AttackEngine(config).run(base)
    base_bms = (base["message_name"] == "BMS_State").sum()
    att_bms = (attacked["message_name"] == "BMS_State").sum()
    assert att_bms == base_bms


def test_labels_are_valid() -> None:
    """Only 'normal' and 'attack' labels may appear."""
    config = load_config()
    attacked, _ = AttackEngine(config).run(_baseline())
    assert set(attacked["label"].unique()).issubset({"normal", "attack"})