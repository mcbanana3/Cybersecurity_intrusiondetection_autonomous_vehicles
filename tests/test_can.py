"""
Phase 2 tests: CAN codec round-trip and CAN trace structure.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

from src.generator.can_bus import CANBus, CAN_COLUMNS
from src.generator.can_codec import (
    decode_message,
    encode_message,
    load_message_specs,
)
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config


def test_codec_roundtrip_speed() -> None:
    """Encoding then decoding a value returns (approximately) the same value."""
    config = load_config()
    specs = {s.name: s for s in load_message_specs(config)}
    pwt = specs["PWT_Speed"]

    original = {"speed_kmh": 87.42, "acceleration_ms2": 1.5}
    payload = encode_message(pwt, original)
    decoded = decode_message(pwt, payload)

    assert abs(decoded["speed_kmh"] - 87.42) < 0.01
    assert abs(decoded["acceleration_ms2"] - 1.5) < 0.01


def test_payload_is_eight_bytes() -> None:
    """Every encoded payload must be exactly 8 bytes in [0, 255]."""
    config = load_config()
    specs = load_message_specs(config)
    for spec in specs:
        payload = encode_message(spec, {s.name: 0.0 for s in spec.signals})
        assert len(payload) == 8
        assert all(0 <= b <= 255 for b in payload)


def test_out_of_range_saturates() -> None:
    """Out-of-range values saturate instead of raising (needed for attacks)."""
    config = load_config()
    specs = {s.name: s for s in load_message_specs(config)}
    pwt = specs["PWT_Speed"]

    payload = encode_message(pwt, {"speed_kmh": 1e9, "acceleration_ms2": 0.0})
    assert all(0 <= b <= 255 for b in payload)


def test_trace_columns_and_labels() -> None:
    """CAN trace must have canonical columns and be all-normal baseline."""
    config = load_config()
    signals = SignalGenerator(config).generate()
    can_df = CANBus(config).generate_trace(signals)

    assert list(can_df.columns) == CAN_COLUMNS
    assert (can_df["label"] == "normal").all()
    assert (can_df["attack_type"] == "none").all()
    assert (can_df["dlc"] == 8).all()


def test_all_message_types_present() -> None:
    """Every configured message type must appear in the trace."""
    config = load_config()
    signals = SignalGenerator(config).generate()
    can_df = CANBus(config).generate_trace(signals)

    expected = {m["name"] for m in config["can"]["messages"]}
    assert expected.issubset(set(can_df["message_name"].unique()))


def test_timestamps_sorted() -> None:
    """CAN frames must be sorted by timestamp."""
    config = load_config()
    signals = SignalGenerator(config).generate()
    can_df = CANBus(config).generate_trace(signals)

    ts = can_df["timestamp"].to_numpy()
    assert (ts[:-1] <= ts[1:]).all()