"""
Phase 1 tests: verify the synthetic signal generator behaves sensibly.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import pandas as pd

from src.generator.signal_generator import SignalGenerator, SIGNAL_COLUMNS
from src.utils.config_loader import load_config


def _make_df() -> pd.DataFrame:
    gen = SignalGenerator(load_config())
    return gen.generate()


def test_columns_present() -> None:
    """Generated DataFrame must contain exactly the expected columns."""
    df = _make_df()
    assert list(df.columns) == SIGNAL_COLUMNS


def test_row_count_matches_config() -> None:
    """Number of samples must equal duration * sample_rate."""
    config = load_config()
    expected = int(
        config["simulation"]["duration_seconds"]
        * config["simulation"]["sample_rate_hz"]
    )
    df = _make_df()
    assert len(df) == expected


def test_speed_within_bounds() -> None:
    """Speed must never be negative or exceed the configured max."""
    config = load_config()
    df = _make_df()
    assert df["speed_kmh"].min() >= 0.0
    assert df["speed_kmh"].max() <= config["vehicle"]["speed_kmh"]["max"] + 1e-6


def test_soc_within_bounds() -> None:
    """State of charge must stay within [min, 100]."""
    config = load_config()
    df = _make_df()
    assert df["battery_soc"].min() >= config["battery"]["soc_percent"]["min"] - 1e-6
    assert df["battery_soc"].max() <= 100.0 + 1e-6


def test_charging_state_is_binary() -> None:
    """charging_state must only ever be 0 or 1."""
    df = _make_df()
    assert set(df["charging_state"].unique()).issubset({0.0, 1.0})


def test_charging_current_is_negative() -> None:
    """While charging, pack current must flow into the pack (<= 0)."""
    df = _make_df()
    charging = df[df["charging_state"] == 1]
    assert (charging["pack_current_a"] <= 0).all()


def test_reproducibility() -> None:
    """Same seed must produce identical data (deterministic)."""
    df1 = _make_df()
    df2 = _make_df()
    pd.testing.assert_frame_equal(df1, df2)