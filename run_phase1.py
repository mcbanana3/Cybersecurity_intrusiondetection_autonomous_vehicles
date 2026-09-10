"""
Phase 1 runner.

Generates one synthetic AV/EV drive + charge session, saves it to
``data/normal_signals.csv`` and prints a summary so you can verify the
signals look physically sensible.

Run from the project root:
    python run_phase1.py
"""

from __future__ import annotations

from src.generator.signal_generator import SignalGenerator, SIGNAL_COLUMNS
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase1")

    logger.info("=== Phase 1: Synthetic AV/EV Data Generation ===")

    generator = SignalGenerator(config)
    df = generator.generate()
    path = generator.save(df, "normal_signals.csv")

    # --- Human-readable summary ---
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE - Synthetic AV/EV session generated")
    print("=" * 70)
    print(f"Saved to        : {path}")
    print(f"Rows (samples)  : {len(df)}")
    print(f"Columns         : {len(df.columns)}")
    print(f"Duration (s)    : {df['timestamp'].iloc[-1]:.1f}")
    print(f"Driving samples : {(df['charging_state'] == 0).sum()}")
    print(f"Charging samples: {(df['charging_state'] == 1).sum()}")

    print("\n--- Column list ---")
    print(", ".join(SIGNAL_COLUMNS))

    print("\n--- First 5 rows (driving) ---")
    print(df.head().to_string(index=False))

    print("\n--- Last 5 rows (charging) ---")
    print(df.tail().to_string(index=False))

    print("\n--- Signal ranges (min / mean / max) ---")
    for col in SIGNAL_COLUMNS:
        if col == "timestamp":
            continue
        print(
            f"{col:18s}: "
            f"{df[col].min():9.3f} / {df[col].mean():9.3f} / {df[col].max():9.3f}"
        )

    # --- Quick sanity assertions (physical plausibility) ---
    assert df["speed_kmh"].min() >= 0, "Speed went negative!"
    assert df["battery_soc"].max() <= 100.0, "SOC exceeded 100%!"
    assert (df["charging_state"].isin([0, 1])).all(), "Bad charging_state!"
    # While charging, current should be negative (into the pack).
    charging = df[df["charging_state"] == 1]
    assert (charging["pack_current_a"] <= 0).all(), "Charging current not negative!"

    print("\nAll physical sanity checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()