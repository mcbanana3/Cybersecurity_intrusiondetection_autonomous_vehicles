"""
Phase 2 runner.

Pipeline: generate physical signals (Phase 1) -> build CAN trace (Phase 2).
Saves the CAN trace to data/normal_can.csv and verifies that a sample
frame decodes back to the original signal values (round-trip check).

Run from the project root:
    python run_phase2.py
"""

from __future__ import annotations

import numpy as np

from src.generator.can_bus import CANBus, CAN_COLUMNS
from src.generator.can_codec import decode_message, load_message_specs
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase2")

    logger.info("=== Phase 2: CAN-like Traffic Generation ===")

    # --- Phase 1: physical signals ---
    generator = SignalGenerator(config)
    signals = generator.generate()

    # --- Phase 2: CAN trace ---
    bus = CANBus(config)
    can_df = bus.generate_trace(signals)
    path = bus.save(can_df, "normal_can.csv")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE - CAN-like trace generated")
    print("=" * 70)
    print(f"Saved to        : {path}")
    print(f"Total frames    : {len(can_df)}")
    print(f"Columns         : {len(can_df.columns)}")
    print(f"Time span (s)   : {can_df['timestamp'].min():.2f} -> "
          f"{can_df['timestamp'].max():.2f}")

    print("\n--- Frames per message type ---")
    counts = can_df.groupby(["id_hex", "message_name"]).size()
    print(counts.to_string())

    print("\n--- First 8 CAN frames ---")
    show_cols = ["timestamp", "id_hex", "message_name", "dlc",
                 "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    print(can_df[show_cols].head(8).to_string(index=False))

    # --- Round-trip decode check on the first PWT_Speed frame ---
    specs = {s.name: s for s in load_message_specs(config)}
    pwt = specs["PWT_Speed"]
    first = can_df[can_df["message_name"] == "PWT_Speed"].iloc[0]
    payload = [int(first[f"b{i}"]) for i in range(8)]
    decoded = decode_message(pwt, payload)

    # Compare against the true signal at that timestamp.
    ts = float(first["timestamp"])
    idx = int(np.searchsorted(signals["timestamp"].to_numpy(), ts))
    idx = min(idx, len(signals) - 1)
    true_speed = float(signals.iloc[idx]["speed_kmh"])

    print("\n--- Decode round-trip check (PWT_Speed, first frame) ---")
    print(f"Timestamp          : {ts:.3f}s")
    print(f"Raw payload bytes  : {payload}")
    print(f"Decoded speed_kmh  : {decoded['speed_kmh']:.3f}")
    print(f"True    speed_kmh  : {true_speed:.3f}")
    print(f"Decode error       : {abs(decoded['speed_kmh'] - true_speed):.4f}")

    assert abs(decoded["speed_kmh"] - true_speed) < 0.05, "Decode round-trip failed!"
    assert list(can_df.columns) == CAN_COLUMNS, "CAN columns mismatch!"
    assert (can_df["label"] == "normal").all(), "Baseline trace must be all normal!"

    print("\nRound-trip and structure checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()