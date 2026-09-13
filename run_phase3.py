"""
Phase 3 runner.

Pipeline: signals (P1) -> CAN trace (P2) -> attacks (P3).
Generates a clean baseline trace, injects all enabled simulated attacks,
saves the attacked trace + attack log, and prints a summary comparing
normal vs attacked traffic.

Run from the project root:
    python run_phase3.py
"""

from __future__ import annotations

from src.attacks.engine import AttackEngine
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase3")

    logger.info("=== Phase 3: Cyberattack Simulation Engine ===")

    # P1 + P2: baseline clean trace
    signals = SignalGenerator(config).generate()
    baseline = CANBus(config).generate_trace(signals)

    # P3: inject attacks
    engine = AttackEngine(config)
    attacked, events = engine.run(baseline)
    trace_path, log_path = engine.save(attacked, events)

    # --- Summary ---
    total = len(attacked)
    n_attack = int((attacked["label"] == "attack").sum())
    n_normal = total - n_attack

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE - Attacks injected into synthetic CAN trace")
    print("=" * 70)
    print(f"Baseline frames : {len(baseline)}")
    print(f"Attacked frames : {total}")
    print(f"  normal        : {n_normal}")
    print(f"  attack        : {n_attack}  ({100*n_attack/total:.1f}%)")
    print(f"Attacked trace  : {trace_path}")
    print(f"Attack log      : {log_path}")

    print("\n--- Frames by attack_type ---")
    print(attacked["attack_type"].value_counts().to_string())

    print("\n--- Attack log (one row per attack event) ---")
    for e in events:
        d = e.as_dict()
        print(
            f"[{d['attack_type']:16s}] {d['start_s']:6.1f}-{d['end_s']:6.1f}s | "
            f"asset={d['target_asset']:26s} | id={d['id_hex']:5s} | "
            f"frames={d['frames_affected']}"
        )

    print("\n--- Normal vs attacked: frame rate in 1s bins (sample) ---")
    attacked["sec"] = attacked["timestamp"].astype(int)
    rate = attacked.groupby("sec").size()
    # Show a few key windows where attacks live.
    for window, label in [((0, 5), "baseline"), ((60, 65), "DoS"),
                          ((90, 95), "tampering"), ((105, 115), "charging")]:
        seg = rate[(rate.index >= window[0]) & (rate.index < window[1])]
        print(f"  {label:10s} t={window[0]}-{window[1]}s: "
              f"avg {seg.mean():.0f} frames/s")

    # --- Sanity checks ---
    assert n_attack > 0, "No attack frames were produced!"
    assert set(attacked["label"].unique()).issubset({"normal", "attack"})
    assert len(events) >= 5, "Expected several attack events!"
    # DoS window must have a much higher frame rate than baseline.
    dos_rate = rate[(rate.index >= 60) & (rate.index < 65)].mean()
    base_rate = rate[(rate.index >= 0) & (rate.index < 5)].mean()
    assert dos_rate > base_rate * 2, "DoS did not raise the frame rate!"

    print("\nAll attack-injection checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()