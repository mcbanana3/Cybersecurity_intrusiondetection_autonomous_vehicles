"""
Phase 8 runner.

EV -> Charging Station -> Grid security demonstration:
    1. Generate a NORMAL charging session and an ATTACKED session.
    2. Compare grid frequency / current (normal vs attacked).
    3. Run the ChargingSecurityMonitor on the attacked session:
       plausibility detection -> TARA risk -> security response.
    4. Demonstrate a secured (AES-GCM + HMAC) charging control message.

Saves telemetry CSVs and an integrated report to results/.

Run from the project root:
    python run_phase8.py
"""

from __future__ import annotations

import json
import os

from src.charging.monitor import ChargingSecurityMonitor
from src.charging.session import ChargingSession
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase8")

    logger.info("=== Phase 8: EV Charging / Grid Security ===")

    os.makedirs("results", exist_ok=True)

    session = ChargingSession(config)
    normal = session.generate_normal()
    attacked = session.generate_attacked()

    normal.to_csv("results/charging_normal.csv", index=False)
    attacked.to_csv("results/charging_attacked.csv", index=False)

    print("\n" + "=" * 76)
    print("PHASE 8 COMPLETE - EV -> Charging Station -> Grid security")
    print("=" * 76)

    # ---- Normal vs attacked comparison ----
    a = config["charging_session"]["attack"]
    print("\n--- Normal vs attacked (grid frequency & current) ---")
    print(f"{'metric':22s} {'NORMAL':>12s} {'ATTACKED':>12s}")
    print("-" * 48)
    print(f"{'grid_freq mean (Hz)':22s} "
          f"{normal['grid_frequency_hz'].mean():12.3f} "
          f"{attacked['grid_frequency_hz'].mean():12.3f}")
    print(f"{'grid_freq min (Hz)':22s} "
          f"{normal['grid_frequency_hz'].min():12.3f} "
          f"{attacked['grid_frequency_hz'].min():12.3f}")
    print(f"{'current max (A)':22s} "
          f"{normal['current_a'].max():12.3f} "
          f"{attacked['current_a'].max():12.3f}")
    print(f"attack window: t={a['start_s']}-{a['end_s']}s")

    # ---- Security monitor on the attacked session ----
    monitor = ChargingSecurityMonitor(config)
    verdicts = monitor.evaluate(attacked)

    flagged = [v for v in verdicts if v.tampered]
    print(f"\n--- Security monitor: {len(flagged)}/{len(verdicts)} "
          f"samples detected as attack ---")

    if flagged:
        v = flagged[0]
        print("\nFirst detected charging attack sample:")
        print(f"  Time (s)          : {v.t_s}")
        print(f"  Grid frequency    : {v.grid_frequency_hz:.3f} Hz")
        print(f"  Current           : {v.current_a:.1f} A")
        print(f"  Attack detected   : YES ({v.detected_attack})")
        print(f"  Affected asset    : {v.affected_asset}")
        print(f"  Severity          : {v.risk_level}")
        print(f"  Risk score        : {v.risk_score}")
        print(f"  Recommended action: {v.response_action}")
        print(f"  Reasons           : {v.reasons}")

    # ---- Secure charging control channel ----
    secured_ok = monitor.secure_control_message(soc=55.0, current=90.0)
    print(f"\n--- Secure charging control channel (AES-GCM + HMAC) ---")
    print(f"  Encrypted+authenticated round-trip OK: {secured_ok}")

    # ---- Save integrated report ----
    report = {
        "normal_grid_freq_mean": float(normal["grid_frequency_hz"].mean()),
        "attacked_grid_freq_mean": float(attacked["grid_frequency_hz"].mean()),
        "attacked_grid_freq_min": float(attacked["grid_frequency_hz"].min()),
        "samples_total": len(verdicts),
        "samples_flagged": len(flagged),
        "secure_channel_ok": secured_ok,
        "verdicts": [v.as_dict() for v in verdicts],
    }
    with open("results/charging_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nSaved: results/charging_normal.csv, results/charging_attacked.csv,")
    print("       results/charging_report.json")

    # ---- Sanity checks ----
    assert len(flagged) > 0, "Charging attack was not detected!"
    assert attacked["grid_frequency_hz"].min() < 49.0, \
        "Attacked grid frequency should drop below plausible range!"
    assert normal["grid_frequency_hz"].min() >= 49.0, \
        "Normal grid frequency should stay plausible!"
    assert secured_ok is True, "Secure charging channel round-trip failed!"

    print("\n" + "=" * 76)
    print("All charging/grid security checks passed.")
    print("=" * 76)


if __name__ == "__main__":
    main()