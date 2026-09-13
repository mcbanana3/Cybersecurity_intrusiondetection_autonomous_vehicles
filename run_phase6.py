"""
Phase 6 runner.

Two demonstrations:

1. Static TARA table: risk for every catalogued threat, prioritized.

2. Live IDS -> TARA bridge: run the real pipeline (generate -> attack ->
   features -> IDS predict) and, for detected attacks, produce the
   risk assessment in the exact demo format:

       Attack detected: YES
       Attack type: <type>
       Confidence: <pct>
       Affected asset: <asset>
       Severity: <level>
       Risk score: <score>
       Recommended action: <action>

Prereq: models trained (run_phase5.py) and dataset built (run_phase4.py).

Run from the project root:
    python run_phase6.py
"""

from __future__ import annotations

import numpy as np

from src.attacks.engine import AttackEngine
from src.features.feature_extractor import FeatureExtractor
from src.features.dataset import _NON_FEATURE  # to align feature columns
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.ids.predict import IDSPredictor
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase6")

    logger.info("=== Phase 6: TARA / Risk Engine ===")

    engine = RiskEngine(config)

    # ---------- 1. Static prioritized TARA table ----------
    print("\n" + "=" * 78)
    print("PHASE 6 - PART 1: Static TARA table (active-detection risk), prioritized")
    print("=" * 78)
    table = engine.static_tara_table(detected=True)
    json_path, csv_path = engine.export(table)

    print(f"{'Attack':16s} {'Asset':26s} {'Impact':7s} {'Feas':5s} "
          f"{'Risk':6s} {'Level':9s}")
    print("-" * 78)
    for ra in table:
        print(f"{ra.attack_type:16s} {ra.asset:26s} "
              f"{ra.aggregated_impact:7.2f} {ra.feasibility:5d} "
              f"{ra.risk_score:6.2f} {ra.risk_level:9s}")
    print(f"\nSaved report -> {json_path}\nSaved table  -> {csv_path}")

    # ---------- 2. Live IDS -> TARA bridge ----------
    print("\n" + "=" * 78)
    print("PHASE 6 - PART 2: Live IDS -> TARA (detected attacks only)")
    print("=" * 78)

    # Build the attacked trace and features (same pipeline as Phase 4/5).
    signals = SignalGenerator(config).generate()
    baseline = CANBus(config).generate_trace(signals)
    attacked, _ = AttackEngine(config).run(baseline)
    features = FeatureExtractor(config).extract(attacked)

    predictor = IDSPredictor("models")
    feature_cols = [c for c in features.columns if c not in _NON_FEATURE]

    # For each attack type, grab one window that truly contains it and run
    # the IDS + risk engine, printing the specified demo output.
    shown = set()
    for _, row in features.iterrows():
        true_type = row["label_multi"]
        if true_type == "normal" or true_type in shown:
            continue
        x = row[feature_cols].to_numpy(dtype=np.float32)
        result = predictor.predict(x)
        ra = engine.assess(result)
        shown.add(true_type)

        print(f"\n[window {row['start_s']:.2f}-{row['end_s']:.2f}s | "
              f"true label = {true_type}]")
        print(f"  Attack detected   : {'YES' if result.is_attack else 'NO'}")
        if ra is not None:
            print(f"  Attack type       : {result.attack_type}")
            print(f"  Confidence        : {result.confidence*100:.0f}%")
            print(f"  Affected asset    : {ra.asset}")
            print(f"  Severity          : {ra.severity}")
            print(f"  Risk score        : {ra.risk_score}")
            print(f"  Recommended action: {ra.recommended_action}")
        else:
            print("  (IDS did not flag this window as an attack)")

    # ---------- Sanity checks ----------
    # Every catalogued threat should produce a valid level.
    assert all(ra.risk_level in {"Low", "Medium", "High", "Critical"}
               for ra in table)
    # Highest-risk threat should be at the top (sorted desc).
    assert table[0].risk_score >= table[-1].risk_score
    # A detected DoS should map to the CAN Bus asset.
    dos = engine.assess_attack_type("dos", detected=True)
    assert dos is not None and dos.asset == "CAN Bus"

    print("\n" + "=" * 78)
    print("All TARA sanity checks passed.")
    print("=" * 78)


if __name__ == "__main__":
    main()