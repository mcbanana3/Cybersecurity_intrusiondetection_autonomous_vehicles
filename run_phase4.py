"""
Phase 4 runner.

Pipeline: signals (P1) -> CAN trace (P2) -> attacks (P3) -> features (P4).
Generates the attacked trace, extracts windowed features, builds a
time-aware train/val/test dataset, saves features.csv + dataset.npz, and
prints a summary.

Run from the project root:
    python run_phase4.py
"""

from __future__ import annotations

from src.attacks.engine import AttackEngine
from src.features.dataset import DatasetBuilder
from src.features.feature_extractor import FeatureExtractor
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase4")

    logger.info("=== Phase 4: Dataset Pipeline & Feature Engineering ===")

    # P1 -> P2 -> P3
    signals = SignalGenerator(config).generate()
    baseline = CANBus(config).generate_trace(signals)
    attacked, _events = AttackEngine(config).run(baseline)

    # P4: features + dataset
    extractor = FeatureExtractor(config)
    features = extractor.extract(attacked)

    builder = DatasetBuilder(config)
    ds = builder.build(features)
    feat_path, npz_path = builder.save(features, ds)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE - ML-ready dataset built")
    print("=" * 70)
    print(f"Windows (rows)  : {len(features)}")
    print(f"Feature count   : {len(ds.feature_names)}")
    print(f"Classes         : {ds.class_names}")
    print(f"Split sizes     : train={len(ds.X_train)}  "
          f"val={len(ds.X_val)}  test={len(ds.X_test)}")
    print(f"Features file   : {feat_path}")
    print(f"Dataset file    : {npz_path}")

    print("\n--- Binary label balance (all windows) ---")
    print(features["label_binary"].value_counts().rename(
        {0: "normal", 1: "attack"}).to_string())

    print("\n--- Multi-class label balance (all windows) ---")
    print(features["label_multi"].value_counts().to_string())

    print("\n--- Feature names ---")
    print(", ".join(ds.feature_names))

    print("\n--- Example attack window (first DoS-labelled window) ---")
    dos_rows = features[features["label_multi"] == "dos"]
    if not dos_rows.empty:
        row = dos_rows.iloc[0]
        print(f"time {row['start_s']:.2f}-{row['end_s']:.2f}s | "
              f"frame_count={row['frame_count']:.0f} | "
              f"frame_rate_hz={row['frame_rate_hz']:.0f} | "
              f"unknown_id_count={row['unknown_id_count']:.0f} | "
              f"min_id={row['min_id']:.0f}")

    print("\n--- Example normal window (first) ---")
    norm_rows = features[features["label_multi"] == "normal"]
    if not norm_rows.empty:
        row = norm_rows.iloc[0]
        print(f"time {row['start_s']:.2f}-{row['end_s']:.2f}s | "
              f"frame_count={row['frame_count']:.0f} | "
              f"frame_rate_hz={row['frame_rate_hz']:.0f} | "
              f"unknown_id_count={row['unknown_id_count']:.0f}")

    # --- Sanity checks ---
    assert len(features) > 0, "No windows extracted!"
    assert features["label_binary"].sum() > 0, "No attack windows!"
    assert (features["label_binary"] == 0).sum() > 0, "No normal windows!"
    assert "normal" == ds.class_names[0], "'normal' should be class index 0!"
    if not dos_rows.empty and not norm_rows.empty:
        assert dos_rows["frame_count"].mean() > norm_rows["frame_count"].mean(), \
            "DoS windows should have more frames than normal!"

    print("\nAll dataset sanity checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()