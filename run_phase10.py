"""
Phase 10 runner - complete end-to-end integration + demonstration.

Proves the full flow works:
    generate normal data -> inject simulated attacks -> extract features
    -> AI detects attack -> classify -> identify affected asset
    -> calculate TARA risk -> apply security response -> report.

Steps:
    1. Ensure the dataset + trained models exist (build/train if missing).
    2. Run the whole pipeline (Phases 1-8) via the orchestrator.
    3. For each attack type, show the canonical demo output using the
       IntegratedIDS façade on a window that truly contains that attack.
    4. Summarise per-attack detection coverage.
    5. Write results/integration_report.json and assert the flow works.

Run from the project root:
    python run_phase10.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.features.dataset import DatasetBuilder, _NON_FEATURE
from src.features.feature_extractor import FeatureExtractor
from src.integration.integrated_ids import IntegratedIDS
from src.pipeline import run_pipeline
from src.attacks.engine import AttackEngine
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.ids.train import IDSTrainer
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def _ensure_dataset_and_models(config) -> None:
    """Build the dataset and train models if they are not present."""
    logger = get_logger("run_phase10")

    if not os.path.exists("data/dataset.npz"):
        logger.info("dataset.npz missing - building it now ...")
        signals = SignalGenerator(config).generate()
        baseline = CANBus(config).generate_trace(signals)
        attacked, _ = AttackEngine(config).run(baseline)
        features = FeatureExtractor(config).extract(attacked)
        builder = DatasetBuilder(config)
        ds = builder.build(features)
        builder.save(features, ds)

    if not os.path.exists("models/binary_rf.joblib"):
        logger.info("models missing - training IDS now ...")
        IDSTrainer(config).train("data/dataset.npz")


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase10")

    logger.info("=== Phase 10: Complete Integration ===")
    _ensure_dataset_and_models(config)

    # ---- Run the full pipeline (Phases 1-8) ----
    result = run_pipeline()
    features = result.features
    feature_cols = [c for c in features.columns if c not in _NON_FEATURE]

    integrated = IntegratedIDS(config, "models")

    print("\n" + "=" * 78)
    print("PHASE 10 - END-TO-END DEMONSTRATION")
    print("Generate -> Attack -> Detect -> Asset -> Risk -> Security Response")
    print("=" * 78)

    # For each real attack type, find a window containing it and show the
    # complete integrated verdict.
    attack_types = ["spoofing", "replay", "dos", "injection",
                    "tampering", "charging_attack"]

    report = {"per_attack": {}, "coverage": {}}
    shown = set()

    for _, row in features.iterrows():
        true_type = row["label_multi"]
        if true_type == "normal" or true_type in shown:
            continue
        x = row[feature_cols].to_numpy(dtype=np.float32)
        verdict = integrated.analyze(x)
        shown.add(true_type)

        print(f"\n--- Injected attack: {true_type.upper()} "
              f"(window {row['start_s']:.2f}-{row['end_s']:.2f}s) ---")
        print(verdict.demo_string())
        report["per_attack"][true_type] = verdict.as_dict()

    # ---- Detection coverage per attack type (all windows) ----
    print("\n" + "=" * 78)
    print("DETECTION COVERAGE (all windows containing each attack)")
    print("=" * 78)
    print(f"{'attack type':18s} {'windows':>8s} {'detected':>9s} {'recall':>8s}")
    print("-" * 46)

    all_ok = True
    for atk in attack_types:
        sub = features[features["label_multi"] == atk]
        if sub.empty:
            continue
        detected = 0
        for _, row in sub.iterrows():
            x = row[feature_cols].to_numpy(dtype=np.float32)
            if integrated.analyze(x).is_attack:
                detected += 1
        recall = detected / len(sub)
        report["coverage"][atk] = {
            "windows": int(len(sub)),
            "detected": int(detected),
            "recall": round(recall, 3),
        }
        print(f"{atk:18s} {len(sub):8d} {detected:9d} {recall:8.2f}")
        if recall == 0.0:
            all_ok = False

    # ---- False-positive check on normal windows ----
    normal = features[features["label_multi"] == "normal"]
    fp = 0
    for _, row in normal.iterrows():
        x = row[feature_cols].to_numpy(dtype=np.float32)
        if integrated.analyze(x).is_attack:
            fp += 1
    fpr = fp / max(len(normal), 1)
    report["false_positive_rate"] = round(fpr, 3)
    print(f"\nNormal windows: {len(normal)} | false positives: {fp} "
          f"| FPR: {fpr:.2f}")

    # ---- Charging integration summary ----
    report["charging"] = {
        "flagged": result.charging_flagged,
        "total": result.charging_total,
    }
    print(f"Charging/grid: {result.charging_flagged}/{result.charging_total} "
          f"samples flagged as attack")

    # ---- Save consolidated report ----
    os.makedirs("results", exist_ok=True)
    with open("results/integration_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nSaved: results/integration_report.json")

    # ---- Sanity checks (the end-to-end flow must work) ----
    assert len(report["per_attack"]) >= 4, \
        "Most attack types should be demonstrable!"
    assert all_ok, "Every present attack type should be detected in >=1 window!"
    assert fpr <= 0.30, f"False-positive rate too high: {fpr:.2f}"
    assert result.charging_flagged > 0, "Charging attack not detected!"
    # Each demonstrated attack must have a risk score + action.
    for atk, v in report["per_attack"].items():
        assert v["risk_score"] is not None, f"{atk} has no risk score!"
        assert v["recommended_action"] is not None, f"{atk} has no action!"

    print("\n" + "=" * 78)
    print("ALL INTEGRATION CHECKS PASSED - complete flow works end to end.")
    print("=" * 78)


if __name__ == "__main__":
    main()