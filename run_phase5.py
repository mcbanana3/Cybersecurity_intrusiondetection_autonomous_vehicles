"""
Phase 5 runner.

Trains the IDS models on data/dataset.npz, prints real evaluation
metrics, then demonstrates the predictor on a few test windows.

Prereq: run_phase4.py must have produced data/dataset.npz.

Run from the project root:
    python run_phase5.py
"""

from __future__ import annotations

import numpy as np

from src.features.dataset import load_dataset
from src.ids.predict import IDSPredictor
from src.ids.train import IDSTrainer
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def _print_confusion(cm, names) -> None:
    header = "            " + " ".join(f"{n[:8]:>9s}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        label = names[i] if i < len(names) else str(i)
        print(f"{label[:11]:>11s} " + " ".join(f"{v:9d}" for v in row))


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase5")

    logger.info("=== Phase 5: AI-based Intrusion Detection ===")

    # --- Train ---
    trainer = IDSTrainer(config)
    artifacts = trainer.train("data/dataset.npz")
    m = artifacts.metrics

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE - IDS models trained and evaluated")
    print("=" * 70)

    print("\n--- BINARY detector (normal vs attack) [TEST] ---")
    bt = m["binary"]["test"]
    print(f"Accuracy      : {bt['accuracy']:.3f}")
    print(f"Precision(mac): {bt['precision_macro']:.3f}")
    print(f"Recall(macro) : {bt['recall_macro']:.3f}")
    print(f"F1(macro)     : {bt['f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    _print_confusion(bt["confusion_matrix"], ["normal", "attack"])

    print("\n--- MULTI-CLASS classifier (attack type) [TEST] ---")
    mt = m["multiclass"]["test"]
    print(f"Accuracy  : {mt['accuracy']:.3f}")
    print(f"F1(macro) : {mt['f1_macro']:.3f}")
    print("Per-class F1:")
    for cls, f1 in m["multiclass"]["test_per_class_f1"].items():
        print(f"  {cls:18s}: {f1:.3f}")

    print("\n--- ANOMALY detector (Isolation Forest) [TEST] ---")
    at = m["anomaly"]["test"]
    print(f"Accuracy      : {at['accuracy']:.3f}")
    print(f"Recall(macro) : {at['recall_macro']:.3f}")
    print(f"(note) {m['anomaly']['note']}")

    # --- Demonstrate the predictor interface on real test windows ---
    print("\n--- Predictor demo on sample TEST windows ---")
    data = load_dataset("data/dataset.npz")
    predictor = IDSPredictor("models")
    X_test = data["X_test"]
    ym_test = data["ym_test"]
    class_names = data["class_names"]

    # Show one normal window and one of each attack window if available.
    shown = set()
    for i in range(len(X_test)):
        true_cls = class_names[ym_test[i]]
        if true_cls in shown:
            continue
        shown.add(true_cls)
        res = predictor.predict(X_test[i])
        print(f"[true={true_cls:16s}] -> "
              f"attack={str(res.is_attack):5s} | "
              f"type={res.attack_type:16s} | "
              f"conf={res.confidence:.2f} | "
              f"anomaly={str(res.is_anomaly):5s}")
        if len(shown) >= len(class_names):
            break

    # --- Sanity checks ---
    assert bt["accuracy"] >= 0.7, "Binary accuracy unexpectedly low!"
    assert mt["accuracy"] >= 0.6, "Multi-class accuracy unexpectedly low!"

    print("\nAll IDS sanity checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()