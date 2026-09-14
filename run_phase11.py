"""
Phase 11 runner - evaluation, graphs, SHAP, experiments.

Generates all report/paper artifacts from the REAL trained models:
    * confusion matrix, per-class F1, ROC, PR, normal-vs-attack figures
    * SHAP global feature importance (plot + CSV)
    * model-comparison table (CSV)
    * evaluation_report.json + RESULTS.md summary

Prereq: data/dataset.npz + trained models (run run_phase4.py, run_phase5.py
or run_phase10.py once).

Run from the project root:
    python run_phase11.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from src.evaluation.experiments import run_model_comparison
from src.evaluation.explain import compute_shap_importance
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_normal_vs_attack,
    plot_per_class_f1,
    plot_pr,
    plot_roc,
)
from src.features.dataset import load_dataset
from src.attacks.engine import AttackEngine
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def _write_results_md(report: dict) -> str:
    """Write a short human-readable RESULTS.md for the write-up."""
    lines = [
        "# Experimental Results (auto-generated)",
        "",
        "> All figures and numbers are computed from the real trained "
        "models on the held-out test split. This is a software simulation "
        "for research/education and is not production-grade security.",
        "",
        "## Detection performance (test split)",
        f"- Binary accuracy: **{report['binary_accuracy']:.3f}**",
        f"- Binary F1 (macro): **{report['binary_f1_macro']:.3f}**",
        f"- Binary ROC-AUC: **{report['roc_auc']:.3f}**",
        f"- Binary PR-AUC (AP): **{report['pr_auc']:.3f}**",
        f"- Multi-class accuracy: **{report['multiclass_accuracy']:.3f}**",
        f"- Multi-class F1 (macro): **{report['multiclass_f1_macro']:.3f}**",
        "",
        "## Top SHAP features (binary detector)",
    ]
    for feat, val in report["top_shap_features"]:
        lines.append(f"- `{feat}`: {val:.4f}")
    lines += [
        "",
        "## Figures (results/figures/)",
        "- confusion_matrix.png",
        "- per_class_f1.png",
        "- roc_curve.png",
        "- pr_curve.png",
        "- normal_vs_attack.png",
        "- shap_importance.png",
        "",
        "## Model comparison",
        "See `results/model_comparison.csv`.",
    ]
    path = os.path.join("results", "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase11")

    logger.info("=== Phase 11: Evaluation, Graphs, SHAP, Experiments ===")

    if not os.path.exists("data/dataset.npz"):
        raise FileNotFoundError("data/dataset.npz missing - run run_phase4.py")
    if not os.path.exists("models/binary_rf.joblib"):
        raise FileNotFoundError("models missing - run run_phase5.py")

    os.makedirs("results", exist_ok=True)

    # ---- Load data + models + metrics ----
    data = load_dataset("data/dataset.npz")
    class_names = data["class_names"]
    feature_names = data["feature_names"]

    scaler = joblib.load("models/scaler.joblib")
    binary = joblib.load("models/binary_rf.joblib")

    with open("results/ids_metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    print("\n" + "=" * 74)
    print("PHASE 11 - EVALUATION & RESEARCH ARTIFACTS")
    print("=" * 74)

    # ---- 1. Confusion matrix + per-class F1 (multi-class, from metrics) ----
    mt = metrics["multiclass"]["test"]
    cm_names = mt.get("class_names", class_names)
    p_cm = plot_confusion_matrix(mt["confusion_matrix"], cm_names)
    p_f1 = plot_per_class_f1(metrics["multiclass"]["test_per_class_f1"])

    # ---- 2. ROC + PR (binary, computed live from probabilities) ----
    Xte_scaled = scaler.transform(data["X_test"])
    attack_proba = binary.predict_proba(Xte_scaled)[:, 1]
    yb_test = data["yb_test"]
    p_roc, roc_auc = plot_roc(yb_test, attack_proba)
    p_pr, pr_ap = plot_pr(yb_test, attack_proba)

    # ---- 3. Normal vs attack CAN frame-rate figure ----
    signals = SignalGenerator(config).generate()
    baseline = CANBus(config).generate_trace(signals)
    attacked, _ = AttackEngine(config).run(baseline)
    p_nva = plot_normal_vs_attack(attacked)

    # ---- 4. SHAP explainability ----
    print("\nComputing SHAP feature importance (may take a few seconds)...")
    p_shap, shap_df = compute_shap_importance(
        binary, scaler, data["X_test"], feature_names
    )
    shap_df.to_csv("results/shap_importance.csv", index=False)
    top_feats = list(zip(shap_df["feature"].head(8),
                         shap_df["mean_abs_shap"].head(8)))

    print("\n--- Top SHAP features (binary IDS) ---")
    for feat, val in top_feats:
        print(f"  {feat:28s} {val:.4f}")

    # ---- 5. Model comparison experiment ----
    print("\nRunning model comparison experiment...")
    comp = run_model_comparison(data, int(config["simulation"]["random_seed"]))
    comp.to_csv("results/model_comparison.csv", index=False)
    print("\n--- Model comparison (test split) ---")
    print(comp.to_string(index=False))

    # ---- 6. Consolidated report + RESULTS.md ----
    report = {
        "binary_accuracy": metrics["binary"]["test"]["accuracy"],
        "binary_f1_macro": metrics["binary"]["test"]["f1_macro"],
        "multiclass_accuracy": mt["accuracy"],
        "multiclass_f1_macro": mt["f1_macro"],
        "roc_auc": roc_auc,
        "pr_auc": pr_ap,
        "top_shap_features": [[f, float(v)] for f, v in top_feats],
        "figures": {
            "confusion_matrix": p_cm,
            "per_class_f1": p_f1,
            "roc_curve": p_roc,
            "pr_curve": p_pr,
            "normal_vs_attack": p_nva,
            "shap_importance": p_shap,
        },
        "model_comparison": comp.to_dict(orient="records"),
    }
    with open("results/evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md_path = _write_results_md(report)

    print("\n--- Saved artifacts ---")
    print("  Figures         : results/figures/*.png (6 files)")
    print("  SHAP table      : results/shap_importance.csv")
    print("  Model comparison: results/model_comparison.csv")
    print("  Report (JSON)   : results/evaluation_report.json")
    print(f"  Summary (MD)    : {md_path}")

    # ---- Sanity checks ----
    for p in report["figures"].values():
        assert os.path.exists(p), f"Figure not created: {p}"
    assert 0.0 <= roc_auc <= 1.0
    assert not shap_df.empty
    assert len(comp) >= 4

    print("\n" + "=" * 74)
    print("All evaluation artifacts generated successfully.")
    print("=" * 74)


if __name__ == "__main__":
    main()