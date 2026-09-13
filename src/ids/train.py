"""
IDS training pipeline.

Trains and saves three models from the Phase 4 dataset:
    1. Binary detector      (Random Forest)     -> models/binary_rf.joblib
    2. Multi-class detector (XGBoost)           -> models/multiclass_xgb.joblib
    3. Anomaly detector     (Isolation Forest)  -> models/anomaly_iforest.joblib
A StandardScaler fitted on training features    -> models/scaler.joblib

All metrics are computed on held-out validation and test splits and
written to results/ids_metrics.json. No metrics are fabricated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.dataset import load_dataset
from src.ids.evaluate import evaluate_classification, per_class_f1
from src.ids.models import (
    build_anomaly_model,
    build_binary_model,
    build_multiclass_model,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainArtifacts:
    """Paths and metrics produced by training."""

    model_dir: str
    metrics_path: str
    metrics: Dict[str, Any]


class IDSTrainer:
    """Trains, evaluates and persists the IDS models."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.seed = int(self.config["simulation"]["random_seed"])
        self.model_dir = "models"
        self.results_dir = "results"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def train(self, dataset_path: str = "data/dataset.npz") -> TrainArtifacts:
        """Run the full training + evaluation pipeline."""
        data = load_dataset(dataset_path)
        class_names = data["class_names"]
        feature_names = data["feature_names"]

        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
        yb_train, yb_val, yb_test = data["yb_train"], data["yb_val"], data["yb_test"]
        ym_train, ym_val, ym_test = data["ym_train"], data["ym_val"], data["ym_test"]

        logger.info(
            "Loaded dataset: train=%d val=%d test=%d features=%d classes=%s",
            len(X_train), len(X_val), len(X_test), len(feature_names), class_names,
        )

        # ---- Scale features (fit on train only) ----
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xva = scaler.transform(X_val)
        Xte = scaler.transform(X_test)

        metrics: Dict[str, Any] = {"class_names": class_names,
                                   "feature_names": feature_names}

        # ---- 1. Binary detector ----
        logger.info("Training binary Random Forest ...")
        binary = build_binary_model(self.seed)
        binary.fit(Xtr, yb_train)
        metrics["binary"] = {
            "val": evaluate_classification(yb_val, binary.predict(Xva),
                                           ["normal", "attack"]),
            "test": evaluate_classification(yb_test, binary.predict(Xte),
                                            ["normal", "attack"]),
        }
        logger.info("Binary test accuracy: %.3f | F1(macro): %.3f",
                    metrics["binary"]["test"]["accuracy"],
                    metrics["binary"]["test"]["f1_macro"])

        # ---- 2. Multi-class detector ----
        # Defensive label remap: XGBoost requires contiguous labels 0..K-1
        # among the TRAINING labels. With stratified splitting all classes
        # are normally present, but we remap to be fully robust.
        logger.info("Training multi-class XGBoost ...")
        present = sorted(set(ym_train.tolist()))
        old_to_new = {old: new for new, old in enumerate(present)}
        new_to_old = {new: old for old, new in old_to_new.items()}

        def _remap(arr: np.ndarray) -> np.ndarray:
            # Unseen labels (not in training) map to -1 -> handled below.
            return np.array([old_to_new.get(int(v), -1) for v in arr],
                            dtype=np.int64)

        ym_train_r = _remap(ym_train)
        ym_val_r = _remap(ym_val)
        ym_test_r = _remap(ym_test)

        multiclass = build_multiclass_model(len(present), self.seed)
        multiclass.fit(Xtr, ym_train_r)

        # Predict, then map indices back to the original class space.
        def _predict_original(Xs: np.ndarray) -> np.ndarray:
            pred_new = multiclass.predict(Xs)
            return np.array([new_to_old[int(p)] for p in pred_new],
                            dtype=np.int64)

        ym_pred_val = _predict_original(Xva)
        ym_pred_test = _predict_original(Xte)

        metrics["multiclass"] = {
            "val": evaluate_classification(ym_val, ym_pred_val, class_names),
            "test": evaluate_classification(ym_test, ym_pred_test, class_names),
            "test_per_class_f1": per_class_f1(ym_test, ym_pred_test, class_names),
            "trained_classes": [class_names[i] for i in present],
        }
        logger.info("Multi-class test accuracy: %.3f | F1(macro): %.3f",
                    metrics["multiclass"]["test"]["accuracy"],
                    metrics["multiclass"]["test"]["f1_macro"])

        # ---- 3. Anomaly detector (train on NORMAL windows only) ----
        logger.info("Training Isolation Forest on normal windows ...")
        normal_mask = yb_train == 0
        anomaly = build_anomaly_model(self.seed)
        anomaly.fit(Xtr[normal_mask])
        raw = anomaly.predict(Xte)
        anomaly_pred = (raw == -1).astype(int)  # 1 = flagged anomaly
        metrics["anomaly"] = {
            "test": evaluate_classification(yb_test, anomaly_pred,
                                            ["normal", "anomaly"]),
            "note": ("Isolation Forest trained on normal windows only; "
                     "evaluated against the binary attack label as a proxy "
                     "for unknown-attack detection."),
        }
        logger.info("Anomaly detector test recall (macro): %.3f",
                    metrics["anomaly"]["test"]["recall_macro"])

        # ---- Persist artifacts ----
        joblib.dump(scaler, os.path.join(self.model_dir, "scaler.joblib"))
        joblib.dump(binary, os.path.join(self.model_dir, "binary_rf.joblib"))
        joblib.dump(multiclass, os.path.join(self.model_dir, "multiclass_xgb.joblib"))
        joblib.dump(anomaly, os.path.join(self.model_dir, "anomaly_iforest.joblib"))

        # Save metadata AND the label remap so the predictor stays correct.
        joblib.dump(
            {
                "feature_names": feature_names,
                "class_names": class_names,
                "xgb_new_to_old": new_to_old,
            },
            os.path.join(self.model_dir, "metadata.joblib"),
        )

        metrics_path = os.path.join(self.results_dir, "ids_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Saved models to %s/ and metrics to %s",
                    self.model_dir, metrics_path)

        return TrainArtifacts(self.model_dir, metrics_path, metrics)