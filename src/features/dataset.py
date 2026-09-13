"""
Dataset builder.

Takes the feature table (from FeatureExtractor) and produces the arrays
consumed by the Phase 5 IDS:

    * X            : numeric feature matrix (float32)
    * y_binary     : 0/1 attack labels
    * y_multi      : integer-encoded attack-type labels
    * feature_names, class_names

Splitting: we use a STRATIFIED shuffled split (fixed seed) so that every
attack class appears in train/val/test in proportion. A purely
contiguous time split was rejected here because each simulated attack
occupies a single time block, which would leave some classes entirely
out of the training split (breaking multi-class training). The original
time order is preserved in features.csv for later temporal experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are NOT model features.
_NON_FEATURE = {
    "window_index",
    "start_s",
    "end_s",
    "label_binary",
    "label_multi",
}


@dataclass
class Dataset:
    """Container for the ML-ready dataset."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    yb_train: np.ndarray
    yb_val: np.ndarray
    yb_test: np.ndarray
    ym_train: np.ndarray
    ym_val: np.ndarray
    ym_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]

    def summary(self) -> str:
        """Return a human-readable summary of the split sizes."""
        return (
            f"train={len(self.X_train)}  val={len(self.X_val)}  "
            f"test={len(self.X_test)}  features={len(self.feature_names)}  "
            f"classes={self.class_names}"
        )


class DatasetBuilder:
    """Builds and splits the ML dataset from a feature table."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        split = self.config["features"]["split"]
        self.train_frac = float(split["train"])
        self.val_frac = float(split["val"])
        self.test_frac = float(split["test"])
        self.seed = int(self.config["simulation"]["random_seed"])

    def build(self, features: pd.DataFrame) -> Dataset:
        """Build a Dataset from a feature table using a stratified split."""
        features = features.reset_index(drop=True)

        feature_names = [c for c in features.columns if c not in _NON_FEATURE]
        X = features[feature_names].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        yb = features["label_binary"].to_numpy(dtype=np.int64)

        # Encode multi-class labels contiguously; 'normal' is index 0.
        class_names = self._ordered_classes(features["label_multi"])
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        ym = features["label_multi"].map(class_to_idx).to_numpy(dtype=np.int64)

        # ---- Stratified split: first split off test, then val from remainder.
        # Stratify on the multi-class label so every class is represented.
        strat = ym if self._can_stratify(ym) else None

        X_tmp, X_test, yb_tmp, yb_test, ym_tmp, ym_test = train_test_split(
            X, yb, ym,
            test_size=self.test_frac,
            random_state=self.seed,
            stratify=strat,
        )

        # val fraction relative to the remaining (train+val) portion.
        val_relative = self.val_frac / (self.train_frac + self.val_frac)
        strat_tmp = ym_tmp if self._can_stratify(ym_tmp) else None

        X_train, X_val, yb_train, yb_val, ym_train, ym_val = train_test_split(
            X_tmp, yb_tmp, ym_tmp,
            test_size=val_relative,
            random_state=self.seed,
            stratify=strat_tmp,
        )

        ds = Dataset(
            X_train=X_train, X_val=X_val, X_test=X_test,
            yb_train=yb_train, yb_val=yb_val, yb_test=yb_test,
            ym_train=ym_train, ym_val=ym_val, ym_test=ym_test,
            feature_names=feature_names,
            class_names=class_names,
        )
        logger.info("Dataset built: %s", ds.summary())

        # Report class coverage per split (transparency, not fabricated).
        for name, arr in [("train", ym_train), ("val", ym_val), ("test", ym_test)]:
            present = sorted(set(arr.tolist()))
            logger.info("  %-5s classes present: %s", name,
                        [class_names[i] for i in present])
        return ds

    @staticmethod
    def _can_stratify(y: np.ndarray) -> bool:
        """Stratify only if every class has at least 2 samples."""
        if len(y) == 0:
            return False
        _, counts = np.unique(y, return_counts=True)
        return bool(counts.min() >= 2)

    def _ordered_classes(self, labels: pd.Series) -> List[str]:
        """Return class names with 'normal' first, others sorted."""
        present = set(labels.unique().tolist())
        ordered = ["normal"] if "normal" in present else []
        ordered += sorted(c for c in present if c != "normal")
        return ordered

    def save(
        self,
        features: pd.DataFrame,
        ds: Dataset,
        features_file: str = "features.csv",
        dataset_file: str = "dataset.npz",
    ) -> Tuple[str, str]:
        """Save the feature table (CSV) and the arrays (npz)."""
        out_dir = self.config["simulation"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        feat_path = os.path.join(out_dir, features_file)
        features.to_csv(feat_path, index=False)

        npz_path = os.path.join(out_dir, dataset_file)
        np.savez(
            npz_path,
            X_train=ds.X_train, X_val=ds.X_val, X_test=ds.X_test,
            yb_train=ds.yb_train, yb_val=ds.yb_val, yb_test=ds.yb_test,
            ym_train=ds.ym_train, ym_val=ds.ym_val, ym_test=ds.ym_test,
            feature_names=np.array(ds.feature_names),
            class_names=np.array(ds.class_names),
        )
        logger.info("Saved features -> %s", feat_path)
        logger.info("Saved dataset  -> %s", npz_path)
        return feat_path, npz_path


def load_dataset(path: str) -> Dict[str, Any]:
    """Load a saved dataset.npz into a plain dict of arrays/lists.

    Note: numpy loads string arrays as np.str_; we cast class_names /
    feature_names to plain Python str so downstream code and logs are clean.
    """
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["feature_names"] = [str(x) for x in out["feature_names"]]
    out["class_names"] = [str(x) for x in out["class_names"]]
    return out