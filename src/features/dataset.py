"""
Dataset builder.

Takes the feature table (from FeatureExtractor) and produces the arrays
consumed by the Phase 5 IDS:

    * X            : numeric feature matrix (float32)
    * y_binary     : 0/1 attack labels
    * y_multi      : integer-encoded attack-type labels
    * feature_names, class_names

Splitting is TIME-AWARE: windows are ordered by time and cut into
train / val / test contiguous blocks (no shuffling) so information from
the future never leaks into training -- important for a credible IDS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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

    def build(self, features: pd.DataFrame) -> Dataset:
        """Build a Dataset from a feature table.

        Args:
            features: Output of FeatureExtractor.extract().

        Returns:
            A populated Dataset with time-aware splits.
        """
        # Keep time order for a leakage-free split.
        features = features.sort_values("start_s").reset_index(drop=True)

        feature_names = [c for c in features.columns if c not in _NON_FEATURE]
        X = features[feature_names].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        yb = features["label_binary"].to_numpy(dtype=np.int64)

        # Encode multi-class labels; ensure 'normal' is index 0 for clarity.
        class_names = self._ordered_classes(features["label_multi"])
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        ym = features["label_multi"].map(class_to_idx).to_numpy(dtype=np.int64)

        n = len(features)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        idx_train = slice(0, n_train)
        idx_val = slice(n_train, n_train + n_val)
        idx_test = slice(n_train + n_val, n)

        ds = Dataset(
            X_train=X[idx_train], X_val=X[idx_val], X_test=X[idx_test],
            yb_train=yb[idx_train], yb_val=yb[idx_val], yb_test=yb[idx_test],
            ym_train=ym[idx_train], ym_val=ym[idx_val], ym_test=ym[idx_test],
            feature_names=feature_names,
            class_names=class_names,
        )
        logger.info("Dataset built: %s", ds.summary())
        return ds

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

    Args:
        path: Path to the .npz file.

    Returns:
        Dict with X_*/yb_*/ym_* arrays and feature_names/class_names lists.
    """
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["feature_names"] = list(out["feature_names"])
    out["class_names"] = list(out["class_names"])
    return out