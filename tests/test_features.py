"""
Phase 4 tests: windowing, feature extraction, dataset building.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.attacks.engine import AttackEngine
from src.features.dataset import DatasetBuilder
from src.features.feature_extractor import FeatureExtractor
from src.features.windowing import iter_windows
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.utils.config_loader import load_config


def _attacked_trace() -> pd.DataFrame:
    config = load_config()
    signals = SignalGenerator(config).generate()
    baseline = CANBus(config).generate_trace(signals)
    attacked, _ = AttackEngine(config).run(baseline)
    return attacked


def test_windows_are_non_empty_and_ordered() -> None:
    trace = _attacked_trace()
    wins = list(iter_windows(trace, 500, 250))
    assert len(wins) > 0
    starts = [w.start_s for w in wins]
    assert starts == sorted(starts)
    assert all(len(w.frames) > 0 for w in wins)


def test_feature_table_has_labels() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    for col in ["label_binary", "label_multi", "frame_count", "unknown_id_count"]:
        assert col in feats.columns
    assert set(feats["label_binary"].unique()).issubset({0, 1})


def test_both_classes_present() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    assert feats["label_binary"].sum() > 0
    assert (feats["label_binary"] == 0).sum() > 0


def test_dos_windows_have_more_frames() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    dos = feats[feats["label_multi"] == "dos"]
    normal = feats[feats["label_multi"] == "normal"]
    assert not dos.empty
    assert dos["frame_count"].mean() > normal["frame_count"].mean()


def test_injection_windows_have_unknown_ids() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    inj = feats[feats["label_multi"] == "injection"]
    assert not inj.empty
    assert inj["unknown_id_count"].max() >= 1


def test_dataset_split_sizes_and_no_nan() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    ds = DatasetBuilder(config).build(feats)
    total = len(ds.X_train) + len(ds.X_val) + len(ds.X_test)
    assert total == len(feats)
    assert not np.isnan(ds.X_train).any()
    assert ds.class_names[0] == "normal"


def test_dataset_feature_matrix_shape() -> None:
    config = load_config()
    feats = FeatureExtractor(config).extract(_attacked_trace())
    ds = DatasetBuilder(config).build(feats)
    assert ds.X_train.shape[1] == len(ds.feature_names)
    assert ds.X_train.shape[1] > 5