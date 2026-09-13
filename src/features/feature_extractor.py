"""
Feature extractor.

Turns each time window of CAN frames into a single numeric feature
vector plus its labels. Features are engineered to expose the attack
signatures created in Phase 3:

    * DoS/injection  -> frame count spikes, tiny inter-arrival times,
                        unknown IDs, ID-count changes.
    * spoofing       -> extra frames on a known ID, decoded speed jumps.
    * tampering      -> payload byte entropy / decoded-value range spikes.
    * charging attack-> abnormal decoded grid frequency / charging state.
    * replay         -> extra frames + repeated payloads on a known ID.

Labels per window:
    label_binary : 0 = normal, 1 = attack
    label_multi  : 'normal' or the dominant attack_type in the window
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features.windowing import Window, iter_windows
from src.generator.can_codec import MessageSpec, decode_message, load_message_specs
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Messages whose decoded signals we track explicitly (attack-sensitive).
_TRACKED = {
    "PWT_Speed": ["speed_kmh"],
    "BMS_State": ["battery_soc", "pack_voltage_v", "battery_temp_c"],
    "CHG_Status": ["grid_frequency_hz", "charging_state"],
}


def _byte_entropy(frames: pd.DataFrame) -> float:
    """Shannon entropy (bits) of all payload byte values in the window."""
    byte_cols = [f"b{i}" for i in range(8)]
    values = frames[byte_cols].to_numpy().astype(int).ravel()
    if values.size == 0:
        return 0.0
    counts = np.bincount(values, minlength=256).astype(float)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


class FeatureExtractor:
    """Computes feature vectors + labels for windows of a CAN trace."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.specs: Dict[str, MessageSpec] = {
            s.name: s for s in load_message_specs(self.config)
        }
        # Legitimate IDs = those defined in the CAN model. Anything else
        # seen in a window is an "unknown" (injection/DoS) ID.
        self.known_ids = {int(s.arbitration_id) for s in self.specs.values()}
        self.window_ms = float(self.config["features"]["window_ms"])
        self.stride_ms = float(self.config["features"]["stride_ms"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Extract a feature table (one row per window) from a CAN trace.

        Args:
            trace: Attacked (or normal) CAN trace DataFrame.

        Returns:
            DataFrame with numeric feature columns plus:
            'window_index', 'start_s', 'end_s', 'label_binary', 'label_multi'.
        """
        rows: List[Dict[str, Any]] = []
        for window in iter_windows(trace, self.window_ms, self.stride_ms):
            rows.append(self._features_for_window(window))

        df = pd.DataFrame(rows)
        n_attack = int((df["label_binary"] == 1).sum())
        logger.info(
            "Extracted %d windows (%d attack, %d normal)",
            len(df), n_attack, len(df) - n_attack,
        )
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _features_for_window(self, window: Window) -> Dict[str, Any]:
        frames = window.frames
        feats: Dict[str, Any] = {}

        # ---- Timing / rate features ----
        ts = frames["timestamp"].to_numpy()
        feats["frame_count"] = int(len(frames))
        if len(ts) >= 2:
            iat = np.diff(np.sort(ts))
            feats["iat_mean"] = float(iat.mean())
            feats["iat_std"] = float(iat.std())
            feats["iat_min"] = float(iat.min())
        else:
            feats["iat_mean"] = 0.0
            feats["iat_std"] = 0.0
            feats["iat_min"] = 0.0
        feats["frame_rate_hz"] = float(
            len(frames) / max(self.window_ms / 1000.0, 1e-6)
        )

        # ---- Arbitration-ID features ----
        ids = frames["arbitration_id"].astype(int).to_numpy()
        unique_ids = set(ids.tolist())
        feats["unique_id_count"] = int(len(unique_ids))
        feats["unknown_id_count"] = int(len(unique_ids - self.known_ids))
        feats["min_id"] = int(ids.min()) if ids.size else 0
        if ids.size:
            _, counts = np.unique(ids, return_counts=True)
            feats["top_id_fraction"] = float(counts.max() / counts.sum())
        else:
            feats["top_id_fraction"] = 0.0

        # ---- Payload features ----
        byte_cols = [f"b{i}" for i in range(8)]
        payload_vals = frames[byte_cols].to_numpy().astype(float)
        if payload_vals.size:
            feats["payload_mean"] = float(payload_vals.mean())
            feats["payload_std"] = float(payload_vals.std())
        else:
            feats["payload_mean"] = 0.0
            feats["payload_std"] = 0.0
        feats["payload_entropy"] = _byte_entropy(frames)

        # ---- Decoded-signal features for attack-sensitive messages ----
        for msg_name, sig_names in _TRACKED.items():
            spec = self.specs.get(msg_name)
            sub = frames[frames["message_name"] == msg_name]
            for sig in sig_names:
                col_max = f"{msg_name}_{sig}_max"
                col_min = f"{msg_name}_{sig}_min"
                col_rng = f"{msg_name}_{sig}_range"
                if spec is None or sub.empty:
                    feats[col_max] = 0.0
                    feats[col_min] = 0.0
                    feats[col_rng] = 0.0
                    continue
                decoded_vals = []
                for _, fr in sub.iterrows():
                    payload = [int(fr[f"b{i}"]) for i in range(8)]
                    decoded_vals.append(decode_message(spec, payload)[sig])
                arr = np.array(decoded_vals, dtype=float)
                feats[col_max] = float(arr.max())
                feats[col_min] = float(arr.min())
                feats[col_rng] = float(arr.max() - arr.min())

        # ---- Labels ----
        attack_frames = frames[frames["label"] == "attack"]
        feats["window_index"] = window.index
        feats["start_s"] = round(window.start_s, 4)
        feats["end_s"] = round(window.end_s, 4)
        feats["label_binary"] = int(len(attack_frames) > 0)

        if len(attack_frames) > 0:
            feats["label_multi"] = (
                attack_frames["attack_type"].value_counts().idxmax()
            )
        else:
            feats["label_multi"] = "normal"

        return feats