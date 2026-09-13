"""
Sliding-window utilities.

Splits a time-sorted CAN trace into overlapping time windows. Each window
is later summarized into one feature vector (see feature_extractor.py).
Windowing is what lets the IDS reason about *bus behaviour over time*
(frame rate, timing, ID mix) rather than isolated frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Window:
    """One time window over the CAN trace.

    Attributes:
        index: Sequential window number (0-based).
        start_s: Window start time (seconds).
        end_s: Window end time (seconds).
        frames: The CAN frames whose timestamp falls in [start_s, end_s).
    """

    index: int
    start_s: float
    end_s: float
    frames: pd.DataFrame


def iter_windows(
    trace: pd.DataFrame,
    window_ms: float,
    stride_ms: float,
) -> Iterator[Window]:
    """Yield sliding windows over a time-sorted CAN trace.

    Args:
        trace: CAN trace DataFrame with a 'timestamp' column (seconds).
        window_ms: Window length in milliseconds.
        stride_ms: Step between consecutive windows in milliseconds.

    Yields:
        Window objects in time order. Empty windows are skipped.
    """
    if "timestamp" not in trace.columns:
        raise ValueError("trace must contain a 'timestamp' column")

    window_s = window_ms / 1000.0
    stride_s = stride_ms / 1000.0

    t_min = float(trace["timestamp"].min())
    t_max = float(trace["timestamp"].max())

    ts = trace["timestamp"].to_numpy()

    index = 0
    start = t_min
    while start < t_max:
        end = start + window_s
        lo = np.searchsorted(ts, start, side="left")
        hi = np.searchsorted(ts, end, side="left")
        if hi > lo:
            frames = trace.iloc[lo:hi]
            yield Window(index=index, start_s=start, end_s=end, frames=frames)
            index += 1
        start += stride_s


def count_windows(trace: pd.DataFrame, window_ms: float, stride_ms: float) -> int:
    """Return how many non-empty windows a trace produces."""
    return sum(1 for _ in iter_windows(trace, window_ms, stride_ms))