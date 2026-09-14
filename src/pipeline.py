"""
End-to-end pipeline orchestrator (for the dashboard and integration).

Runs the full chain in one call and returns a bundle of everything the
UI needs:

    signals   -> CAN trace -> attacked trace -> features
    -> IDS predictions per window
    -> TARA risk per detected window
    -> security response per detected window

The IDS models are loaded from models/ (train them with run_phase5.py).
Charging/grid telemetry (Phase 8) is included too.

This module writes NO files; it just computes and returns data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.attacks.engine import AttackEngine
from src.charging.monitor import ChargingSecurityMonitor
from src.charging.session import ChargingSession
from src.features.dataset import _NON_FEATURE
from src.features.feature_extractor import FeatureExtractor
from src.generator.can_bus import CANBus
from src.generator.signal_generator import SignalGenerator
from src.ids.predict import IDSPredictor
from src.security.responder import SecurityResponder
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WindowVerdict:
    """Combined IDS + TARA + security result for one feature window."""

    start_s: float
    end_s: float
    true_label: str
    is_attack: bool
    attack_type: str
    confidence: float
    is_anomaly: bool
    affected_asset: str | None
    risk_score: float | None
    risk_level: str | None
    recommended_action: str | None


@dataclass
class PipelineResult:
    """Everything the dashboard needs from one pipeline run."""

    signals: pd.DataFrame
    can_normal: pd.DataFrame
    can_attacked: pd.DataFrame
    features: pd.DataFrame
    verdicts: List[WindowVerdict]
    charging_normal: pd.DataFrame
    charging_attacked: pd.DataFrame
    charging_flagged: int
    charging_total: int
    config: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(config_overrides: Dict[str, Any] | None = None) -> PipelineResult:
    """Run the full pipeline and return a PipelineResult.

    Args:
        config_overrides: Optional dict merged into config['attacks'] to
            enable/disable attacks from the UI. Shape:
            {"spoofing": {"enabled": False}, ...}

    Returns:
        A populated PipelineResult.
    """
    config = load_config()

    # Apply attack enable/disable overrides (deep-ish merge for 'attacks').
    if config_overrides:
        for atk, patch in config_overrides.items():
            if atk in config["attacks"]:
                config["attacks"][atk].update(patch)

    # ---- P1 -> P2 -> P3 ----
    signals = SignalGenerator(config).generate()
    bus = CANBus(config)
    can_normal = bus.generate_trace(signals)
    attacked, _events = AttackEngine(config).run(can_normal)

    # ---- P4 features ----
    features = FeatureExtractor(config).extract(attacked)
    feature_cols = [c for c in features.columns if c not in _NON_FEATURE]

    # ---- P5 IDS + P6 TARA + P7 response ----
    predictor = IDSPredictor("models")
    risk_engine = RiskEngine(config)
    responder = SecurityResponder(config)

    verdicts: List[WindowVerdict] = []
    for _, row in features.iterrows():
        x = row[feature_cols].to_numpy(dtype=np.float32)
        res = predictor.predict(x)

        asset = risk_score = risk_level = action = None
        if res.is_attack:
            ra = risk_engine.assess_attack_type(res.attack_type, detected=True)
            if ra is not None:
                asset = ra.asset
                risk_score = ra.risk_score
                risk_level = ra.risk_level
            action = responder.respond_to_threat(res.attack_type).action

        verdicts.append(WindowVerdict(
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            true_label=str(row["label_multi"]),
            is_attack=res.is_attack,
            attack_type=res.attack_type,
            confidence=res.confidence,
            is_anomaly=res.is_anomaly,
            affected_asset=asset,
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=action,
        ))

    # ---- P8 charging ----
    session = ChargingSession(config)
    charging_normal = session.generate_normal()
    charging_attacked = session.generate_attacked()
    charging_verdicts = ChargingSecurityMonitor(config).evaluate(charging_attacked)
    charging_flagged = sum(1 for v in charging_verdicts if v.tampered)

    logger.info("Pipeline complete: %d windows, %d flagged as attack",
                len(verdicts), sum(1 for v in verdicts if v.is_attack))

    return PipelineResult(
        signals=signals,
        can_normal=can_normal,
        can_attacked=attacked,
        features=features,
        verdicts=verdicts,
        charging_normal=charging_normal,
        charging_attacked=charging_attacked,
        charging_flagged=charging_flagged,
        charging_total=len(charging_verdicts),
        config=config,
    )