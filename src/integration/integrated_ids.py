"""
Integrated IDS façade.

Combines the detection, risk, and response layers behind one call so the
rest of the system (dashboard, CLI, tests) can go from a feature vector
to a complete decision in a single step:

    analyze(feature_vector) -> IntegratedVerdict
        1. IDS prediction        (Phase 5)  -> is_attack, type, confidence
        2. TARA risk assessment  (Phase 6)  -> asset, risk score/level
        3. Security response      (Phase 7)  -> mitigating control/action

This is the reusable "brain" that realises the end-to-end flow:
    detect -> identify asset -> score risk -> apply security response.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

from src.ids.predict import IDSPredictor
from src.security.responder import SecurityResponder
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntegratedVerdict:
    """A complete decision for one analysed window."""

    is_attack: bool
    attack_type: str
    confidence: float
    is_anomaly: bool
    affected_asset: str | None
    severity: str | None
    risk_score: float | None
    recommended_action: str | None
    security_control: str | None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def demo_string(self) -> str:
        """Return the canonical demo output block for this verdict."""
        if not self.is_attack:
            return "Attack detected: NO"
        return (
            f"Attack detected: YES\n"
            f"Attack type: {self.attack_type}\n"
            f"Confidence: {self.confidence*100:.0f}%\n"
            f"Affected asset: {self.affected_asset}\n"
            f"Severity: {self.severity}\n"
            f"Risk score: {self.risk_score}\n"
            f"Recommended action: {self.recommended_action}"
        )


class IntegratedIDS:
    """Unified detect -> risk -> respond pipeline over feature vectors."""

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        model_dir: str = "models",
    ) -> None:
        """Load all layers.

        Args:
            config: Optional pre-loaded config.
            model_dir: Directory containing trained IDS artifacts.
        """
        self.config = config or load_config()
        self.predictor = IDSPredictor(model_dir)
        self.risk_engine = RiskEngine(self.config)
        self.responder = SecurityResponder(self.config)
        logger.info("IntegratedIDS ready (%d features)",
                    len(self.predictor.feature_names))

    # ------------------------------------------------------------------
    def analyze(self, feature_vector: np.ndarray) -> IntegratedVerdict:
        """Run the full detect -> risk -> respond chain on one window.

        Args:
            feature_vector: 1-D feature vector matching the trained
                feature order (see predictor.feature_names).

        Returns:
            A fully populated IntegratedVerdict.
        """
        res = self.predictor.predict(feature_vector)

        if not res.is_attack:
            return IntegratedVerdict(
                is_attack=False,
                attack_type="normal",
                confidence=res.confidence,
                is_anomaly=res.is_anomaly,
                affected_asset=None,
                severity=None,
                risk_score=None,
                recommended_action=None,
                security_control=None,
            )

        ra = self.risk_engine.assess_attack_type(res.attack_type, detected=True)
        resp = self.responder.respond_to_threat(res.attack_type)

        return IntegratedVerdict(
            is_attack=True,
            attack_type=res.attack_type,
            confidence=res.confidence,
            is_anomaly=res.is_anomaly,
            affected_asset=ra.asset if ra else None,
            severity=ra.risk_level if ra else None,
            risk_score=ra.risk_score if ra else None,
            recommended_action=resp.action,
            security_control=resp.control,
        )

    def analyze_batch(self, X: np.ndarray) -> List[IntegratedVerdict]:
        """Analyse a 2-D array of feature vectors."""
        return [self.analyze(row) for row in np.asarray(X)]