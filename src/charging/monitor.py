"""
Charging security monitor.

Runs the defensive chain on charging/grid telemetry:

    telemetry sample -> plausibility/tamper check (Phase 7)
                     -> if anomalous: TARA risk (Phase 6, 'charging_attack')
                                      + security response (Phase 7)
                     -> secure the charging control channel with AES-GCM
                        + HMAC to demonstrate 'secure charging session'.

This ties the charging domain into the same IDS/TARA/security machinery
used elsewhere, without needing the CAN ML model (the charging attack
here is caught by physics-plausibility + integrity checks, which is the
natural control for false-data injection).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd

from src.security.crypto import AESGCMCipher
from src.security.message_auth import MessageAuthenticator, MessageVerifier
from src.security.responder import SecurityResponder
from src.security.tamper import TamperDetector
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChargingVerdict:
    """Security verdict for a single charging telemetry sample."""

    t_s: int
    grid_frequency_hz: float
    current_a: float
    tampered: bool
    reasons: List[str]
    detected_attack: str | None
    affected_asset: str | None
    risk_score: float | None
    risk_level: str | None
    response_action: str | None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChargingSecurityMonitor:
    """Applies plausibility + TARA + security response to charging data."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.detector = TamperDetector(self.config)
        self.risk_engine = RiskEngine(self.config)
        self.responder = SecurityResponder(self.config)
        self.cipher = AESGCMCipher(self.config)

    # ------------------------------------------------------------------
    def secure_control_message(self, soc: float, current: float) -> bool:
        """Demonstrate a secured charging control channel (AES-GCM + HMAC).

        Encrypts + authenticates a control command and verifies it on the
        receiving side. Returns True if the round-trip succeeds.
        """
        sender = MessageAuthenticator(self.config)
        verifier = MessageVerifier(self.config)

        command = f"SET_CURRENT soc={soc:.1f} current={current:.1f}".encode()
        ct = self.cipher.encrypt(command)
        decrypted = self.cipher.try_decrypt(ct)
        if decrypted is None:
            return False
        signed = sender.sign(decrypted)
        ok, _ = verifier.verify(signed)
        return ok

    # ------------------------------------------------------------------
    def evaluate(self, telemetry: pd.DataFrame) -> List[ChargingVerdict]:
        """Run the monitor over a charging telemetry DataFrame.

        Args:
            telemetry: Output of ChargingSession (normal or attacked).

        Returns:
            One ChargingVerdict per sample.
        """
        verdicts: List[ChargingVerdict] = []

        for _, row in telemetry.iterrows():
            values = {
                "grid_frequency_hz": float(row["grid_frequency_hz"]),
                # Reuse pack current plausibility loosely: extreme current
                # is implausible for this station's rated max.
            }
            checks = self.detector.check_signals(values)
            reasons = [r.reason for r in checks.values() if not r.ok]

            # Additional station-specific check: current above station max.
            station_max = float(self.config["charging_session"]["max_current_a"])
            if float(row["current_a"]) > station_max * 1.5:
                reasons.append(
                    f"current {row['current_a']:.1f}A exceeds station max "
                    f"{station_max:.1f}A"
                )

            tampered = len(reasons) > 0

            detected_attack = None
            affected_asset = None
            risk_score = None
            risk_level = None
            response_action = None

            if tampered:
                detected_attack = "charging_attack"
                ra = self.risk_engine.assess_attack_type(
                    "charging_attack", detected=True
                )
                if ra is not None:
                    affected_asset = ra.asset
                    risk_score = ra.risk_score
                    risk_level = ra.risk_level
                resp = self.responder.respond_to_threat("charging_attack")
                response_action = resp.action

            verdicts.append(ChargingVerdict(
                t_s=int(row["t_s"]),
                grid_frequency_hz=float(row["grid_frequency_hz"]),
                current_a=float(row["current_a"]),
                tampered=tampered,
                reasons=reasons,
                detected_attack=detected_attack,
                affected_asset=affected_asset,
                risk_score=risk_score,
                risk_level=risk_level,
                response_action=response_action,
            ))

        n_flag = sum(1 for v in verdicts if v.tampered)
        logger.info("Charging monitor: %d/%d samples flagged as attack",
                    n_flag, len(verdicts))
        return verdicts