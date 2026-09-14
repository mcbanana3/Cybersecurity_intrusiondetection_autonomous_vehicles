"""
Security responder.

Maps a detected attack type to the concrete simulated security control
that mitigates it, producing a SecurityResponse. This is the 'apply
security control' step that follows IDS detection and TARA scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityResponse:
    """The security control applied in response to a threat."""

    attack_type: str
    control: str        # which mechanism handles it
    action: str         # concrete action taken
    outcome: str        # simulated result

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Attack type -> (control mechanism, concrete action).
_RESPONSE_MAP = {
    "spoofing": (
        "HMAC message authentication + plausibility check",
        "Reject forged frames; drop implausible speed values",
    ),
    "replay": (
        "HMAC + monotonic counter",
        "Reject stale/duplicate counters (freshness enforced)",
    ),
    "dos": (
        "Rate limiting / bus isolation",
        "Throttle flood source; isolate affected bus segment",
    ),
    "injection": (
        "ID allow-list filtering",
        "Drop frames with unknown/unauthorized arbitration IDs",
    ),
    "tampering": (
        "HMAC integrity + plausibility check",
        "Reject frames failing integrity or range validation",
    ),
    "charging_attack": (
        "AES-GCM secure session + plausibility check",
        "Reject false grid telemetry; secure charging session",
    ),
}


class SecurityResponder:
    """Selects and 'applies' a control for a given attack type."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()

    def respond_to_threat(self, attack_type: str) -> SecurityResponse:
        """Return the security response for an attack type.

        Args:
            attack_type: The detected attack type (from IDS/TARA).

        Returns:
            A SecurityResponse describing the mitigating control.
        """
        control, action = _RESPONSE_MAP.get(
            attack_type,
            ("Generic IDS alert", "Log event and raise alert for review"),
        )
        response = SecurityResponse(
            attack_type=attack_type,
            control=control,
            action=action,
            outcome="control applied (simulated) - malicious message blocked",
        )
        logger.info("Security response for '%s': %s", attack_type, action)
        return response