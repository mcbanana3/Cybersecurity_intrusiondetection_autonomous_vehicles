"""
HMAC-SHA256 message authentication with monotonic counter.

This single control gives us BOTH:
    * Tamper detection  -> if the payload changes, the HMAC won't match.
    * Replay protection -> each message carries a strictly increasing
                           counter; a repeated or stale counter is rejected.

Used to simulate the 'enforce message authentication' and 'enforce
freshness counters' actions recommended by the TARA engine.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuthenticatedMessage:
    """A payload plus its freshness counter and HMAC tag."""

    payload: bytes
    counter: int
    tag: bytes


class MessageAuthenticator:
    """Signs and verifies messages using HMAC-SHA256 + a counter.

    A separate verifier state (last accepted counter) is kept per
    'channel' so replays are detected. Create one authenticator for the
    sender side and use a MessageVerifier on the receiver side.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self._key = bytes.fromhex(self.config["security"]["hmac_key_hex"])
        self._counter = 0

    def _compute_tag(self, payload: bytes, counter: int) -> bytes:
        msg = counter.to_bytes(8, "big") + payload
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def sign(self, payload: bytes) -> AuthenticatedMessage:
        """Sign a payload with the next counter value."""
        self._counter += 1
        tag = self._compute_tag(payload, self._counter)
        return AuthenticatedMessage(payload=payload, counter=self._counter, tag=tag)


class MessageVerifier:
    """Verifies authenticated messages and enforces counter freshness."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self._key = bytes.fromhex(self.config["security"]["hmac_key_hex"])
        self._last_counter = 0

    def _compute_tag(self, payload: bytes, counter: int) -> bytes:
        msg = counter.to_bytes(8, "big") + payload
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def verify(self, message: AuthenticatedMessage) -> tuple[bool, str]:
        """Verify integrity + freshness of a message.

        Returns:
            (accepted, reason). accepted is False for a bad HMAC (tamper)
            or a non-increasing counter (replay).
        """
        expected = self._compute_tag(message.payload, message.counter)
        if not hmac.compare_digest(expected, message.tag):
            return False, "HMAC mismatch (payload tampered or wrong key)"
        if message.counter <= self._last_counter:
            return False, (
                f"stale counter {message.counter} <= last "
                f"{self._last_counter} (replay detected)"
            )
        self._last_counter = message.counter
        return True, "accepted"