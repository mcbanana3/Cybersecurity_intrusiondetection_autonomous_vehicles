"""
Secure diagnostic access (simulated UDS-style seed-key).

Flow:
    1. Client requests access -> server issues a random 'seed'.
    2. Client computes key = HMAC(secret, seed) and sends it back.
    3. Server recomputes and grants access only if keys match.

Failed attempts are counted; too many failures locks access. This models
protection against unauthorized diagnostic sessions.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AccessResult:
    """Result of a diagnostic access attempt."""

    granted: bool
    reason: str
    attempts_used: int


class DiagnosticAccess:
    """Server side of a seed-key challenge-response."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        sec = self.config["security"]
        self._secret = sec["diag_secret"].encode("utf-8")
        self._max_attempts = int(sec["max_diag_attempts"])
        self._attempts = 0
        self._locked = False
        self._current_seed: bytes | None = None

    def request_seed(self) -> bytes:
        """Issue a fresh random seed for the next authentication."""
        self._current_seed = os.urandom(8)
        return self._current_seed

    @staticmethod
    def compute_key(secret: bytes, seed: bytes) -> bytes:
        """Client-side key derivation: HMAC-SHA256(secret, seed)."""
        return hmac.new(secret, seed, hashlib.sha256).digest()

    def authenticate(self, provided_key: bytes) -> AccessResult:
        """Verify a client's key against the issued seed.

        Args:
            provided_key: The key the client computed from the seed.

        Returns:
            AccessResult indicating grant/deny and remaining state.
        """
        if self._locked:
            return AccessResult(False, "locked out (too many failed attempts)",
                                self._attempts)
        if self._current_seed is None:
            return AccessResult(False, "no seed issued; request seed first",
                                self._attempts)

        expected = self.compute_key(self._secret, self._current_seed)
        if hmac.compare_digest(expected, provided_key):
            self._attempts = 0
            self._current_seed = None
            return AccessResult(True, "access granted", self._attempts)

        self._attempts += 1
        if self._attempts >= self._max_attempts:
            self._locked = True
            return AccessResult(False, "incorrect key; now locked out",
                                self._attempts)
        return AccessResult(False, "incorrect key", self._attempts)