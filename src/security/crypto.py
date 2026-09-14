"""
AES-256-GCM authenticated encryption (simulated security control).

Provides confidentiality AND integrity: GCM produces an authentication
tag, so any tampering with the ciphertext causes decryption to fail.
Keys here are demo/test values from config -- NOT real automotive keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Ciphertext:
    """An AES-GCM encrypted payload.

    Attributes:
        nonce: 12-byte random nonce (unique per encryption).
        data: ciphertext + appended GCM authentication tag.
    """

    nonce: bytes
    data: bytes


class AESGCMCipher:
    """AES-256-GCM encrypt/decrypt using the configured key."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        key = bytes.fromhex(self.config["security"]["aes_key_hex"])
        if len(key) != 32:
            raise ValueError("aes_key_hex must be 32 bytes (64 hex chars)")
        self._aesgcm = AESGCM(key)

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> Ciphertext:
        """Encrypt plaintext, returning nonce + ciphertext(+tag).

        Args:
            plaintext: Bytes to protect.
            associated_data: Optional data that is authenticated but not
                encrypted (e.g. a CAN arbitration ID).
        """
        nonce = os.urandom(12)
        data = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return Ciphertext(nonce=nonce, data=data)

    def decrypt(self, ct: Ciphertext, associated_data: bytes = b"") -> bytes:
        """Decrypt and verify. Raises InvalidTag if tampered.

        Args:
            ct: The Ciphertext to decrypt.
            associated_data: Must match what was used at encryption time.

        Returns:
            The original plaintext bytes.

        Raises:
            InvalidTag: If the ciphertext/tag/associated data was altered.
        """
        return self._aesgcm.decrypt(ct.nonce, ct.data, associated_data)

    def try_decrypt(self, ct: Ciphertext, associated_data: bytes = b"") -> bytes | None:
        """Decrypt, returning None instead of raising on tampering."""
        try:
            return self.decrypt(ct, associated_data)
        except InvalidTag:
            logger.warning("AES-GCM decryption failed: authentication tag invalid")
            return None