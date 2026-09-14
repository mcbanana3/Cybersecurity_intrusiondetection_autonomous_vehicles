"""
OTA update security (simulated).

An update server SIGNS a package (RSA over a SHA-256 hash). The vehicle
verifies BOTH the signature and the hash before 'installing'. A forged
package (bad signature) or a corrupted package (hash/signature mismatch)
is rejected. Keys are generated in-memory for the simulation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OTAPackage:
    """An OTA update package with its hash and signature."""

    payload: bytes
    sha256: str
    signature: bytes


class OTAUpdateServer:
    """Signs OTA packages with an RSA private key (simulation)."""

    def __init__(self) -> None:
        # 2048-bit RSA keypair generated in-memory for the demo.
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        self.public_key = self._private_key.public_key()

    def create_package(self, payload: bytes) -> OTAPackage:
        """Hash and sign a payload, returning a signed package."""
        digest = hashlib.sha256(payload).hexdigest()
        signature = self._private_key.sign(
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return OTAPackage(payload=payload, sha256=digest, signature=signature)


class OTAUpdateClient:
    """Verifies OTA packages using the server's public key (simulation)."""

    def __init__(self, public_key) -> None:
        self._public_key = public_key

    def verify(self, package: OTAPackage) -> Tuple[bool, str]:
        """Verify hash + signature before 'installing'.

        Returns:
            (accepted, reason). Rejects corrupted or forged packages.
        """
        # 1. Hash integrity.
        if hashlib.sha256(package.payload).hexdigest() != package.sha256:
            return False, "SHA-256 hash mismatch (package corrupted)"
        # 2. Signature authenticity.
        try:
            self._public_key.verify(
                package.signature,
                package.payload,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
        except InvalidSignature:
            return False, "invalid signature (package forged/unauthorized)"
        return True, "verified (safe to install)"