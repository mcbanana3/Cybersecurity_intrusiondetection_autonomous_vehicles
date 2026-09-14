"""
Phase 7 tests: crypto, message auth, tamper, diagnostics, OTA, responder.

Run from the project root:
    python -m pytest -v
"""

from __future__ import annotations

from src.security.crypto import AESGCMCipher, Ciphertext
from src.security.diagnostics import DiagnosticAccess
from src.security.message_auth import MessageAuthenticator, MessageVerifier
from src.security.ota import OTAPackage, OTAUpdateClient, OTAUpdateServer
from src.security.responder import SecurityResponder
from src.security.tamper import TamperDetector
from src.utils.config_loader import load_config


def test_aes_roundtrip() -> None:
    cipher = AESGCMCipher(load_config())
    ct = cipher.encrypt(b"hello")
    assert cipher.decrypt(ct) == b"hello"


def test_aes_rejects_tamper() -> None:
    cipher = AESGCMCipher(load_config())
    ct = cipher.encrypt(b"hello")
    corrupted = bytearray(ct.data)
    corrupted[0] ^= 0xFF
    assert cipher.try_decrypt(Ciphertext(ct.nonce, bytes(corrupted))) is None


def test_hmac_accepts_valid() -> None:
    cfg = load_config()
    s, v = MessageAuthenticator(cfg), MessageVerifier(cfg)
    m = s.sign(b"payload")
    ok, _ = v.verify(m)
    assert ok


def test_hmac_rejects_replay() -> None:
    cfg = load_config()
    s, v = MessageAuthenticator(cfg), MessageVerifier(cfg)
    m1 = s.sign(b"a")
    s.sign(b"b")
    assert v.verify(m1)[0] is True
    # Replaying m1 now has a stale counter.
    assert v.verify(m1)[0] is False


def test_hmac_rejects_tamper() -> None:
    cfg = load_config()
    s, v = MessageAuthenticator(cfg), MessageVerifier(cfg)
    m = s.sign(b"payload")
    m.payload = b"changed"
    assert v.verify(m)[0] is False


def test_plausibility_flags_out_of_range() -> None:
    d = TamperDetector(load_config())
    assert d.any_tampered({"speed_kmh": 200.0}) is True
    assert d.any_tampered({"speed_kmh": 60.0}) is False


def test_diagnostics_grant_and_lockout() -> None:
    cfg = load_config()
    diag = DiagnosticAccess(cfg)
    seed = diag.request_seed()
    key = DiagnosticAccess.compute_key(cfg["security"]["diag_secret"].encode(), seed)
    assert diag.authenticate(key).granted is True

    diag2 = DiagnosticAccess(cfg)
    diag2.request_seed()
    last = None
    for _ in range(cfg["security"]["max_diag_attempts"]):
        last = diag2.authenticate(b"bad")
    assert last.granted is False


def test_ota_valid_and_forged() -> None:
    server = OTAUpdateServer()
    client = OTAUpdateClient(server.public_key)
    pkg = server.create_package(b"firmware")
    assert client.verify(pkg)[0] is True
    forged = OTAPackage(b"evil", pkg.sha256, pkg.signature)
    assert client.verify(forged)[0] is False


def test_responder_maps_all_attacks() -> None:
    r = SecurityResponder(load_config())
    for atk in ["spoofing", "replay", "dos", "injection", "tampering",
                "charging_attack"]:
        resp = r.respond_to_threat(atk)
        assert resp.control and resp.action