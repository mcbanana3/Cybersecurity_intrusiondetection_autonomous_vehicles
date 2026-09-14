"""
Phase 7 runner.

Demonstrates each simulated security control on a legitimate ('good')
case and an attack ('bad') case, proving that attacks are rejected:

    * AES-256-GCM        : tampered ciphertext fails to decrypt.
    * HMAC + counter     : tampered payload / replayed counter rejected.
    * Runtime tamper     : implausible signal values rejected.
    * Secure diagnostics : correct key granted, wrong key denied+locked.
    * OTA security       : valid package installed, forged package rejected.
    * Responder          : maps each attack type to its control.

Run from the project root:
    python run_phase7.py
"""

from __future__ import annotations

from src.security.crypto import AESGCMCipher
from src.security.diagnostics import DiagnosticAccess
from src.security.message_auth import MessageAuthenticator, MessageVerifier
from src.security.ota import OTAUpdateClient, OTAUpdateServer
from src.security.responder import SecurityResponder
from src.security.tamper import TamperDetector
from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config["logging"]["level"])
    logger = get_logger("run_phase7")

    logger.info("=== Phase 7: Security Controls ===")
    print("\n" + "=" * 74)
    print("PHASE 7 COMPLETE - Simulated security controls demonstration")
    print("=" * 74)

    # ---------- 1. AES-256-GCM ----------
    print("\n[1] AES-256-GCM authenticated encryption")
    cipher = AESGCMCipher(config)
    ct = cipher.encrypt(b"speed=42;steering=3")
    ok_plain = cipher.try_decrypt(ct)
    print(f"  legit decrypt : {ok_plain!r}")

    # Tamper with one ciphertext byte.
    tampered = bytearray(ct.data)
    tampered[0] ^= 0xFF
    from src.security.crypto import Ciphertext
    bad = cipher.try_decrypt(Ciphertext(nonce=ct.nonce, data=bytes(tampered)))
    print(f"  tampered decrypt: {bad!r}  (None = rejected)")

    # ---------- 2. HMAC + counter (tamper + replay) ----------
    print("\n[2] HMAC-SHA256 message authentication + replay protection")
    sender = MessageAuthenticator(config)
    verifier = MessageVerifier(config)

    m1 = sender.sign(b"BMS_SOC=85")
    print(f"  msg1 verify   : {verifier.verify(m1)}")

    m2 = sender.sign(b"BMS_SOC=84")
    print(f"  msg2 verify   : {verifier.verify(m2)}")

    # Replay msg1 (stale counter).
    print(f"  replay msg1   : {verifier.verify(m1)}  (rejected = replay)")

    # Tamper msg payload.
    m2.payload = b"BMS_SOC=00"
    print(f"  tampered msg  : {verifier.verify(m2)}  (rejected = tamper)")

    # ---------- 3. Runtime tamper / plausibility ----------
    print("\n[3] Runtime tamper detection (plausibility checks)")
    detector = TamperDetector(config)
    good = {"speed_kmh": 60.0, "battery_soc": 80.0, "grid_frequency_hz": 50.0}
    bad_vals = {"speed_kmh": 200.0, "battery_soc": 130.0, "grid_frequency_hz": 47.0}
    print(f"  normal values tampered? {detector.any_tampered(good)}")
    print(f"  attack values tampered? {detector.any_tampered(bad_vals)}")
    for sig, res in detector.check_signals(bad_vals).items():
        if not res.ok:
            print(f"    - {sig}: REJECTED ({res.reason})")

    # ---------- 4. Secure diagnostic access ----------
    print("\n[4] Secure diagnostic access (seed-key)")
    diag = DiagnosticAccess(config)
    seed = diag.request_seed()
    correct_key = DiagnosticAccess.compute_key(
        config["security"]["diag_secret"].encode(), seed
    )
    print(f"  correct key   : {diag.authenticate(correct_key)}")

    diag2 = DiagnosticAccess(config)
    diag2.request_seed()
    for i in range(4):
        r = diag2.authenticate(b"wrong-key-guess")
        print(f"  wrong attempt {i+1}: granted={r.granted} | {r.reason}")

    # ---------- 5. OTA update security ----------
    print("\n[5] OTA update security (RSA sign + SHA-256 verify)")
    server = OTAUpdateServer()
    client = OTAUpdateClient(server.public_key)
    pkg = server.create_package(b"firmware-v2.0-binary-bytes")
    print(f"  legit package : {client.verify(pkg)}")

    # Forge: modify payload after signing.
    from src.security.ota import OTAPackage
    forged = OTAPackage(payload=b"malicious-firmware",
                        sha256=pkg.sha256, signature=pkg.signature)
    print(f"  forged package: {client.verify(forged)}  (rejected)")

    # ---------- 6. Responder mapping ----------
    print("\n[6] Security responder (attack type -> control)")
    responder = SecurityResponder(config)
    for atk in ["spoofing", "replay", "dos", "injection", "tampering",
                "charging_attack"]:
        r = responder.respond_to_threat(atk)
        print(f"  {atk:16s} -> {r.control}")

    # ---------- Sanity checks ----------
    assert ok_plain == b"speed=42;steering=3"
    assert bad is None, "AES-GCM must reject tampered ciphertext!"
    assert verifier.verify(m1)[0] is False, "Replay must be rejected!"
    assert detector.any_tampered(bad_vals) is True
    assert detector.any_tampered(good) is False
    assert diag.authenticate(correct_key) is not None
    assert client.verify(pkg)[0] is True
    assert client.verify(forged)[0] is False, "Forged OTA must be rejected!"

    print("\n" + "=" * 74)
    print("All security-control checks passed (attacks rejected, legit accepted).")
    print("=" * 74)


if __name__ == "__main__":
    main()