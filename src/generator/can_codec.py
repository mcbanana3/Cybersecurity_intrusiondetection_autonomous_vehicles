"""
CAN signal codec.

Encodes physical signal values (floats such as speed_kmh) into the raw
byte fields of a CAN payload, and decodes them back. This mirrors how a
real vehicle packs engineering values into CAN frames using a scale and
offset -- but is entirely software-simulated.

Encoding (per signal):
    raw_int = round((value - offset) / scale)
    raw_int is clamped to the unsigned range that fits in `length` bytes
    raw_int is written little-endian into payload[start : start+length]

Decoding (per signal):
    raw_int = int.from_bytes(payload[start:start+length], "little")
    value   = raw_int * scale + offset

A CAN payload is always 8 bytes (DLC 8) here; unused bytes are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

PAYLOAD_LENGTH = 8  # bytes per CAN frame (fixed for this simulation)


@dataclass
class SignalSpec:
    """Describes how one physical signal is packed into a CAN payload.

    Attributes:
        name: Signal name, matching a column produced in Phase 1.
        start: First byte index within the 8-byte payload.
        length: Number of bytes occupied (1-4).
        scale: Multiplier used when decoding (resolution per raw unit).
        offset: Value added when decoding.
    """

    name: str
    start: int
    length: int
    scale: float
    offset: float

    def max_raw(self) -> int:
        """Largest unsigned integer that fits in `length` bytes."""
        return (1 << (8 * self.length)) - 1


@dataclass
class MessageSpec:
    """Describes one periodic CAN message.

    Attributes:
        arbitration_id: CAN ID (integer).
        name: Message name.
        ecu: Sending ECU name.
        asset: Logical asset this message belongs to.
        cycle_ms: Nominal transmission period in milliseconds.
        signals: The signals packed into this message's payload.
    """

    arbitration_id: int
    name: str
    ecu: str
    asset: str
    cycle_ms: float
    signals: List[SignalSpec]

    @classmethod
    def from_config(cls, entry: Dict[str, Any]) -> "MessageSpec":
        """Build a MessageSpec from a config.yaml `can.messages` entry."""
        signals = [
            SignalSpec(
                name=s["name"],
                start=int(s["start"]),
                length=int(s["length"]),
                scale=float(s["scale"]),
                offset=float(s["offset"]),
            )
            for s in entry["signals"]
        ]
        return cls(
            arbitration_id=int(entry["arbitration_id"]),
            name=str(entry["name"]),
            ecu=str(entry["ecu"]),
            asset=str(entry["asset"]),
            cycle_ms=float(entry["cycle_ms"]),
            signals=signals,
        )


def encode_message(spec: MessageSpec, values: Dict[str, float]) -> List[int]:
    """Encode signal values into an 8-byte CAN payload.

    Args:
        spec: The message specification.
        values: Mapping of signal name -> physical value.

    Returns:
        A list of 8 integers (0-255), the CAN data bytes.
    """
    payload = bytearray(PAYLOAD_LENGTH)

    for sig in spec.signals:
        value = float(values.get(sig.name, 0.0))
        raw = round((value - sig.offset) / sig.scale)

        # Clamp to the field's representable range. Out-of-range values
        # (e.g. from tampering) saturate rather than crash -- still
        # producing an anomalous but valid byte pattern.
        raw = max(0, min(raw, sig.max_raw()))

        raw_bytes = int(raw).to_bytes(sig.length, byteorder="little")
        payload[sig.start : sig.start + sig.length] = raw_bytes

    return list(payload)


def decode_message(spec: MessageSpec, payload: List[int]) -> Dict[str, float]:
    """Decode an 8-byte CAN payload back into physical signal values.

    Args:
        spec: The message specification.
        payload: List of 8 integers (0-255).

    Returns:
        Mapping of signal name -> decoded physical value.
    """
    if len(payload) != PAYLOAD_LENGTH:
        raise ValueError(
            f"Payload must be {PAYLOAD_LENGTH} bytes, got {len(payload)}"
        )

    data = bytes(int(b) & 0xFF for b in payload)
    result: Dict[str, float] = {}

    for sig in spec.signals:
        raw = int.from_bytes(data[sig.start : sig.start + sig.length], "little")
        result[sig.name] = raw * sig.scale + sig.offset

    return result


def load_message_specs(config: Dict[str, Any]) -> List[MessageSpec]:
    """Load all CAN message specs from the project config.

    Args:
        config: The full configuration dictionary.

    Returns:
        A list of MessageSpec objects.
    """
    specs = [MessageSpec.from_config(e) for e in config["can"]["messages"]]
    logger.debug("Loaded %d CAN message specifications", len(specs))
    return specs