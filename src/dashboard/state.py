"""
Control-Center UI state and orchestration (UI layer only).

This module holds NO detection/risk/security logic of its own. It only:
    * keeps interactive state in st.session_state (current attack config,
      the control-center phase, the last PipelineResult, an event log),
    * translates UI choices (attack type / target / intensity) into the
      config-override shape that the EXISTING run_pipeline() understands,
    * calls the EXISTING backend (run_pipeline, IntegratedIDS) and stores
      the real results for the widgets to render.

All attacks are simulated against synthetic data (as in every phase).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import streamlit as st

from src.features.dataset import _NON_FEATURE
from src.integration.integrated_ids import IntegratedIDS, IntegratedVerdict
from src.pipeline import PipelineResult, run_pipeline
from src.utils.config_loader import load_config

# The six supported attack types (must match config.yaml keys).
ATTACK_TYPES = ["spoofing", "replay", "dos", "injection",
                "tampering", "charging_attack"]

# Human labels for the UI.
ATTACK_LABELS = {
    "spoofing": "Spoofing (forged sensor/speed)",
    "replay": "Replay (stale messages)",
    "dos": "DoS / Jamming (bus flood)",
    "injection": "Message Injection (rogue ID)",
    "tampering": "Tampering (modified values)",
    "charging_attack": "Charging / Grid attack",
}

# Which asset / target each attack primarily hits (for the target selector
# default). These mirror the assets configured in the TARA engine.
ATTACK_TARGETS = {
    "spoofing": "Powertrain ECU",
    "replay": "Steering/ADAS ECU",
    "dos": "CAN Bus",
    "injection": "CAN Bus",
    "tampering": "Battery Management (BMS)",
    "charging_attack": "Charging Controller",
}

# Control-center lifecycle phases.
PHASE_NORMAL = "NORMAL"
PHASE_UNDER_ATTACK = "UNDER_ATTACK"
PHASE_DETECTED = "DETECTED"
PHASE_TREATED = "TREATED"
PHASE_RECOVERED = "RECOVERED"


@dataclass
class ControlState:
    """Serializable snapshot of the control center (stored in session)."""

    phase: str = PHASE_NORMAL
    active_attack: str | None = None
    intensity: str = "Medium"
    target: str | None = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Cached verdict for the currently launched attack.
    verdict: Dict[str, Any] | None = None
    treated: bool = False


# ----------------------------------------------------------------------
# Session-state helpers
# ----------------------------------------------------------------------
def init_state() -> None:
    """Initialise session_state keys once per session."""
    if "control" not in st.session_state:
        st.session_state.control = ControlState()
    if "pipeline_result" not in st.session_state:
        # Baseline (no attacks) so the first render shows a normal system.
        st.session_state.pipeline_result = None


def get_state() -> ControlState:
    """Return the current ControlState."""
    init_state()
    return st.session_state.control


def log_event(message: str, level: str = "info") -> None:
    """Append a timestamped event to the live timeline."""
    ctrl = get_state()
    ctrl.events.append({
        "time": time.strftime("%H:%M:%S"),
        "message": message,
        "level": level,
    })


# ----------------------------------------------------------------------
# Intensity -> concrete attack parameters
# ----------------------------------------------------------------------
def _intensity_overrides(attack_type: str, intensity: str) -> Dict[str, Any]:
    """Map a qualitative intensity to real attack parameters.

    Only parameters that the existing attack engine already understands
    are set here; everything else keeps its config default.
    """
    # Multipliers: higher intensity => more frames / larger deviation.
    scale = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(intensity, 1.0)

    ov: Dict[str, Any] = {"enabled": True}

    if attack_type == "spoofing":
        # Faster injection + more extreme forged speed at higher intensity.
        ov["inject_rate_ms"] = max(2.0, 10.0 / scale)
        ov["forged_speed_kmh"] = 160.0 + 40.0 * scale
    elif attack_type == "dos":
        ov["flood_rate_ms"] = max(0.25, 1.0 / scale)
    elif attack_type == "injection":
        ov["inject_rate_ms"] = max(3.0, 15.0 / scale)
    elif attack_type == "tampering":
        ov["fraction"] = min(1.0, 0.25 * scale + 0.25)
    elif attack_type == "charging_attack":
        # Lower (more dangerous) frequency + bigger oscillation.
        ov["false_grid_freq_hz"] = 48.0 - 1.5 * scale
        ov["oscillation_hz"] = 1.0 * scale + 1.0
    # replay uses its configured windows; intensity left as default.

    return ov


def build_overrides(active_attack: str | None, intensity: str) -> Dict[str, Any]:
    """Build the run_pipeline() override dict for a single active attack.

    All other attacks are disabled so the console reflects exactly the
    one attack the operator launched.
    """
    overrides: Dict[str, Any] = {a: {"enabled": False} for a in ATTACK_TYPES}
    if active_attack is not None:
        overrides[active_attack] = _intensity_overrides(active_attack, intensity)
    return overrides


# ----------------------------------------------------------------------
# Backend calls (reuse existing pipeline + IntegratedIDS)
# ----------------------------------------------------------------------
def run_backend(active_attack: str | None, intensity: str) -> PipelineResult:
    """Run the EXISTING pipeline with the chosen attack and cache result."""
    overrides = build_overrides(active_attack, intensity)
    result = run_pipeline(overrides)
    st.session_state.pipeline_result = result
    return result


def summarize_detection(result: PipelineResult,
                        active_attack: str | None) -> IntegratedVerdict | None:
    """Return the strongest IntegratedVerdict for the active attack.

    Uses the EXISTING IntegratedIDS on the real feature windows. We pick,
    among windows whose true label matches the launched attack, the one
    the model is most confident is an attack -- this is the natural
    "representative detection" for the console (still 100% real output).
    """
    if active_attack is None:
        return None

    features = result.features
    feature_cols = [c for c in features.columns if c not in _NON_FEATURE]
    sub = features[features["label_multi"] == active_attack]
    if sub.empty:
        return None

    integrated = IntegratedIDS(load_config(), "models")

    best: IntegratedVerdict | None = None
    for _, row in sub.iterrows():
        x = row[feature_cols].to_numpy(dtype=np.float32)
        v = integrated.analyze(x)
        if v.is_attack:
            if best is None or v.confidence > best.confidence:
                best = v
    # If the model never flagged it, still return the highest-confidence
    # verdict so the UI shows the real (possibly missed) result.
    if best is None:
        row = sub.iloc[0]
        best = integrated.analyze(row[feature_cols].to_numpy(dtype=np.float32))
    return best


def launch_attack(attack_type: str, intensity: str, target: str) -> None:
    """Handle the 'Launch Attack' button: run backend + record state."""
    ctrl = get_state()
    ctrl.active_attack = attack_type
    ctrl.intensity = intensity
    ctrl.target = target
    ctrl.treated = False
    ctrl.phase = PHASE_UNDER_ATTACK
    log_event(f"Attack launched: {ATTACK_LABELS[attack_type]} "
              f"(intensity={intensity}, target={target})", "attack")

    result = run_backend(attack_type, intensity)
    verdict = summarize_detection(result, attack_type)

    if verdict is not None and verdict.is_attack:
        ctrl.phase = PHASE_DETECTED
        ctrl.verdict = verdict.as_dict()
        log_event("Anomaly detected by AI IDS", "detect")
        log_event(f"Classified as '{verdict.attack_type}' "
                  f"(confidence {verdict.confidence*100:.0f}%)", "detect")
        log_event(f"Affected asset identified: {verdict.affected_asset}",
                  "detect")
        log_event(f"Risk calculated: {verdict.risk_score} "
                  f"({verdict.severity})", "risk")
    else:
        ctrl.verdict = verdict.as_dict() if verdict else None
        log_event("Attack launched but not flagged by IDS in sampled "
                  "windows (real model output).", "warn")


def apply_treatment() -> None:
    """Handle 'Apply Treatment': record the existing security response."""
    ctrl = get_state()
    if not ctrl.verdict:
        log_event("No active detection to treat.", "warn")
        return
    ctrl.treated = True
    ctrl.phase = PHASE_TREATED
    action = ctrl.verdict.get("recommended_action") or "Apply security control"
    control = ctrl.verdict.get("security_control") or "Security control"
    log_event(f"Security control applied: {control}", "treat")
    log_event(f"Action: {action}", "treat")
    log_event("Threat contained (malicious message blocked, simulated).",
              "treat")
    ctrl.phase = PHASE_RECOVERED
    log_event("System recovered to secure state.", "recover")


def reset_console() -> None:
    """Handle 'Reset': clear attack state and return to NORMAL."""
    st.session_state.control = ControlState()
    st.session_state.pipeline_result = None
    log_event("Console reset to NORMAL.", "info")