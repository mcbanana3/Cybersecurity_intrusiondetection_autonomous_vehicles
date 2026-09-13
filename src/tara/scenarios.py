"""
Threat and damage scenario catalog + impact aggregation.

Maps each attack type to its threat scenario, damage scenario, affected
asset, S-F-O-P impact ratings and recommended action, all defined in
config.yaml under `tara.threats`. Also aggregates the four impact
categories into a single impact score using configured weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThreatScenario:
    """A catalogued threat for one attack type.

    Attributes:
        attack_type: e.g. "spoofing".
        threat_scenario: What the attacker achieves.
        damage_scenario: The resulting harm.
        asset: Primary affected asset name.
        impact: S-F-O-P labels, e.g. {"safety": "Severe", ...}.
        base_feasibility: Baseline feasibility 1..5.
        recommended_action: Security response to recommend.
    """

    attack_type: str
    threat_scenario: str
    damage_scenario: str
    asset: str
    impact: Dict[str, str]
    base_feasibility: int
    recommended_action: str


class ScenarioCatalog:
    """Loads threat scenarios and computes impact scores."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        tara = self.config["tara"]
        self.severity_scale: Dict[str, int] = {
            k: int(v) for k, v in tara["severity_scale"].items()
        }
        self.impact_weights: Dict[str, float] = {
            k: float(v) for k, v in tara["impact_weights"].items()
        }

        self._threats: Dict[str, ThreatScenario] = {}
        for atk, entry in tara["threats"].items():
            self._threats[atk] = ThreatScenario(
                attack_type=atk,
                threat_scenario=str(entry["threat_scenario"]),
                damage_scenario=str(entry["damage_scenario"]),
                asset=str(entry["asset"]),
                impact={k: str(v) for k, v in entry["impact"].items()},
                base_feasibility=int(entry["base_feasibility"]),
                recommended_action=str(entry["recommended_action"]),
            )
        logger.info("ScenarioCatalog loaded %d threat scenarios",
                    len(self._threats))

    def get(self, attack_type: str) -> ThreatScenario | None:
        """Return the scenario for an attack type, or None if unknown."""
        return self._threats.get(attack_type)

    def all(self) -> Dict[str, ThreatScenario]:
        """Return all catalogued scenarios keyed by attack type."""
        return dict(self._threats)

    # ------------------------------------------------------------------
    def impact_scores(self, scenario: ThreatScenario) -> Dict[str, int]:
        """Convert a scenario's S-F-O-P labels to numeric scores 1..5."""
        return {
            cat: self.severity_scale.get(label, 1)
            for cat, label in scenario.impact.items()
        }

    def aggregate_impact(self, scenario: ThreatScenario) -> float:
        """Combine S-F-O-P scores into one weighted impact value (1..5).

        Uses the configured impact_weights. Safety is weighted highest,
        reflecting its importance in automotive risk assessment.
        """
        scores = self.impact_scores(scenario)
        total = 0.0
        weight_sum = 0.0
        for cat, score in scores.items():
            w = self.impact_weights.get(cat, 0.0)
            total += w * score
            weight_sum += w
        if weight_sum == 0:
            return float(max(scores.values()) if scores else 1.0)
        return total / weight_sum