"""
TARA risk engine.

Computes risk for each threat and bridges the IDS to TARA:

    Risk = aggregated_impact (1..5) * feasibility (1..5)   -> 1..25
    Risk level (Low/Medium/High/Critical) via configured thresholds.

When the IDS actively DETECTS an attack, feasibility is boosted (the
observed likelihood is higher than the baseline assumption), which is a
faithful, simple reflection of ISO/SAE 21434 risk determination.

Public entry points:
    assess_attack_type(attack_type, detected) -> RiskAssessment
    assess(ids_result)                        -> RiskAssessment
    static_tara_table()                       -> list of assessments
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd

from src.tara.assets import AssetRegistry
from src.tara.scenarios import ScenarioCatalog, ThreatScenario
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAssessment:
    """The scored risk result for one threat/detection."""

    attack_type: str
    detected: bool
    asset: str
    asset_criticality: str
    threat_scenario: str
    damage_scenario: str
    impact_scores: Dict[str, int]     # S-F-O-P numeric
    aggregated_impact: float          # 1..5
    feasibility: int                  # 1..5 (post-detection boost)
    risk_score: float                 # impact * feasibility (rounded)
    risk_level: str                   # Low / Medium / High / Critical
    severity: str                     # human label derived from risk_level
    recommended_action: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskEngine:
    """Calculates risk and prioritizes threats (ISO/SAE 21434-aligned)."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.assets = AssetRegistry(self.config)
        self.catalog = ScenarioCatalog(self.config)

        tara = self.config["tara"]
        self.boost = int(tara["detected_feasibility_boost"])
        # Ordered thresholds low->high.
        self.risk_levels: Dict[str, int] = {
            k: int(v) for k, v in tara["risk_levels"].items()
        }
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------
    def assess_attack_type(
        self, attack_type: str, detected: bool = True
    ) -> RiskAssessment | None:
        """Score a single attack type.

        Args:
            attack_type: One of the catalogued attack types.
            detected: Whether the IDS actively detected it (raises
                feasibility toward the observed likelihood).

        Returns:
            A RiskAssessment, or None if the attack type is unknown
            (e.g. 'normal').
        """
        scenario = self.catalog.get(attack_type)
        if scenario is None:
            return None

        asset = self.assets.get(scenario.asset)
        impact_scores = self.catalog.impact_scores(scenario)
        agg_impact = self.catalog.aggregate_impact(scenario)

        feasibility = scenario.base_feasibility
        if detected:
            feasibility = min(5, feasibility + self.boost)

        risk_score = round(agg_impact * feasibility, 2)
        level = self._risk_level(risk_score)

        return RiskAssessment(
            attack_type=attack_type,
            detected=detected,
            asset=asset.name,
            asset_criticality=asset.criticality,
            threat_scenario=scenario.threat_scenario,
            damage_scenario=scenario.damage_scenario,
            impact_scores=impact_scores,
            aggregated_impact=round(agg_impact, 2),
            feasibility=feasibility,
            risk_score=risk_score,
            risk_level=level,
            severity=level,  # severity label mirrors the risk level
            recommended_action=scenario.recommended_action,
        )

    def assess(self, ids_result: Any) -> RiskAssessment | None:
        """Bridge an IDSResult (Phase 5) into a RiskAssessment.

        Args:
            ids_result: An object with .is_attack and .attack_type
                (duck-typed so we avoid a hard import dependency).

        Returns:
            A RiskAssessment if an attack was detected, else None.
        """
        if not getattr(ids_result, "is_attack", False):
            return None
        attack_type = getattr(ids_result, "attack_type", "unknown")
        return self.assess_attack_type(attack_type, detected=True)

    # ------------------------------------------------------------------
    # Prioritization + static table
    # ------------------------------------------------------------------
    def static_tara_table(self, detected: bool = False) -> List[RiskAssessment]:
        """Return risk assessments for ALL catalogued threats.

        Args:
            detected: If True, compute the 'active detection' risk;
                if False, the baseline (pre-detection) risk.

        Returns:
            Assessments sorted by risk_score descending (prioritized).
        """
        rows = []
        for attack_type in self.catalog.all().keys():
            ra = self.assess_attack_type(attack_type, detected=detected)
            if ra is not None:
                rows.append(ra)
        rows.sort(key=lambda r: r.risk_score, reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _risk_level(self, score: float) -> str:
        """Map a numeric risk score to a discrete level via thresholds."""
        # thresholds: Low<=6, Medium<=12, High<=18, else Critical
        if score <= self.risk_levels["Low"]:
            return "Low"
        if score <= self.risk_levels["Medium"]:
            return "Medium"
        if score <= self.risk_levels["High"]:
            return "High"
        return "Critical"

    def export(
        self,
        assessments: List[RiskAssessment],
        json_file: str = "tara_report.json",
        csv_file: str = "tara_table.csv",
    ) -> tuple[str, str]:
        """Save a list of assessments to JSON and CSV in results/."""
        json_path = os.path.join(self.results_dir, json_file)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump([a.as_dict() for a in assessments], fh, indent=2)

        # Flatten S-F-O-P into columns for the CSV.
        flat = []
        for a in assessments:
            d = a.as_dict()
            imp = d.pop("impact_scores")
            for cat, val in imp.items():
                d[f"impact_{cat}"] = val
            flat.append(d)
        csv_path = os.path.join(self.results_dir, csv_file)
        pd.DataFrame(flat).to_csv(csv_path, index=False)

        logger.info("Exported TARA report -> %s and %s", json_path, csv_path)
        return json_path, csv_path