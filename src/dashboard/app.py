"""
AEV Cybersecurity Control Center (Streamlit).

Interactive console over the EXISTING backend (attack engine, IntegratedIDS,
TARA/risk engine, security modules). This file is UI only -- it calls
run_pipeline() and IntegratedIDS via src/dashboard/state.py and never
duplicates or hard-codes detection/risk/security logic.

Run from the project root:
    streamlit run src/dashboard/app.py

NOTE: research/education simulation. All data and attacks are synthetic;
this is not production-grade automotive security.
"""

from __future__ import annotations

import json
import os
import sys

# Make 'src' importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import state as ctl
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config

# ----------------------------------------------------------------------
st.set_page_config(page_title="AEV Cybersecurity Control Center",
                   layout="wide", page_icon="🔐")

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825",
               "High": "#ef6c00", "Critical": "#c62828"}
PHASE_BADGE = {
    ctl.PHASE_NORMAL: ("🟢 NORMAL", "#2e7d32"),
    ctl.PHASE_UNDER_ATTACK: ("🔴 UNDER ATTACK", "#c62828"),
    ctl.PHASE_DETECTED: ("🟠 ATTACK DETECTED", "#ef6c00"),
    ctl.PHASE_TREATED: ("🛡️ TREATMENT APPLIED", "#1565c0"),
    ctl.PHASE_RECOVERED: ("🟢 RECOVERED", "#2e7d32"),
}


def _load_metrics():
    path = os.path.join("results", "ids_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# Initialise interactive state.
ctl.init_state()
control = ctl.get_state()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
label, color = PHASE_BADGE.get(control.phase, ("🟢 NORMAL", "#2e7d32"))
st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
    f"<h1 style='margin:0'>🔐 AEV CYBERSECURITY CONTROL CENTER</h1>"
    f"<div style='padding:8px 16px;border-radius:8px;background:{color};"
    f"color:white;font-size:18px;font-weight:bold'>{label}</div></div>",
    unsafe_allow_html=True,
)
st.caption("Interactive intrusion-detection & risk console for a simulated "
           "Autonomous EV. All attacks are synthetic (research/education).")

# ----------------------------------------------------------------------
# Sidebar: Attack Simulator controls (Step 1)
# ----------------------------------------------------------------------
st.sidebar.title("⚔️ Attack Simulator")

attack_type = st.sidebar.selectbox(
    "Attack type",
    ctl.ATTACK_TYPES,
    format_func=lambda a: ctl.ATTACK_LABELS[a],
)

# Target selector: default to the attack's natural asset, allow override.
asset_names = [a["name"] for a in load_config()["assets"]]
default_target = ctl.ATTACK_TARGETS.get(attack_type, asset_names[0])
target = st.sidebar.selectbox(
    "Target asset",
    asset_names,
    index=asset_names.index(default_target) if default_target in asset_names else 0,
)

intensity = st.sidebar.select_slider(
    "Attack intensity",
    options=["Low", "Medium", "High"],
    value="Medium",
)

st.sidebar.markdown("---")

c1, c2 = st.sidebar.columns(2)
launch = c1.button("🚀 Launch Attack", use_container_width=True, type="primary")
treat = c2.button("🛡️ Apply Treatment", use_container_width=True)
reset = st.sidebar.button("🔄 Reset", use_container_width=True)

if launch:
    ctl.launch_attack(attack_type, intensity, target)
if treat:
    ctl.apply_treatment()
if reset:
    ctl.reset_console()

# Refresh local reference after actions.
control = ctl.get_state()

st.sidebar.markdown("---")
st.sidebar.info("All attacks are simulated against synthetic data only. "
                "Not a production security product.")

# ----------------------------------------------------------------------
# Top status metrics
# ----------------------------------------------------------------------
result = st.session_state.pipeline_result
verdict = control.verdict

m1, m2, m3, m4 = st.columns(4)
m1.metric("Console phase", control.phase.replace("_", " ").title())
m2.metric("Active attack",
          ctl.ATTACK_LABELS.get(control.active_attack, "—")
          if control.active_attack else "—")
m3.metric("Detected",
          "YES" if (verdict and verdict.get("is_attack")) else "NO")
m4.metric("Risk score",
          verdict.get("risk_score") if verdict and verdict.get("risk_score")
          else "—")

# ----------------------------------------------------------------------
# Tabs (existing views preserved; Attack Simulator added)
# ----------------------------------------------------------------------
(tab_live, tab_sim, tab_ai, tab_tara, tab_sec,
 tab_chg, tab_ml) = st.tabs(
    ["Live Monitor", "Attack Simulator", "AI Detection", "TARA / Risk",
     "Security", "Charging / Grid", "Model Performance"]
)

# ---- Live Monitor (placeholder in Step 1; animation added in Step 2) ----
with tab_live:
    st.subheader("Live Monitor")
    st.info("Vehicle animation and live attack visualization arrive in "
            "Step 2. Current console phase and events are shown below.")
    if control.events:
        st.markdown("**Recent events**")
        for ev in control.events[-8:][::-1]:
            st.write(f"`{ev['time']}` — {ev['message']}")
    else:
        st.write("No events yet. Launch an attack from the sidebar.")

# ---- Attack Simulator (Step 1 core) ----
with tab_sim:
    st.subheader("Attack Simulator")
    st.markdown(
        f"**Selected:** {ctl.ATTACK_LABELS[attack_type]}  \n"
        f"**Target:** {target}  \n"
        f"**Intensity:** {intensity}"
    )
    st.caption("Launch runs the real attack engine + pipeline; Apply Treatment "
               "uses the real security responder; Reset clears state.")

    if result is not None:
        can = result.can_attacked
        n_attack = int((can["label"] == "attack").sum())
        st.markdown("**Effect of the launched attack on the CAN trace**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Total CAN frames", len(can))
        cc2.metric("Attack frames", n_attack)
        cc3.metric("Attack windows detected",
                   sum(1 for v in result.verdicts if v.is_attack))

        can = can.copy()
        can["sec"] = can["timestamp"].astype(int)
        rate = can.groupby("sec").size().reset_index(name="frames_per_s")
        atk = (can[can["label"] == "attack"].groupby("sec").size()
               .reset_index(name="attack_frames"))
        merged = rate.merge(atk, on="sec", how="left").fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["sec"], y=merged["frames_per_s"],
                                 name="all frames/s"))
        fig.add_trace(go.Bar(x=merged["sec"], y=merged["attack_frames"],
                             name="attack frames/s", marker_color="#c62828"))
        fig.update_layout(height=340, xaxis_title="time (s)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Launch an attack to see its effect on the simulated system.")

# ---- AI Detection ----
with tab_ai:
    st.subheader("AI Intrusion Detection")
    if verdict and verdict.get("is_attack"):
        st.code(
            f"Attack detected  : YES\n"
            f"Attack type      : {verdict['attack_type']}\n"
            f"Confidence       : {verdict['confidence']*100:.0f}%\n"
            f"Affected asset   : {verdict['affected_asset']}\n"
            f"Detection status : {control.phase}",
            language="text",
        )
    elif verdict is not None:
        st.warning("Attack launched but IDS did not flag it in the sampled "
                   "windows (real model output).")
    else:
        st.write("No detection yet. Launch an attack from the sidebar.")

# ---- TARA / Risk ----
with tab_tara:
    st.subheader("Threat Analysis & Risk Assessment (ISO/SAE 21434-aligned)")
    engine = RiskEngine(load_config())
    table = engine.static_tara_table(detected=True)
    tdf = pd.DataFrame([{
        "attack_type": r.attack_type, "asset": r.asset,
        "impact": r.aggregated_impact, "feasibility": r.feasibility,
        "risk_score": r.risk_score, "risk_level": r.risk_level,
        "action": r.recommended_action,
    } for r in table])
    st.dataframe(tdf, height=280, use_container_width=True)
    fig = px.bar(tdf, x="risk_score", y="attack_type", orientation="h",
                 color="risk_level", color_discrete_map=RISK_COLORS, height=300)
    st.plotly_chart(fig, use_container_width=True)

# ---- Security ----
with tab_sec:
    st.subheader("Security Treatment")
    if control.treated and verdict:
        st.success("Treatment applied — threat contained (simulated).")
        st.code(
            f"Attack           : {verdict['attack_type']}\n"
            f"Security control : {verdict.get('security_control')}\n"
            f"Action           : {verdict.get('recommended_action')}\n"
            f"Outcome          : malicious message blocked (simulated)",
            language="text",
        )
    elif verdict and verdict.get("is_attack"):
        st.info("Attack detected. Click 'Apply Treatment' in the sidebar to "
                "run the existing security control.")
    else:
        st.write("No active threat.")

# ---- Charging / Grid ----
with tab_chg:
    st.subheader("EV → Charging Station → Grid")
    if result is not None:
        cn, ca = result.charging_normal, result.charging_attacked
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cn["t_s"], y=cn["grid_frequency_hz"],
                                 name="normal"))
        fig.add_trace(go.Scatter(x=ca["t_s"], y=ca["grid_frequency_hz"],
                                 name="attacked", line=dict(color="#c62828")))
        fig.add_hrect(y0=49.0, y1=51.0, fillcolor="green", opacity=0.08,
                      line_width=0)
        fig.update_layout(height=340, xaxis_title="time (s)",
                          title="Grid frequency (Hz)")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Charging samples flagged",
                  f"{result.charging_flagged}/{result.charging_total}")
    else:
        st.write("Launch an attack to populate charging/grid telemetry.")

# ---- Model Performance ----
with tab_ml:
    st.subheader("ML model performance (from trained models)")
    metrics = _load_metrics()
    if metrics is None:
        st.warning("No metrics found. Run `python run_phase5.py` first.")
    else:
        b = metrics["binary"]["test"]
        m = metrics["multiclass"]["test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Binary accuracy", f"{b['accuracy']:.3f}")
        c2.metric("Binary F1 (macro)", f"{b['f1_macro']:.3f}")
        c3.metric("Multi-class accuracy", f"{m['accuracy']:.3f}")
        cm = m["confusion_matrix"]
        names = m.get("class_names", [str(i) for i in m["labels"]])
        fig = px.imshow(cm, x=names, y=names, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"), height=420)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Simulated environment. Attacks are synthetic and for research "
           "only. Not a production automotive security product.")