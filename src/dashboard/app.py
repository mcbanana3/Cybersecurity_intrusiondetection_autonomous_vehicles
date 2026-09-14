"""
AEV Cybersecurity Dashboard (Streamlit).

Visualizes the full pipeline: synthetic AV/EV data -> simulated attacks
-> AI intrusion detection -> affected asset -> TARA risk -> security
response -> charging/grid monitoring -> ML performance.

Run from the project root:
    streamlit run src/dashboard/app.py

NOTE: This is a research/education simulation. All data and attacks are
synthetic; the system does not provide production-grade security.
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

from src.pipeline import run_pipeline
from src.tara.risk_engine import RiskEngine
from src.utils.config_loader import load_config

# ----------------------------------------------------------------------
st.set_page_config(page_title="AEV Cybersecurity IDS & TARA",
                   layout="wide", page_icon="🔐")

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825",
               "High": "#ef6c00", "Critical": "#c62828"}


@st.cache_data(show_spinner="Running simulation pipeline...")
def _cached_pipeline(overrides_key: str):
    """Run the pipeline; cache keyed by the attack-toggle signature."""
    overrides = json.loads(overrides_key)
    result = run_pipeline(overrides)
    # Return plain, cacheable structures.
    return result


def _load_metrics():
    path = os.path.join("results", "ids_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ----------------------------------------------------------------------
# Sidebar: attack controls
# ----------------------------------------------------------------------
st.sidebar.title("Attack Simulation Control")
st.sidebar.caption("Toggle simulated attacks, then re-run the pipeline.")

attack_names = ["spoofing", "replay", "dos", "injection",
                "tampering", "charging_attack"]
toggles = {}
for atk in attack_names:
    toggles[atk] = st.sidebar.checkbox(atk, value=True)

overrides = {atk: {"enabled": bool(val)} for atk, val in toggles.items()}
overrides_key = json.dumps(overrides, sort_keys=True)

st.sidebar.markdown("---")
st.sidebar.info("All attacks are simulated against synthetic data only.")

# Run pipeline (cached).
result = _cached_pipeline(overrides_key)

# ----------------------------------------------------------------------
# Header + global status
# ----------------------------------------------------------------------
st.title("🔐 AI-Based Intrusion Detection & Threat Risk Assessment")
st.caption("Autonomous Electric Vehicle Systems — software simulation "
           "(research/education; not production-grade security).")

n_attack_windows = sum(1 for v in result.verdicts if v.is_attack)
total_windows = len(result.verdicts)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Windows analysed", total_windows)
col2.metric("Attack windows detected", n_attack_windows)
status = "UNDER ATTACK" if n_attack_windows > 0 else "SECURE"
col3.metric("System status", status)
col4.metric("Charging samples flagged",
            f"{result.charging_flagged}/{result.charging_total}")

# Highest current risk across detected windows.
detected = [v for v in result.verdicts if v.is_attack and v.risk_score]
if detected:
    top = max(detected, key=lambda v: v.risk_score)
    color = RISK_COLORS.get(top.risk_level, "#555")
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{color};"
        f"color:white;font-size:18px'><b>Top threat:</b> "
        f"{top.attack_type} on {top.affected_asset} — "
        f"risk {top.risk_score} ({top.risk_level})</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_over, tab_can, tab_ai, tab_tara, tab_chg, tab_ml = st.tabs(
    ["Vehicle Status", "CAN Traffic", "AI Detection",
     "TARA & Risk", "Charging / Grid", "ML Performance"]
)

# ---- Tab 1: Vehicle status ----
with tab_over:
    st.subheader("Synthetic vehicle signals")
    sig = result.signals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig["timestamp"], y=sig["speed_kmh"],
                             name="Speed (km/h)"))
    fig.add_trace(go.Scatter(x=sig["timestamp"], y=sig["battery_soc"],
                             name="Battery SOC (%)"))
    fig.add_trace(go.Scatter(x=sig["timestamp"], y=sig["battery_temp_c"],
                             name="Battery Temp (°C)"))
    fig.update_layout(height=380, xaxis_title="time (s)",
                      legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Charging state over time**")
        st.area_chart(sig.set_index("timestamp")[["charging_state"]])
    with c2:
        st.markdown("**Pack current (+ discharge / − charge)**")
        st.line_chart(sig.set_index("timestamp")[["pack_current_a"]])

# ---- Tab 2: CAN traffic ----
with tab_can:
    st.subheader("Simulated CAN traffic")
    can = result.can_attacked

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frames per arbitration ID**")
        by_id = can.groupby("id_hex").size().reset_index(name="frames")
        st.plotly_chart(px.bar(by_id, x="id_hex", y="frames", height=340),
                        use_container_width=True)
    with c2:
        st.markdown("**Frame rate over time (attack windows spike)**")
        can = can.copy()
        can["sec"] = can["timestamp"].astype(int)
        rate = can.groupby("sec").size().reset_index(name="frames_per_s")
        atk_by_sec = (can[can["label"] == "attack"]
                      .groupby("sec").size().reset_index(name="attack_frames"))
        merged = rate.merge(atk_by_sec, on="sec", how="left").fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["sec"], y=merged["frames_per_s"],
                                 name="all frames/s"))
        fig.add_trace(go.Bar(x=merged["sec"], y=merged["attack_frames"],
                             name="attack frames/s", marker_color="#c62828"))
        fig.update_layout(height=340, xaxis_title="time (s)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sample frames**")
    show_cols = ["timestamp", "id_hex", "message_name", "label",
                 "attack_type"] + [f"b{i}" for i in range(8)]
    st.dataframe(can[show_cols].head(200), height=260, use_container_width=True)

# ---- Tab 3: AI detection ----
with tab_ai:
    st.subheader("AI intrusion detection (per window)")

    vdf = pd.DataFrame([{
        "start_s": v.start_s, "end_s": v.end_s, "true_label": v.true_label,
        "is_attack": v.is_attack, "attack_type": v.attack_type,
        "confidence": v.confidence, "is_anomaly": v.is_anomaly,
        "affected_asset": v.affected_asset, "risk_score": v.risk_score,
        "risk_level": v.risk_level,
    } for v in result.verdicts])

    # Detection card for the first detected attack.
    det = [v for v in result.verdicts if v.is_attack]
    if det:
        v = det[0]
        st.markdown("#### Live detection")
        st.code(
            f"Attack detected  : YES\n"
            f"Attack type      : {v.attack_type}\n"
            f"Confidence       : {v.confidence*100:.0f}%\n"
            f"Affected asset   : {v.affected_asset}\n"
            f"Severity         : {v.risk_level}\n"
            f"Risk score       : {v.risk_score}\n"
            f"Recommended action: {v.recommended_action}",
            language="text",
        )
    else:
        st.success("No attacks detected in the current run.")

    st.markdown("**Detected attack types (count)**")
    atk_only = vdf[vdf["is_attack"]]
    if not atk_only.empty:
        counts = atk_only["attack_type"].value_counts().reset_index()
        counts.columns = ["attack_type", "count"]
        st.plotly_chart(px.bar(counts, x="attack_type", y="count", height=320),
                        use_container_width=True)

    st.markdown("**All windows**")
    st.dataframe(vdf, height=320, use_container_width=True)

# ---- Tab 4: TARA & risk ----
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

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Prioritized risk table**")
        st.dataframe(tdf, height=300, use_container_width=True)
    with c2:
        st.markdown("**Risk score by threat**")
        fig = px.bar(tdf, x="risk_score", y="attack_type", orientation="h",
                     color="risk_level", color_discrete_map=RISK_COLORS,
                     height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Asset × risk heatmap**")
    heat = tdf.pivot_table(index="asset", columns="attack_type",
                           values="risk_score", aggfunc="max").fillna(0)
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto",
                              color_continuous_scale="Reds", height=320),
                    use_container_width=True)

# ---- Tab 5: Charging / grid ----
with tab_chg:
    st.subheader("EV → Charging Station → Grid")
    cn, ca = result.charging_normal, result.charging_attacked

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Grid frequency (Hz): normal vs attacked**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cn["t_s"], y=cn["grid_frequency_hz"],
                                 name="normal"))
        fig.add_trace(go.Scatter(x=ca["t_s"], y=ca["grid_frequency_hz"],
                                 name="attacked", line=dict(color="#c62828")))
        fig.add_hrect(y0=49.0, y1=51.0, fillcolor="green", opacity=0.08,
                      line_width=0, annotation_text="plausible band")
        fig.update_layout(height=340, xaxis_title="time (s)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Charging current (A): normal vs attacked**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cn["t_s"], y=cn["current_a"], name="normal"))
        fig.add_trace(go.Scatter(x=ca["t_s"], y=ca["current_a"],
                                 name="attacked", line=dict(color="#c62828")))
        fig.update_layout(height=340, xaxis_title="time (s)")
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Charging samples flagged as attack",
              f"{result.charging_flagged}/{result.charging_total}")

# ---- Tab 6: ML performance ----
with tab_ml:
    st.subheader("ML model performance (from trained models)")
    metrics = _load_metrics()
    if metrics is None:
        st.warning("No metrics found. Run `python run_phase5.py` to train "
                   "models and generate results/ids_metrics.json.")
    else:
        b = metrics["binary"]["test"]
        m = metrics["multiclass"]["test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Binary accuracy", f"{b['accuracy']:.3f}")
        c2.metric("Binary F1 (macro)", f"{b['f1_macro']:.3f}")
        c3.metric("Multi-class accuracy", f"{m['accuracy']:.3f}")

        # Confusion matrix (multi-class).
        st.markdown("**Multi-class confusion matrix (test)**")
        cm = m["confusion_matrix"]
        names = m.get("class_names", [str(i) for i in m["labels"]])
        fig = px.imshow(cm, x=names, y=names, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"), height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Per-class F1.
        pcf1 = metrics["multiclass"].get("test_per_class_f1", {})
        if pcf1:
            st.markdown("**Per-class F1 (test)**")
            f1df = pd.DataFrame(
                {"class": list(pcf1.keys()), "f1": list(pcf1.values())})
            st.plotly_chart(px.bar(f1df, x="class", y="f1", height=320,
                                   range_y=[0, 1]),
                            use_container_width=True)

st.markdown("---")
st.caption("Simulated environment. Attacks are synthetic and for research "
           "only. Not a production automotive security product.")