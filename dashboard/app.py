"""
Cyber-Resilient Autonomous Drone Navigation System - Interactive Dashboard
Streamlit + Plotly Implementation
Tabs:
- Tab 1: Attack Simulation (Deliverable 1)
- Tab 2: Resilience & Recovery (Deliverable 2)
- Tab 3: Comparative Analysis & Leaderboard (OFF vs ON)
"""

import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set repo path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyber_drone.simulation_engine import SimulationEngine


st.set_page_config(
    page_title="Cyber-Resilient Drone Navigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .state-banner {
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 16px;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .state-normal { background-color: #065f46; color: #34d399; border: 1px solid #059669; }
    .state-suspicious { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .state-attack { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    .state-safe { background-color: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
    .state-recovered { background-color: #064e3b; color: #10b981; border: 1px solid #059669; }
    .state-off { background-color: #451a03; color: #f97316; border: 1px solid #ea580c; }
</style>
""", unsafe_allow_html=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def run_simulation(scenario_id: str, defense_on: bool, seed: int):
    scenarios_dir = REPO_ROOT / "config" / "scenarios"
    sc_file = scenario_id if scenario_id.endswith(".yaml") else f"{scenario_id}.yaml"
    scenario_cfg = load_yaml(scenarios_dir / sc_file)
    sensors_cfg = load_yaml(REPO_ROOT / "config" / "sensors.yaml")
    estimator_cfg = load_yaml(REPO_ROOT / "config" / "estimator.yaml")
    detector_cfg = load_yaml(REPO_ROOT / "config" / "detector.yaml")
    resilience_cfg = load_yaml(REPO_ROOT / "config" / "resilience.yaml")
    planner_cfg = load_yaml(REPO_ROOT / "config" / "planner.yaml")

    engine = SimulationEngine(
        scenario_config=scenario_cfg,
        sensors_config=sensors_cfg,
        estimator_config=estimator_cfg,
        detector_config=detector_cfg,
        resilience_config=resilience_cfg,
        planner_config=planner_cfg,
        defense_on=defense_on,
        seed=seed
    )
    return engine.run(), scenario_cfg, planner_cfg


# Sidebar Controls
st.sidebar.title("🛡️ Cyber-Resilient Drone")
st.sidebar.caption("PX4/ROS 2 Autonomous Resilient Navigation System")

scenarios_dir = REPO_ROOT / "config" / "scenarios"
scenario_files = sorted([f for f in os.listdir(scenarios_dir) if f.endswith(".yaml")])
scenario_labels = {f: load_yaml(scenarios_dir / f).get("name", f) for f in scenario_files}

selected_scenario_file = st.sidebar.selectbox(
    "Select Scenario",
    scenario_files,
    format_func=lambda x: scenario_labels.get(x, x),
    index=1 if len(scenario_files) > 1 else 0
)

defense_mode = st.sidebar.radio(
    "Defense System",
    ["ON (Resilience & Recovery - D2)", "OFF (Unprotected Derailment - D1)"],
    index=0
)
is_defense_on = "ON" in defense_mode

seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

# Run simulation
with st.spinner("Simulating closed-loop mission..."):
    sim_data, scenario_cfg, planner_cfg = run_simulation(selected_scenario_file, is_defense_on, seed)

df_tel = sim_data["telemetry"]
df_trust = sim_data["trust"]
events = sim_data["events"]
metrics = sim_data["metrics"]
obstacles = planner_cfg.get("default_obstacles", [])

# Time slider for replay
max_time = float(df_tel["time"].max())
t_slice = st.sidebar.slider("Mission Time Scrub (seconds)", 0.0, max_time, max_time, step=0.5)
current_tel = df_tel[df_tel["time"] <= t_slice]
current_state = current_tel.iloc[-1]["state"] if not current_tel.empty else "NORMAL"

# Top Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Tab 1: Attack Simulation (D1)",
    "🛡️ Tab 2: Resilience & Recovery (D2)",
    "📊 Tab 3: Comparative Analysis (OFF vs ON)"
])

# =========================================================================
# TAB 1: ATTACK SIMULATION (D1)
# =========================================================================
with tab1:
    st.subheader("Deliverable 1: Simulated Cyberattack and Drone Trajectory Deviation")
    st.caption("Demonstrates normal navigation derailed by spoofing/disruption when defense is OFF vs contained when defense is ON.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"**Attack Type:** `{scenario_cfg.get('attack', {}).get('type', 'None')}`")
    with c2:
        st.markdown(f"**Attack Window:** `{scenario_cfg.get('attack', {}).get('start_time', 0)}s - {scenario_cfg.get('attack', {}).get('start_time', 0) + scenario_cfg.get('attack', {}).get('duration', 0)}s`")
    with c3:
        st.markdown(f"**Description:** {scenario_cfg.get('description', '')}")

    # 2D Arena Map: True Path vs Spoofed GPS
    fig_map = go.Figure()

    # Obstacles
    for obs in obstacles:
        theta = np.linspace(0, 2*np.pi, 50)
        ox = obs["x"] + obs.get("radius", 3.0) * np.cos(theta)
        oy = obs["y"] + obs.get("radius", 3.0) * np.sin(theta)
        fig_map.add_trace(go.Scatter(
            x=ox, y=oy, fill="toself", fillcolor="rgba(239, 68, 68, 0.25)",
            line=dict(color="#ef4444", width=1.5), name="Obstacle", showlegend=False
        ))

    # True Drone Path (Ground Truth)
    fig_map.add_trace(go.Scatter(
        x=current_tel["truth_x"], y=current_tel["truth_y"],
        mode="lines", line=dict(color="#ffffff", width=3), name="True Drone Path"
    ))

    # Raw / Spoofed GPS Feed
    gps_valid = current_tel.dropna(subset=["gps_x", "gps_y"])
    fig_map.add_trace(go.Scatter(
        x=gps_valid["gps_x"], y=gps_valid["gps_y"],
        mode="lines", line=dict(color="#ef4444", width=1.5, dash="dot"), name="GPS Reported (Spoofed)"
    ))

    # Start and Goal
    goal = scenario_cfg.get("mission", {}).get("goal_point", [60, 40])
    fig_map.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text", marker=dict(color="#10b981", size=12),
        text=["Start"], textposition="bottom center", name="Start"
    ))
    fig_map.add_trace(go.Scatter(
        x=[goal[0]], y=[goal[1]], mode="markers+text", marker=dict(color="#3b82f6", size=14, symbol="star"),
        text=["Goal"], textposition="top center", name="Goal"
    ))

    # Current drone position marker
    if not current_tel.empty:
        curr_row = current_tel.iloc[-1]
        fig_map.add_trace(go.Scatter(
            x=[curr_row["truth_x"]], y=[curr_row["truth_y"]],
            mode="markers", marker=dict(color="#f59e0b", size=14, symbol="cross"),
            name="Drone Current Pos"
        ))

    fig_map.update_layout(
        title="Arena 2D Map: Where the drone is vs what the sensor reports",
        xaxis=dict(title="X (meters)", range=[-10, 80]),
        yaxis=dict(title="Y (meters)", range=[-10, 60]),
        template="plotly_dark", height=450, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Discrepancy Plot: Where the drone thinks it is vs where it really is
    current_tel["pos_discrepancy"] = np.linalg.norm(
        current_tel[["truth_x", "truth_y"]].values - current_tel[["active_est_x", "active_est_y"]].values, axis=1
    )
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(
        x=current_tel["time"], y=current_tel["pos_discrepancy"],
        mode="lines", line=dict(color="#f97316", width=2), name="Position Error (m)"
    ))
    # Attack region shading
    atk_start = scenario_cfg.get("attack", {}).get("start_time", 30.0)
    atk_dur = scenario_cfg.get("attack", {}).get("duration", 20.0)
    fig_disc.add_vrect(
        x0=atk_start, x1=atk_start + atk_dur,
        fillcolor="rgba(239, 68, 68, 0.15)", line_width=1, line_dash="dash", line_color="#ef4444",
        annotation_text="Cyberattack Active", annotation_position="top left"
    )
    fig_disc.update_layout(
        title="Where the drone thinks it is vs where it really is (Tracking Error Over Time)",
        xaxis_title="Time (seconds)", yaxis_title="Tracking Error (meters)",
        template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_disc, use_container_width=True)


# =========================================================================
# TAB 2: RESILIENCE & RECOVERY (D2)
# =========================================================================
with tab2:
    st.subheader("Deliverable 2: Cyber-Resilience and Autonomous Recovery")

    # State Banner colored by system state
    state_css_map = {
        "NORMAL": "state-normal",
        "SUSPICIOUS": "state-suspicious",
        "ATTACK_CONFIRMED": "state-attack",
        "CONTAINMENT": "state-attack",
        "SAFE_NAVIGATION": "state-safe",
        "RECOVERY": "state-suspicious",
        "RECOVERED": "state-recovered",
        "DEFENSE_OFF": "state-off"
    }
    banner_class = state_css_map.get(current_state, "state-normal")
    st.markdown(f'<div class="state-banner {banner_class}">SYSTEM STATUS: {current_state} (Active Estimator: {current_tel.iloc[-1]["active_estimator"].upper() if not current_tel.empty else "MAIN"})</div>', unsafe_allow_html=True)

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Time to Detect", f"{metrics['detection']['time_to_detect_s'] or 'N/A'} s")
    with m2:
        st.metric("Time to Contain", f"{metrics['resilience']['time_to_contain_s'] or 'N/A'} s")
    with m3:
        st.metric("Position RMSE", f"{metrics['navigation']['position_rmse_m']} m")
    with m4:
        st.metric("Final Goal Error", f"{metrics['navigation']['final_goal_error_m']} m")
    with m5:
        st.metric("Collisions", f"{metrics['mission']['collisions']}")

    col_map, col_trust = st.columns([3, 2])

    with col_map:
        # Resilience Map: Estimated path, Safe path, Quarantined points
        fig_res = go.Figure()
        for obs in obstacles:
            theta = np.linspace(0, 2*np.pi, 50)
            # Draw attack inflation boundary (3.5m)
            ox_infl = obs["x"] + (obs.get("radius", 3.0) + 3.5) * np.cos(theta)
            oy_infl = obs["y"] + (obs.get("radius", 3.0) + 3.5) * np.sin(theta)
            fig_res.add_trace(go.Scatter(
                x=ox_infl, y=oy_infl, fill="toself", fillcolor="rgba(59, 130, 246, 0.12)",
                line=dict(color="#3b82f6", width=1, dash="dot"), name="Safe Inflation (3.5m)", showlegend=False
            ))
            # Obstacle core
            ox = obs["x"] + obs.get("radius", 3.0) * np.cos(theta)
            oy = obs["y"] + obs.get("radius", 3.0) * np.sin(theta)
            fig_res.add_trace(go.Scatter(
                x=ox, y=oy, fill="toself", fillcolor="rgba(239, 68, 68, 0.4)",
                line=dict(color="#ef4444", width=2), showlegend=False
            ))

        # Trusted EKF Path
        fig_res.add_trace(go.Scatter(
            x=current_tel["trusted_est_x"], y=current_tel["trusted_est_y"],
            mode="lines", line=dict(color="#3b82f6", width=3), name="Trusted Filter Estimate"
        ))

        # True path
        fig_res.add_trace(go.Scatter(
            x=current_tel["truth_x"], y=current_tel["truth_y"],
            mode="lines", line=dict(color="#10b981", width=2, dash="dash"), name="True Ground Path"
        ))

        # Quarantined GPS points (Red Xs)
        quarantined_pts = current_tel[current_tel["gps_quarantined"] == True]
        if not quarantined_pts.empty:
            fig_res.add_trace(go.Scatter(
                x=quarantined_pts["gps_x"], y=quarantined_pts["gps_y"],
                mode="markers", marker=dict(color="#ef4444", size=8, symbol="x"),
                name="Quarantined GPS Points"
            ))

        fig_res.update_layout(
            title="Safe Navigation & Quarantined Sensor Data",
            xaxis=dict(title="X (m)", range=[-10, 80]),
            yaxis=dict(title="Y (m)", range=[-10, 60]),
            template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_trust:
        # Sensor Trust Scores Bar Chart
        current_trust_row = df_trust[df_trust["time"] <= t_slice].iloc[-1] if not df_trust[df_trust["time"] <= t_slice].empty else df_trust.iloc[0]
        sensor_names = ["GPS", "IMU", "Barometer", "Vision", "LiDAR"]
        trust_vals = [
            current_trust_row["gps_trust"],
            current_trust_row["imu_trust"],
            current_trust_row["baro_trust"],
            current_trust_row["vision_trust"],
            current_trust_row["lidar_trust"]
        ]
        bar_colors = ["#10b981" if v > 0.7 else "#f59e0b" if v > 0.3 else "#ef4444" for v in trust_vals]

        fig_trust = go.Figure(go.Bar(
            x=sensor_names, y=trust_vals, marker_color=bar_colors, text=[f"{v:.2f}" for v in trust_vals], textposition="auto"
        ))
        fig_trust.update_layout(
            title="Dynamic Sensor Trust Scores", yaxis=dict(title="Trust Score (0-1)", range=[0, 1.1]),
            template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trust, use_container_width=True)

    # Event Timeline Table
    st.markdown("### ⏱️ Resilience State Transition Events")
    if events:
        df_events = pd.DataFrame(events)
        st.dataframe(df_events[["timestamp", "event", "state", "active_estimator", "details"]], use_container_width=True)
    else:
        st.info("No transition events recorded (Defense is OFF or clean flight).")


# =========================================================================
# TAB 3: COMPARATIVE ANALYSIS (OFF VS ON)
# =========================================================================
with tab3:
    st.subheader("Golden Rule Comparison: Defense OFF vs Defense ON")
    st.caption("Demonstrating proof of resilience by contrasting unprotected flight vs autonomous containment on the same scenario.")

    # Run Defense OFF version of same scenario for side-by-side
    with st.spinner("Generating side-by-side comparison..."):
        sim_off, _, _ = run_simulation(selected_scenario_file, False, seed)
        sim_on, _, _ = run_simulation(selected_scenario_file, True, seed)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### ❌ Defense OFF (Unprotected)")
        st.metric("Status", sim_off["metrics"]["mission"]["status"])
        st.metric("Position RMSE", f"{sim_off['metrics']['navigation']['position_rmse_m']} m")
        st.metric("Final Goal Error", f"{sim_off['metrics']['navigation']['final_goal_error_m']} m")
        st.metric("Collisions", sim_off["metrics"]["mission"]["collisions"])

        fig_cmp_off = go.Figure()
        for obs in obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
            ox = obs["x"] + obs.get("radius", 3.0) * np.cos(theta)
            oy = obs["y"] + obs.get("radius", 3.0) * np.sin(theta)
            fig_cmp_off.add_trace(go.Scatter(x=ox, y=oy, fill="toself", fillcolor="rgba(239, 68, 68, 0.3)", line=dict(color="#ef4444"), showlegend=False))
        fig_cmp_off.add_trace(go.Scatter(x=sim_off["telemetry"]["truth_x"], y=sim_off["telemetry"]["truth_y"], mode="lines", line=dict(color="#f97316", width=2.5), name="Drone True Path"))
        fig_cmp_off.update_layout(title="Trajectory (Defense OFF)", xaxis_title="X (m)", yaxis_title="Y (m)", template="plotly_dark", height=320)
        st.plotly_chart(fig_cmp_off, use_container_width=True)

    with c_right:
        st.markdown("#### ✅ Defense ON (Resilient)")
        st.metric("Status", sim_on["metrics"]["mission"]["status"])
        st.metric("Position RMSE", f"{sim_on['metrics']['navigation']['position_rmse_m']} m")
        st.metric("Final Goal Error", f"{sim_on['metrics']['navigation']['final_goal_error_m']} m")
        st.metric("Collisions", sim_on["metrics"]["mission"]["collisions"])

        fig_cmp_on = go.Figure()
        for obs in obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
            ox = obs["x"] + obs.get("radius", 3.0) * np.cos(theta)
            oy = obs["y"] + obs.get("radius", 3.0) * np.sin(theta)
            fig_cmp_on.add_trace(go.Scatter(x=ox, y=oy, fill="toself", fillcolor="rgba(16, 185, 129, 0.3)", line=dict(color="#10b981"), showlegend=False))
        fig_cmp_on.add_trace(go.Scatter(x=sim_on["telemetry"]["truth_x"], y=sim_on["telemetry"]["truth_y"], mode="lines", line=dict(color="#10b981", width=2.5), name="Drone True Path"))
        fig_cmp_on.update_layout(title="Trajectory (Defense ON)", xaxis_title="X (m)", yaxis_title="Y (m)", template="plotly_dark", height=320)
        st.plotly_chart(fig_cmp_on, use_container_width=True)

    # Leaderboard Table across Scenarios
    st.markdown("### 🏆 Leaderboard Summary across Scenarios")
    report_file = REPO_ROOT / "runs" / "reports" / "batch_summary.csv"
    if report_file.exists():
        df_batch = pd.read_csv(report_file)
        st.dataframe(df_batch, use_container_width=True)
    else:
        st.info("Run `python tools/run_batch.py` to generate the complete 70-run comparative leaderboard.")
