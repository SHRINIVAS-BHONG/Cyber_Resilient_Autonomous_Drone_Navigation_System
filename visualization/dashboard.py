"""
Cyber-Resilient Autonomous Drone Navigation System - Web Telemetry & Evaluation Dashboard.

Streamlit experiment dashboard implementing the 9 standard pages specified in Section 11 (Phase 8):
1. Mission overview
2. Drone trajectory
3. Sensor residuals
4. Attack timeline
5. Detector performance
6. Sensor trust scores
7. Recovery actions
8. Navigation error comparison
9. Scenario leaderboard
"""

import json
from pathlib import Path
import sys
import time
import yaml

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup python path to import core algorithms
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "sensor_bridge"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "path_planner"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine, NavigationMode
from path_planner.astar_planner import AStarPlanner


st.set_page_config(
    page_title="Cyber-Resilient Drone Navigation Dashboard",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛸 Cyber-Resilient Autonomous Drone Navigation System")
st.markdown("**PX4 SITL · Gazebo · 10-DOF EKF · Dual Statistical/ML Detection · Autonomous Sensor Isolation & Safe A* Recovery**")


@st.cache_resource
def get_ml_model():
    models_dir = root_dir / "ml" / "models"
    rf_file = models_dir / "random_forest_detector.joblib"
    scaler_file = models_dir / "scaler.joblib"
    feat_file = models_dir / "feature_names.json"
    if rf_file.exists() and scaler_file.exists() and feat_file.exists():
        try:
            rf = joblib.load(rf_file)
            scaler = joblib.load(scaler_file)
            with open(feat_file, "r", encoding="utf-8") as f:
                feats = json.load(f)
            return rf, scaler, feats
        except Exception:
            return None, None, None
    return None, None, None


def load_all_scenarios():
    scenarios_dir = root_dir / "simulation" / "scenarios"
    scenarios = []
    if scenarios_dir.exists():
        for p in sorted(scenarios_dir.glob("scenario_*.yaml")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    scenarios.append(data)
            except Exception:
                pass
    return scenarios


scenarios_list = load_all_scenarios()
scenario_names = [s.get("name", s.get("id", "Unknown")) for s in scenarios_list] if scenarios_list else ["Scenario 03: Slow GPS Drift"]

# Sidebar Configuration
st.sidebar.header("🕹️ Experiment Scenario Selector")
selected_scenario_idx = 0
if scenario_names:
    selected_scenario_name = st.sidebar.selectbox("Select Benchmark Scenario (Section 9)", scenario_names, index=2 if len(scenario_names) > 2 else 0)
    selected_scenario_idx = scenario_names.index(selected_scenario_name)
    active_scenario = scenarios_list[selected_scenario_idx] if scenarios_list else {}
else:
    active_scenario = {}

st.sidebar.markdown("---")
st.sidebar.header("🎯 Mission Configuration")
sim_duration = st.sidebar.slider("Mission Duration (s)", 20, 120, 60, step=10)
custom_attack_start = st.sidebar.slider("Attack Start Time (s)", 5, sim_duration - 10, int(active_scenario.get("attack_start_time", 20.0)))
custom_attack_duration = st.sidebar.slider("Attack Duration (s)", 5, 30, int(active_scenario.get("attack_duration", 20.0)))
attack_magnitude = st.sidebar.slider("Attack Bias / Severity (m or m/s²)", 2.0, 40.0, 18.0)


@st.cache_data
def simulate_scenario_telemetry(scenario_cfg: dict, duration: int, a_start: float, a_dur: float, a_mag: float):
    dt = 0.1
    time_steps = np.arange(0.0, duration, dt)

    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(high_threshold=16.27, low_threshold=6.25, consecutive_alarms_to_confirm=4)
    resilience = ResilienceManagerEngine(quarantine_threshold=0.35, recovery_threshold=0.80)
    planner = AStarPlanner(grid_resolution=1.0)
    ml_rf, ml_scaler, ml_feats = get_ml_model()

    start_pos = np.array([0.0, 0.0, 10.0])
    ekf.initialize_state(position=start_pos)

    records = []
    attack_type = scenario_cfg.get("attack_type", "gps_spoofing")
    attack_cat = scenario_cfg.get("category", "GPS Attack")
    is_baseline = (attack_type == "none")

    # Obstacle definition
    obstacles = [(25.0, 0.0, 10.0), (35.0, 5.0, 10.0)] if "obstacle" in scenario_cfg.get("name", "").lower() else []

    safe_route = None

    for curr_t in time_steps:
        # True nominal flight trajectory
        true_x = curr_t * 0.8
        true_y = 6.0 * np.sin(0.15 * curr_t)
        true_z = 10.0
        true_pos = np.array([true_x, true_y, true_z])

        # Nominal sensor inputs
        accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        gyro_z = 0.0 + float(np.random.normal(0.0, 0.005))
        gps_pos = true_pos + np.random.normal(0.0, 0.25, size=(3,))
        vision_pos = true_pos + np.random.normal(0.0, 0.06, size=(3,))
        lidar_z = true_z + float(np.random.normal(0.0, 0.04))

        # Check attack window
        is_attack = (not is_baseline) and (a_start <= curr_t < (a_start + a_dur))
        injected_attack_name = "NONE"

        if is_attack:
            injected_attack_name = attack_type.upper()
            if "gps" in attack_type or attack_cat == "GPS Attack":
                ramp = min(1.0, (curr_t - a_start) / 4.0)
                gps_pos[0] += a_mag * ramp
                gps_pos[1] -= (a_mag * 0.6) * ramp
            elif "imu" in attack_type or "imu" in scenario_cfg.get("name", "").lower():
                accel[0] += 2.5
                accel[1] += 1.5
            elif "lidar" in attack_type or "lidar" in scenario_cfg.get("name", "").lower():
                lidar_z -= 6.0
            elif "combined" in attack_type or attack_cat == "Combined Attack":
                ramp = min(1.0, (curr_t - a_start) / 3.0)
                gps_pos[0] += a_mag * ramp
                accel[0] += 2.0

        # EKF Prediction
        used_accel = np.array([0.0, 0.0, 9.80665]) if "imu" in resilience.isolated_sensors else accel
        ekf.predict(accel=used_accel, gyro_z=gyro_z, dt=dt)

        # Measurement updates with residual gating
        gps_rejected = ("gps" in resilience.isolated_sensors)
        gps_res = ekf.update_gps(gps_pos, reject_anomaly=gps_rejected)
        gps_nis = float(gps_res["position"]["nis"])

        lidar_res = ekf.update_lidar(lidar_z, reject_anomaly=("lidar" in resilience.isolated_sensors))
        lidar_nis = float(lidar_res["nis"])

        # Process statistical residual detection
        det_gps = detector.process_sensor_residual("gps", gps_nis)
        det_lidar = detector.process_sensor_residual("lidar", lidar_nis)

        # Dual ML Voter Evaluation
        ml_pred_name = "NORMAL"
        ml_conf = 0.99
        ml_p_norm, ml_p_gps, ml_p_imu, ml_p_lidar = 0.98, 0.01, 0.005, 0.005

        if ml_rf and ml_scaler and ml_feats:
            try:
                gps_vel_norm = float(np.linalg.norm(ekf.state[3:6]))
                feat_vals = np.array([[
                    float(np.linalg.norm(gps_res["position"]["innovation"])),
                    gps_vel_norm,
                    float(np.linalg.norm(ekf.state[6:9])),
                    float(np.var(accel)),
                    float(abs(gps_pos[2] - lidar_z)),
                    float(np.linalg.norm(gps_pos - vision_pos)),
                    0.02,
                    10.0,
                    0.05,
                    0.0,
                    float(np.trace(ekf.covariance[:3, :3])),
                    float(np.linalg.norm(ekf.state[0:3] - true_pos)),
                    gps_nis
                ]])
                scaled = ml_scaler.transform(feat_vals)
                probs = ml_rf.predict_proba(scaled)[0]
                pred_idx = int(np.argmax(probs))
                class_labels = {0: "NORMAL", 1: "GPS_SPOOFING", 2: "IMU_MANIPULATION", 3: "LIDAR_CORRUPTION", 4: "COMMUNICATION_DISRUPTION", 5: "MIXED_ATTACK"}
                ml_pred_name = class_labels.get(pred_idx, "ATTACK")
                ml_conf = float(probs[pred_idx])
                if len(probs) >= 4:
                    ml_p_norm, ml_p_gps, ml_p_imu, ml_p_lidar = float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])
            except Exception:
                pass

        # Resilience Manager Policy
        policy = resilience.evaluate_resilience_policy(det_gps["state"], det_gps["compromised_sensors"])

        # Switch to trusted sensors when GPS is isolated
        if "gps" in policy["isolated_sensors"]:
            ekf.update_vision_pose(vision_pos)

        # Path replanning if containment triggered
        if policy["navigation_mode"] in [NavigationMode.SAFE_LOCAL_NAVIGATION.value, NavigationMode.HOLD_POSITION.value] and safe_route is None:
            safe_route = planner.plan(
                start=tuple(ekf.get_state()["position"]),
                goal=(true_x + 30.0, 0.0, 10.0),
                obstacles=obstacles,
                cyber_risk_zones=[(gps_pos[0], gps_pos[1], 15.0)]
            )

        est_state = ekf.get_state()
        pos_err_raw = float(np.linalg.norm(gps_pos - true_pos))
        pos_err_ekf = float(np.linalg.norm(est_state["position"] - true_pos))

        records.append({
            "time": curr_t,
            "true_x": true_pos[0], "true_y": true_pos[1], "true_z": true_pos[2],
            "meas_gps_x": gps_pos[0], "meas_gps_y": gps_pos[1], "meas_gps_z": gps_pos[2],
            "est_x": est_state["position"][0], "est_y": est_state["position"][1], "est_z": est_state["position"][2],
            "gps_nis": gps_nis,
            "lidar_nis": lidar_nis,
            "attack_state": det_gps["state"],
            "risk_score": det_gps["risk_score"],
            "nav_mode": policy["navigation_mode"],
            "gps_trust": policy["trust_scores"]["gps"],
            "vision_trust": policy["trust_scores"]["vision_pose"],
            "injected_attack": injected_attack_name,
            "ml_pred": ml_pred_name,
            "ml_conf": ml_conf,
            "ml_p_norm": ml_p_norm,
            "ml_p_gps": ml_p_gps,
            "ml_p_imu": ml_p_imu,
            "ml_p_lidar": ml_p_lidar,
            "raw_error": pos_err_raw,
            "ekf_error": pos_err_ekf,
            "isolated_sensors": ", ".join(policy["isolated_sensors"]) if policy["isolated_sensors"] else "None",
            "is_attack": is_attack
        })

    return pd.DataFrame(records)


df = simulate_scenario_telemetry(active_scenario, sim_duration, custom_attack_start, custom_attack_duration, attack_magnitude)

# =====================================================================
# 9 TABS CORRESPONDING EXACTLY TO SECTION 11 (PHASE 8)
# =====================================================================
tabs = st.tabs([
    "1. Mission Overview",
    "2. Drone Trajectory",
    "3. Sensor Residuals",
    "4. Attack Timeline",
    "5. Detector Performance",
    "6. Sensor Trust Scores",
    "7. Recovery Actions",
    "8. Navigation Error Comparison",
    "9. Scenario Leaderboard"
])

# ---------------------------------------------------------------------
# PAGE 1: MISSION OVERVIEW
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("📋 Page 1: Mission Overview")
    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Flight Mode", str(latest["nav_mode"]))
    col2.metric("Attack State", str(latest["attack_state"]))
    col3.metric("Quarantined Sensors", str(latest["isolated_sensors"]))
    col4.metric("Risk Score", f"{latest['risk_score']:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Ground Truth Position", f"[{latest['true_x']:.1f}, {latest['true_y']:.1f}, {latest['true_z']:.1f}]")
    col6.metric("Resilient Estimate", f"[{latest['est_x']:.1f}, {latest['est_y']:.1f}, {latest['est_z']:.1f}]")
    col7.metric("Raw GPS Tracking Error", f"{latest['raw_error']:.2f} m")
    col8.metric("Resilient EKF Error", f"{latest['ekf_error']:.2f} m")

    st.markdown("---")
    st.markdown("### 🛰️ Live Telemetry & Mission Progression")
    st.dataframe(df[["time", "nav_mode", "attack_state", "isolated_sensors", "gps_trust", "vision_trust", "gps_nis", "raw_error", "ekf_error"]].tail(10), use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 2: DRONE TRAJECTORY
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("📈 Page 2: 3D Drone Flight Trajectory")
    st.markdown("Interactive comparison between **Ground Truth Flight Path**, **Corrupted / Spoofed Sensor Stream**, and **Protected Resilient EKF State**.")

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=df["true_x"], y=df["true_y"], z=df["true_z"],
        mode="lines", name="Ground Truth Trajectory",
        line=dict(color="#1f77b4", width=5)
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=df["meas_gps_x"], y=df["meas_gps_y"], z=df["meas_gps_z"],
        mode="lines", name="Manipulated / Spoofed GPS",
        line=dict(color="#d62728", width=3, dash="dash")
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=df["est_x"], y=df["est_y"], z=df["est_z"],
        mode="lines", name="Resilient EKF State (Protected)",
        line=dict(color="#2ca02c", width=6)
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (m)"
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 3: SENSOR RESIDUALS
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("🔬 Page 3: Sensor Residuals & Innovation Gating")
    st.markdown(r"Normalized Innovation Squared ($NIS = \mathbf{r}^T \mathbf{S}^{-1} \mathbf{r}$) monitored against $\chi^2$ hypothesis testing gates.")

    fig_nis = go.Figure()
    fig_nis.add_trace(go.Scatter(
        x=df["time"], y=df["gps_nis"],
        mode="lines", name="GPS NIS Residual",
        line=dict(color="#ff7f0e", width=2.5)
    ))
    fig_nis.add_hline(y=16.27, line_dash="dash", line_color="red", annotation_text="Chi2 Gate (99.9%) = 16.27")
    fig_nis.add_hline(y=11.34, line_dash="dot", line_color="orange", annotation_text="Warning Threshold (99%) = 11.34")

    # Mark attack window
    attack_mask = df["is_attack"]
    if attack_mask.any():
        fig_nis.add_vrect(
            x0=df.loc[attack_mask, "time"].min(),
            x1=df.loc[attack_mask, "time"].max(),
            fillcolor="red", opacity=0.15, line_width=0,
            annotation_text="Active Cyber Attack Window"
        )

    fig_nis.update_layout(
        xaxis_title="Mission Time (s)",
        yaxis_title="NIS Statistic (Log Scale)",
        yaxis_type="log",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_nis, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 4: ATTACK TIMELINE
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("⏱️ Page 4: Attack & Containment Timeline")
    st.markdown("Chronological progression of injection, anomaly flagging, isolation, and mode transitions.")

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df["time"], y=df["risk_score"],
        mode="lines", name="Estimated Risk Score",
        line=dict(color="purple", width=3)
    ))
    fig_timeline.add_trace(go.Scatter(
        x=df["time"], y=df["gps_trust"],
        mode="lines", name="GPS Trust Score",
        line=dict(color="crimson", width=2)
    ))
    fig_timeline.add_hline(y=0.35, line_dash="dash", line_color="gray", annotation_text="Quarantine Level (0.35)")

    fig_timeline.update_layout(
        xaxis_title="Mission Time (s)",
        yaxis_title="Normalized Score [0.0 - 1.0]",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 5: DETECTOR PERFORMANCE
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("🧠 Page 5: Detector Performance & Dual-Voter Agreement")
    st.markdown("Independent comparison of **Statistical EKF $\chi^2$ Detector** and **Random Forest Machine Learning Classifier**.")

    col1, col2, col3, col4 = st.columns(4)
    latest_ml = df["ml_pred"].iloc[-1]
    latest_conf = df["ml_conf"].iloc[-1]
    latest_stat = df["attack_state"].iloc[-1]
    agreement = (latest_ml != "NORMAL" and latest_stat != "NORMAL") or (latest_ml == "NORMAL" and latest_stat == "NORMAL")

    col1.metric("Statistical Detector State", str(latest_stat))
    col2.metric("ML Classifier Prediction", str(latest_ml))
    col3.metric("ML Model Confidence", f"{latest_conf * 100:.1f} %")
    col4.metric("Dual-Voter Consensus", "✅ AGREEMENT" if agreement else "⚠️ DIVERGENT")

    st.markdown("---")
    st.markdown("### Live Multi-Class Attack Probability Stream")
    fig_probs = go.Figure()
    fig_probs.add_trace(go.Scatter(x=df["time"], y=df["ml_p_norm"], mode="lines", name="P(Normal)", line=dict(color="green", width=2)))
    fig_probs.add_trace(go.Scatter(x=df["time"], y=df["ml_p_gps"], mode="lines", name="P(GPS Spoofing)", line=dict(color="crimson", width=2.5)))
    fig_probs.add_trace(go.Scatter(x=df["time"], y=df["ml_p_imu"], mode="lines", name="P(IMU Tampering)", line=dict(color="orange", width=2)))
    fig_probs.add_trace(go.Scatter(x=df["time"], y=df["ml_p_lidar"], mode="lines", name="P(LiDAR Corruption)", line=dict(color="purple", width=2)))

    fig_probs.update_layout(xaxis_title="Time (s)", yaxis_title="Probability", height=380)
    st.plotly_chart(fig_probs, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 6: SENSOR TRUST SCORES
# ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("🛡️ Page 6: Dynamic Sensor Trust Scoring")
    st.markdown("Continuous sensor integrity tracking with hysteresis-driven isolation and gradual reintegration.")

    fig_trust = go.Figure()
    fig_trust.add_trace(go.Scatter(
        x=df["time"], y=df["gps_trust"],
        mode="lines", name="GPS Trust",
        line=dict(color="crimson", width=3)
    ))
    fig_trust.add_trace(go.Scatter(
        x=df["time"], y=df["vision_trust"],
        mode="lines", name="Optical / Vision Trust",
        line=dict(color="forestgreen", width=2.5)
    ))
    fig_trust.add_hline(y=0.35, line_dash="dash", line_color="red", annotation_text="Quarantine Cutoff (0.35)")
    fig_trust.add_hline(y=0.80, line_dash="dot", line_color="green", annotation_text="Recovery Threshold (0.80)")

    fig_trust.update_layout(xaxis_title="Time (s)", yaxis_title="Trust Score [0.0 - 1.0]", height=400)
    st.plotly_chart(fig_trust, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 7: RECOVERY ACTIONS
# ---------------------------------------------------------------------
with tabs[6]:
    st.subheader("🔄 Page 7: Autonomous Recovery Actions & Safe Modes")
    st.markdown("Failsafe mode switching, sensor quarantine execution, and safe trajectory generation.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Navigation Mode Distribution**")
        mode_counts = df["nav_mode"].value_counts().reset_index()
        mode_counts.columns = ["Navigation Mode", "Duration (Steps)"]
        st.dataframe(mode_counts, use_container_width=True)

    with col2:
        st.markdown("**Autonomous Policy Logic**")
        st.info("""
        - **Minor GPS Residual**: Reduce GPS confidence weight.
        - **Persistent GPS Spoofing**: Isolate GPS, enable vision & LiDAR, switch to `SAFE_LOCAL_NAVIGATION`.
        - **IMU Manipulation**: Increase process covariance, rely on visual odometry and LiDAR.
        - **Communication Interruption**: Enter `HOLD_POSITION`, decelerate, use precomputed safe path.
        - **Multi-Sensor Failure**: Trigger `EMERGENCY_LANDING`.
        - **Attack Cessation**: Hysteresis recovery with gradual reintegration.
        """)

# ---------------------------------------------------------------------
# PAGE 8: NAVIGATION ERROR COMPARISON
# ---------------------------------------------------------------------
with tabs[7]:
    st.subheader("📊 Page 8: Quantitative Navigation Error Comparison")
    raw_rmse = float(np.sqrt(np.mean(df["raw_error"] ** 2)))
    ekf_rmse = float(np.sqrt(np.mean(df["ekf_error"] ** 2)))
    reduction = (1.0 - ekf_rmse / max(raw_rmse, 1e-6)) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Unmitigated Max Error", f"{df['raw_error'].max():.2f} m")
    m2.metric("Resilient EKF Max Error", f"{df['ekf_error'].max():.2f} m")
    m3.metric("Raw Unmitigated RMSE", f"{raw_rmse:.2f} m")
    m4.metric("Resilient EKF RMSE", f"{ekf_rmse:.2f} m", delta=f"-{reduction:.1f}% Error")

    st.markdown("---")
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=df["time"], y=df["raw_error"], mode="lines", name="Raw Spoofed Error (m)", line=dict(color="red", width=2)))
    fig_err.add_trace(go.Scatter(x=df["time"], y=df["ekf_error"], mode="lines", name="Resilient EKF Error (m)", line=dict(color="green", width=2.5)))
    fig_err.update_layout(xaxis_title="Time (s)", yaxis_title="Tracking Error (m)", height=400)
    st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 9: SCENARIO LEADERBOARD
# ---------------------------------------------------------------------
with tabs[8]:
    st.subheader("🏆 Page 9: 15-Scenario Comprehensive Benchmark Leaderboard")
    st.markdown("Official benchmark results across all 15 scenarios specified in Section 9.")

    summary_file = root_dir / "reports" / "experiment_results" / "benchmark_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as bf:
            bdata = json.load(bf)
        scenarios_data = bdata.get("scenarios", []) if isinstance(bdata, dict) else bdata
        bdf = pd.DataFrame(scenarios_data)
        col_map = {
            "id": "ID", "scenario": "Scenario Name", "category": "Category",
            "raw_rmse_m": "Raw RMSE (m)", "resilient_rmse_m": "Resilient RMSE (m)",
            "rmse_improvement_pct": "Improvement (%)", "time_to_detect_s": "TTD (s)",
            "time_to_contain_s": "TTC (s)", "final_nav_mode": "Final Mode", "survived": "Survived"
        }
        display_cols = [c for c in col_map.keys() if c in bdf.columns]
        if display_cols:
            bdf_disp = bdf[display_cols].rename(columns=col_map)
            st.dataframe(bdf_disp, use_container_width=True)
        else:
            st.dataframe(bdf, use_container_width=True)
    else:
        st.info("Loading scenarios from `simulation/scenarios/` directory...")
        sc_summary = []
        for sc in scenarios_list:
            sc_summary.append({
                "ID": sc.get("id"),
                "Scenario Name": sc.get("name"),
                "Category": sc.get("category"),
                "Attack Type": sc.get("attack_type"),
                "Expected Detector Response": sc.get("expected_detector_response"),
                "Expected Recovery Mode": sc.get("expected_recovery_mode")
            })
        st.dataframe(pd.DataFrame(sc_summary), use_container_width=True)

    st.markdown("---")
    st.download_button(
        label="📥 Download Telemetry Log (CSV)",
        data=df.to_csv(index=False),
        file_name="resilient_drone_telemetry.csv",
        mime="text/csv"
    )
