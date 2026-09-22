"""
Cyber-Resilient Autonomous Drone Navigation System - Web Telemetry & Evaluation Dashboard.

Production Streamlit web application providing live telemetry visualization,
cyber-attack injection triggers, 3D flight trajectory reconstruction,
innovation residual analysis, sensor trust scoring, and resilience metrics.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup python path to import core algorithms
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "path_planner"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine
from resilience_manager.manager_engine import ResilienceManagerEngine
from path_planner.astar_planner import AStarPlanner


st.set_page_config(
    page_title="Cyber-Resilient Drone Navigation",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛸 Cyber-Resilient Autonomous Drone Navigation Dashboard")
st.markdown("Real-time telemetry, cyberattack detection, and autonomous resilience engine.")

# Sidebar Configuration
st.sidebar.header("🎯 Mission & Cyber Attack Controls")

sim_duration = st.sidebar.slider("Mission Duration (seconds)", min_value=20, max_value=120, value=60, step=10)

st.sidebar.subheader("⚔️ Cyber Attack Injection")
attack_type = st.sidebar.selectbox(
    "Attack Type",
    ["None (Baseline)", "GPS Spoofing (Drift & Jump)", "IMU Manipulation (Bias)", "LiDAR Range Corruption", "Multi-Sensor Coordinated Attack"]
)

attack_start = st.sidebar.slider("Attack Start Time (s)", min_value=5, max_value=sim_duration - 10, value=20)
attack_duration = st.sidebar.slider("Attack Duration (s)", min_value=5, max_value=30, value=20)
attack_magnitude = st.sidebar.slider("Attack Severity / Bias (meters or m/s²)", min_value=2.0, max_value=40.0, value=18.0)

# Run Simulation Engine
@st.cache_data
def run_simulation(duration: int, attack: str, a_start: float, a_dur: float, a_mag: float):
    dt = 0.1
    time_steps = np.arange(0.0, duration, dt)
    n_steps = len(time_steps)

    # Waypoint flight trajectory (Figure-8 / Waypoint patrol)
    t = time_steps
    ground_truth_x = 15.0 * np.sin(0.15 * t)
    ground_truth_y = 15.0 * np.sin(0.30 * t)
    ground_truth_z = 10.0 + 2.0 * np.sin(0.1 * t)

    # Initialize Core Engines
    ekf = EKF10DOF()
    ekf.initialize_state(
        position=np.array([ground_truth_x[0], ground_truth_y[0], ground_truth_z[0]]),
        velocity=np.array([0.0, 0.0, 0.0])
    )
    detector = ResidualDetectorEngine()
    resilience = ResilienceManagerEngine()

    records = []
    a_end = a_start + a_dur

    for i, curr_t in enumerate(time_steps):
        true_pos = np.array([ground_truth_x[i], ground_truth_y[i], ground_truth_z[i]])

        # 1. Generate normal sensor measurements
        gps_pos = true_pos + np.random.normal(0.0, 0.3, size=(3,))
        imu_accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        lidar_z = float(true_pos[2] + np.random.normal(0.0, 0.03))
        vision_pos = true_pos + np.random.normal(0.0, 0.05, size=(3,))

        # 2. Inject Cyber Attack
        is_attack_active = a_start <= curr_t < a_end
        injected_attack_name = "None"

        if is_attack_active:
            if "GPS Spoofing" in attack:
                injected_attack_name = "GPS Spoofing"
                # Ramp bias
                ramp = min(1.0, (curr_t - a_start) / 4.0)
                gps_pos += np.array([a_mag * ramp, -a_mag * 0.7 * ramp, 0.0])
            elif "IMU Manipulation" in attack:
                injected_attack_name = "IMU Manipulation"
                imu_accel += np.array([a_mag * 0.2, -a_mag * 0.15, 0.5])
            elif "LiDAR" in attack:
                injected_attack_name = "LiDAR Corruption"
                lidar_z -= a_mag * 0.3
            elif "Multi-Sensor" in attack:
                injected_attack_name = "Multi-Sensor Coordinated"
                gps_pos += np.array([a_mag * 0.8, -a_mag * 0.5, 0.0])
                imu_accel += np.array([1.5, -1.0, 0.0])

        # 3. EKF Prediction
        ekf.predict(accel=imu_accel, gyro_z=0.0, dt=dt)

        # 4. Sensor Update & Anomaly Extraction
        gps_res = ekf.update_gps(gps_pos, reject_anomaly=False)
        gps_nis = gps_res["position"]["nis"]

        # 5. Residual Detection Engine
        det_res = detector.process_sensor_residual(sensor_name="gps", nis=gps_nis)

        # 6. Resilience & Isolation Manager
        res_policy = resilience.evaluate_resilience_policy(
            attack_status=det_res["state"],
            detected_compromised_sensors=det_res["compromised_sensors"]
        )

        # 7. Resilient Estimator correction: If GPS quarantined, fuse Vision + LiDAR
        if "gps" in res_policy["isolated_sensors"]:
            # Correct using trusted alternate sensors
            ekf.update_vision_pose(vision_pos)
            ekf.update_lidar(lidar_z)

        current_est = ekf.get_state()

        records.append({
            "time": curr_t,
            "true_x": true_pos[0],
            "true_y": true_pos[1],
            "true_z": true_pos[2],
            "meas_gps_x": gps_pos[0],
            "meas_gps_y": gps_pos[1],
            "meas_gps_z": gps_pos[2],
            "est_x": current_est["position"][0],
            "est_y": current_est["position"][1],
            "est_z": current_est["position"][2],
            "gps_nis": gps_nis,
            "attack_state": det_res["state"],
            "is_anomaly": det_res["is_anomaly"],
            "confidence": det_res["confidence"],
            "risk_score": det_res["risk_score"],
            "nav_mode": res_policy["navigation_mode"],
            "gps_trust": res_policy["trust_scores"]["gps"],
            "vision_trust": res_policy["trust_scores"]["vision_pose"],
            "injected_attack": injected_attack_name
        })

    return pd.DataFrame(records)

df = run_simulation(sim_duration, attack_type, attack_start, attack_duration, attack_magnitude)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 3D Flight Trajectory & Recovery",
    "🔬 Innovation Residuals & NIS Gating",
    "🛡️ Dynamic Sensor Trust & Resilience",
    "📊 Performance Metrics & Export"
])

with tab1:
    st.subheader("3D Trajectory Reconstruction (Ground Truth vs. Spoofed vs. Resilient EKF)")
    fig_3d = go.Figure()

    # Ground Truth
    fig_3d.add_trace(go.Scatter3d(
        x=df["true_x"], y=df["true_y"], z=df["true_z"],
        mode="lines", name="Ground Truth Trajectory",
        line=dict(color="cyan", width=5)
    ))

    # Raw GPS (Spoofed)
    fig_3d.add_trace(go.Scatter3d(
        x=df["meas_gps_x"], y=df["meas_gps_y"], z=df["meas_gps_z"],
        mode="lines", name="Raw Measured GPS (Under Attack)",
        line=dict(color="red", width=3, dash="dot")
    ))

    # Resilient EKF Estimate
    fig_3d.add_trace(go.Scatter3d(
        x=df["est_x"], y=df["est_y"], z=df["est_z"],
        mode="lines", name="Resilient Estimated Path (Protected)",
        line=dict(color="#00FF7F", width=6)
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (m)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=550
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Normalized Innovation Squared (NIS) vs Chi-Square Thresholds")
    fig_nis = go.Figure()

    fig_nis.add_trace(go.Scatter(
        x=df["time"], y=df["gps_nis"],
        mode="lines", name="GPS NIS Residual",
        line=dict(color="#FF8C00", width=2)
    ))

    # 99.9% Chi-Square threshold
    fig_nis.add_hline(y=16.27, line_dash="dash", line_color="red", annotation_text="Chi2 Alarm Gate (99.9%) = 16.27")
    fig_nis.add_hline(y=11.34, line_dash="dot", line_color="yellow", annotation_text="Warning Threshold (99%) = 11.34")

    fig_nis.update_layout(
        xaxis_title="Simulation Time (s)",
        yaxis_title="NIS Statistic",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_nis, use_container_width=True)

with tab3:
    st.subheader("Sensor Trust Scoring & Autonomous Failsafe Switching")
    col1, col2 = st.columns(2)

    with col1:
        fig_trust = go.Figure()
        fig_trust.add_trace(go.Scatter(
            x=df["time"], y=df["gps_trust"],
            mode="lines", name="GPS Trust Score",
            line=dict(color="crimson", width=3)
        ))
        fig_trust.add_trace(go.Scatter(
            x=df["time"], y=df["vision_trust"],
            mode="lines", name="Optical / Vision Trust",
            line=dict(color="lightgreen", width=2)
        ))
        fig_trust.add_hline(y=0.35, line_dash="dash", line_color="red", annotation_text="Quarantine Cutoff (0.35)")
        fig_trust.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Trust Score (0.0 - 1.0)",
            height=350
        )
        st.plotly_chart(fig_trust, use_container_width=True)

    with col2:
        st.markdown("### Active Flight Safety Mode")
        latest_mode = df["nav_mode"].iloc[-1]
        st.metric(label="Current Navigation Mode", value=latest_mode)
        st.markdown("**Mode Transition Timeline:**")
        mode_counts = df["nav_mode"].value_counts()
        st.dataframe(mode_counts, use_container_width=True)

with tab4:
    st.subheader("Quantitative Resilience Benchmarks")
    pos_error_raw = np.sqrt((df["meas_gps_x"] - df["true_x"])**2 + (df["meas_gps_y"] - df["true_y"])**2)
    pos_error_ekf = np.sqrt((df["est_x"] - df["true_x"])**2 + (df["est_y"] - df["true_y"])**2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw GPS Max Error", f"{pos_error_raw.max():.2f} m")
    m2.metric("Resilient EKF Max Error", f"{pos_error_ekf.max():.2f} m")
    m3.metric("Raw GPS RMSE", f"{np.sqrt(np.mean(pos_error_raw**2)):.2f} m")
    m4.metric("Resilient EKF RMSE", f"{np.sqrt(np.mean(pos_error_ekf**2)):.2f} m")

    st.markdown("---")
    st.subheader("📋 System Benchmark Suite Evaluation (5 Critical Attack Scenarios)")
    benchmark_file = root_dir / "reports" / "experiment_results" / "benchmark_summary.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r", encoding="utf-8") as bf:
            bdata = json.load(bf)
        bdf = pd.DataFrame(bdata)
        bdf.columns = ["Scenario", "Raw RMSE (m)", "Resilient RMSE (m)", "Improvement (%)", "TTD (s)", "TTC (s)", "Final Mode"]
        st.dataframe(bdf, use_container_width=True)
    else:
        st.info("Run `python scripts/run_benchmarks.py` to generate the complete scenario benchmark suite.")

    st.markdown("---")
    st.download_button(
        label="📥 Download Telemetry Log (CSV)",
        data=df.to_csv(index=False),
        file_name="resilient_drone_telemetry.csv",
        mime="text/csv"
    )
