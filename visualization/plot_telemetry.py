"""
Offline Telemetry Plotting and Publication-Quality Figure Generator.

Generates high-resolution performance plots for:
1. 3D Flight Trajectory Comparison (Ground Truth vs. Spoofed vs. Resilient EKF)
2. Innovation Residuals (NIS) with Chi-Square statistical thresholds
3. Dynamic Sensor Trust Decay & Autonomous Navigation Mode Switching

Usage:
    python visualization/plot_telemetry.py
    python visualization/plot_telemetry.py --show
    python visualization/plot_telemetry.py --input path/to/telemetry.csv
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

# Add core modules to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.append(str(root_dir / "ros2_ws" / "src" / "resilience_manager"))

from state_estimator.ekf_10dof import EKF10DOF
from residual_detector.detector_engine import ResidualDetectorEngine, DetectionState
from resilience_manager.manager_engine import ResilienceManagerEngine


def generate_flight_data(duration: float = 40.0, dt: float = 0.1) -> pd.DataFrame:
    """Generates synthetic flight telemetry with an injected GPS spoofing attack."""
    time_steps = np.arange(0.0, duration, dt)
    attack_start = 12.0
    attack_end = 30.0
    attack_offset = 25.0

    ekf = EKF10DOF()
    detector = ResidualDetectorEngine(consecutive_alarms_to_confirm=4)
    resilience = ResilienceManagerEngine(quarantine_threshold=0.35)

    ekf.initialize_state(position=np.array([0.0, 0.0, 10.0]))
    records = []

    for t in time_steps:
        # Waypoint kinematics: Figure-8 path
        true_x = 10.0 * np.sin(0.2 * t)
        true_y = 10.0 * np.sin(0.4 * t)
        true_z = 10.0 + 1.5 * np.cos(0.15 * t)
        true_pos = np.array([true_x, true_y, true_z])

        accel = np.array([0.0, 0.0, 9.80665]) + np.random.normal(0.0, 0.02, size=(3,))
        gps_meas = true_pos + np.random.normal(0.0, 0.25, size=(3,))
        vision_meas = true_pos + np.random.normal(0.0, 0.05, size=(3,))

        # Inject GPS spoofing ramp
        is_attack = attack_start <= t <= attack_end
        if is_attack:
            ramp = min(1.0, (t - attack_start) / 3.0)
            gps_meas += np.array([attack_offset * ramp, -15.0 * ramp, 0.0])

        # Pipeline execution
        used_accel = np.array([0.0, 0.0, 9.80665]) if "imu" in resilience.isolated_sensors else accel
        ekf.predict(accel=used_accel, gyro_z=0.0, dt=dt)

        gps_res = ekf.update_gps(gps_meas, reject_anomaly=True)
        det_gps = detector.process_sensor_residual("gps", gps_res["position"]["nis"])
        policy = resilience.evaluate_resilience_policy(det_gps["state"], det_gps["compromised_sensors"])

        if "gps" in policy["isolated_sensors"]:
            ekf.update_vision_pose(vision_meas)

        est = ekf.get_state()
        records.append({
            "time": t,
            "true_x": true_x,
            "true_y": true_y,
            "true_z": true_z,
            "meas_gps_x": gps_meas[0],
            "meas_gps_y": gps_meas[1],
            "meas_gps_z": gps_meas[2],
            "est_x": est["position"][0],
            "est_y": est["position"][1],
            "est_z": est["position"][2],
            "gps_nis": gps_res["position"]["nis"],
            "attack_state": det_gps["state"],
            "nav_mode": policy["navigation_mode"],
            "gps_trust": policy["trust_scores"]["gps"],
            "vision_trust": policy["trust_scores"]["vision_pose"],
            "is_attack": is_attack
        })

    return pd.DataFrame(records)


def plot_all(df: pd.DataFrame, output_dir: Path, show: bool = False):
    """Generates and saves the 3 core evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # 1. 3D Trajectory Comparison
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df["true_x"], df["true_y"], df["true_z"], "b-", linewidth=2.5, label="Ground Truth Trajectory")
    ax.plot(df["meas_gps_x"], df["meas_gps_y"], df["meas_gps_z"], "r--", alpha=0.6, linewidth=1.5, label="Raw Spoofed GPS")
    ax.plot(df["est_x"], df["est_y"], df["est_z"], "g-", linewidth=2.5, label="Resilient EKF Estimate (Protected)")

    # Start and End Markers
    ax.scatter([df["true_x"].iloc[0]], [df["true_y"].iloc[0]], [df["true_z"].iloc[0]], color="blue", s=80, marker="o", label="Mission Start")
    ax.scatter([df["true_x"].iloc[-1]], [df["true_y"].iloc[-1]], [df["true_z"].iloc[-1]], color="black", s=80, marker="X", label="Mission End")

    ax.set_title("3D Flight Trajectory under GPS Spoofing Attack", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("East (m)", fontsize=10, labelpad=8)
    ax.set_ylabel("North (m)", fontsize=10, labelpad=8)
    ax.set_zlabel("Altitude (m)", fontsize=10, labelpad=8)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    traj_path = output_dir / "trajectory_3d_comparison.png"
    plt.savefig(traj_path, dpi=300)
    print(f"[*] Saved: {traj_path}")
    if show:
        plt.show()
    plt.close()

    # -------------------------------------------------------------
    # 2. Innovation Residuals (NIS) Analysis
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(df["time"], df["gps_nis"], color="#d95f02", linewidth=1.8, label="GPS Normalized Innovation Squared (NIS)")
    ax.axhline(y=16.27, color="red", linestyle="--", linewidth=1.5, label=r"Chi-Square Gate $\chi^2_{0.999}(3) = 16.27$")
    ax.axhline(y=11.34, color="orange", linestyle=":", linewidth=1.2, label=r"Warning Threshold $\chi^2_{0.99}(3) = 11.34$")

    # Highlight attack window
    attack_mask = df["is_attack"]
    if attack_mask.any():
        t_start = df.loc[attack_mask, "time"].min()
        t_end = df.loc[attack_mask, "time"].max()
        ax.axvspan(t_start, t_end, color="red", alpha=0.12, label="GPS Spoofing Active Window")

    ax.set_yscale("log")
    ax.set_title("Statistical Anomaly Detection: GPS Innovation Residual (NIS)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("NIS Statistic (Log Scale)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    nis_path = output_dir / "nis_residuals_analysis.png"
    plt.savefig(nis_path, dpi=300)
    print(f"[*] Saved: {nis_path}")
    if show:
        plt.show()
    plt.close()

    # -------------------------------------------------------------
    # 3. Dynamic Sensor Trust & Flight Modes
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)

    # Subplot 1: Sensor Trust
    ax1.plot(df["time"], df["gps_trust"], color="crimson", linewidth=2.0, label="GPS Trust Score")
    ax1.plot(df["time"], df["vision_trust"], color="forestgreen", linewidth=2.0, label="Vision/LiDAR Trust Score")
    ax1.axhline(y=0.35, color="black", linestyle="--", linewidth=1.2, label="Quarantine Threshold (0.35)")
    ax1.set_ylabel("Trust Score [0.0 - 1.0]", fontsize=11)
    ax1.set_title("Dynamic Sensor Trust & Autonomous Mode Reconfiguration", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Navigation Mode
    modes = sorted(df["nav_mode"].unique())
    mode_to_int = {m: idx for idx, m in enumerate(modes)}
    numeric_modes = df["nav_mode"].map(mode_to_int)

    ax2.step(df["time"], numeric_modes, where="post", color="darkblue", linewidth=2.0, label="Active Flight Mode")
    ax2.set_yticks(range(len(modes)))
    ax2.set_yticklabels(modes, fontsize=9)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Navigation Mode", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    trust_path = output_dir / "resilience_trust_modes.png"
    plt.savefig(trust_path, dpi=300)
    print(f"[*] Saved: {trust_path}")
    if show:
        plt.show()
    plt.close()

    # -------------------------------------------------------------
    # Summary Metrics
    # -------------------------------------------------------------
    raw_error = np.sqrt((df["meas_gps_x"] - df["true_x"])**2 + (df["meas_gps_y"] - df["true_y"])**2)
    ekf_error = np.sqrt((df["est_x"] - df["true_x"])**2 + (df["est_y"] - df["true_y"])**2)

    raw_rmse = float(np.sqrt(np.mean(raw_error ** 2)))
    ekf_rmse = float(np.sqrt(np.mean(ekf_error ** 2)))
    improvement = (1.0 - ekf_rmse / max(raw_rmse, 1e-6)) * 100.0

    print("\n" + "=" * 55)
    print("        TELEMETRY EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Raw GPS RMSE:             {raw_rmse:.3f} m")
    print(f"Resilient EKF RMSE:       {ekf_rmse:.3f} m")
    print(f"Tracking Improvement:     +{improvement:.1f} %")
    print(f"Max Position Error (Raw): {raw_error.max():.2f} m")
    print(f"Max Position Error (EKF): {ekf_error.max():.2f} m")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot offline drone telemetry and cyber-resilience metrics.")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV telemetry file.")
    parser.add_argument("--output", type=str, default="reports/experiment_results", help="Directory to save figures.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root_dir / out_path

    if args.input:
        df = pd.read_csv(args.input)
        print(f"[*] Loaded telemetry log: {args.input}")
    else:
        print("[*] Simulating flight mission with GPS spoofing attack...")
        df = generate_flight_data()

    plot_all(df, out_path, show=args.show)


if __name__ == "__main__":
    main()
