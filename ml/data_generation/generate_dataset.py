"""
Synthetic Multi-Sensor Flight Data Generator for Cyber-Attack Classification.

Generates realistic multi-sensor telemetry with labeled attacks across 4 classes:
  0: NORMAL (Attack-free nominal flight)
  1: GPS_SPOOFING (Gradual ramp drift, sudden coordinate jumps, velocity spoofing)
  2: IMU_MANIPULATION (Accelerometer bias manipulation, scaling, gyroscope bias)
  3: LIDAR_CORRUPTION (Ground range corruption, altitude step drops, multipath noise)

Extracts 13 statistical and kinematic features per sample and saves:
  - ml/data/train_dataset.csv
  - ml/data/test_dataset.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Feature names
FEATURE_NAMES = [
    "gps_nis",
    "gps_pos_residual_norm",
    "gps_vel_residual_norm",
    "imu_accel_norm_diff",
    "imu_accel_var",
    "imu_gyro_var",
    "lidar_nis",
    "lidar_residual_z",
    "vision_gps_discrepancy",
    "vision_lidar_z_discrepancy",
    "accel_kinematic_consistency",
    "consecutive_anomalies",
    "sensor_trust_composite"
]

LABEL_NAMES = {
    0: "NORMAL",
    1: "GPS_SPOOFING",
    2: "IMU_MANIPULATION",
    3: "LIDAR_CORRUPTION"
}


def simulate_scenario(class_id: int, num_samples: int, dt: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Simulates a flight trajectory under the specified cyber attack class."""
    np.random.seed(seed)
    t = np.arange(num_samples) * dt

    # Nominal 3D waypoint motion
    vx = 2.0 * np.cos(0.15 * t)
    vy = 2.0 * np.sin(0.15 * t)
    vz = 0.5 * np.cos(0.1 * t)

    true_x = np.cumsum(vx) * dt
    true_y = np.cumsum(vy) * dt
    true_z = 10.0 + np.cumsum(vz) * dt

    # Sensor measurements with nominal noise
    gps_x = true_x + np.random.normal(0.0, 0.25, size=num_samples)
    gps_y = true_y + np.random.normal(0.0, 0.25, size=num_samples)
    gps_z = true_z + np.random.normal(0.0, 0.40, size=num_samples)
    gps_vx = vx + np.random.normal(0.0, 0.1, size=num_samples)
    gps_vy = vy + np.random.normal(0.0, 0.1, size=num_samples)

    imu_ax = np.gradient(vx, dt) + np.random.normal(0.0, 0.05, size=num_samples)
    imu_ay = np.gradient(vy, dt) + np.random.normal(0.0, 0.05, size=num_samples)
    imu_az = 9.80665 + np.gradient(vz, dt) + np.random.normal(0.0, 0.05, size=num_samples)
    imu_gz = np.random.normal(0.0, 0.02, size=num_samples)

    lidar_z = true_z + np.random.normal(0.0, 0.03, size=num_samples)
    vision_x = true_x + np.random.normal(0.0, 0.04, size=num_samples)
    vision_y = true_y + np.random.normal(0.0, 0.04, size=num_samples)
    vision_z = true_z + np.random.normal(0.0, 0.04, size=num_samples)

    # Inject Cyber Attack based on class_id
    if class_id == 1:  # GPS Spoofing
        # Inject ramp drift and step offsets
        drift_ramp = np.linspace(0.0, 25.0, num_samples)
        gps_x += drift_ramp * np.random.choice([1.0, -1.0])
        gps_y += (drift_ramp * 0.7) * np.random.choice([1.0, -1.0])
        gps_vx += np.random.normal(1.5, 0.5, size=num_samples)

    elif class_id == 2:  # IMU Manipulation
        # Constant bias and noise corruption
        bias_x = np.random.uniform(2.0, 5.0)
        bias_y = np.random.uniform(1.5, 4.0)
        imu_ax += bias_x + np.random.normal(0.0, 0.4, size=num_samples)
        imu_ay += bias_y + np.random.normal(0.0, 0.4, size=num_samples)
        imu_gz += np.random.normal(0.5, 0.2, size=num_samples)

    elif class_id == 3:  # LiDAR Corruption
        # Altitude drop / range manipulation
        corrupted_offset = np.random.uniform(5.0, 15.0)
        lidar_z -= corrupted_offset + np.random.normal(0.0, 0.3, size=num_samples)

    # Feature Extraction (13 features)
    features = []
    window = 10
    consecutive_alarm = 0
    trust_score = 1.0

    for i in range(num_samples):
        # 1. GPS Position Residual Norm against vision reference
        pos_err = np.sqrt((gps_x[i] - vision_x[i])**2 + (gps_y[i] - vision_y[i])**2 + (gps_z[i] - vision_z[i])**2)
        # 2. GPS NIS
        gps_nis = float((pos_err / 0.3)**2)
        # 3. GPS Velocity Residual Norm
        vel_err = float(np.sqrt((gps_vx[i] - vx[i])**2 + (gps_vy[i] - vy[i])**2))

        # 4. IMU Accel Norm Difference from gravity + expected kinematics
        accel_norm = np.sqrt(imu_ax[i]**2 + imu_ay[i]**2 + imu_az[i]**2)
        imu_accel_norm_diff = float(abs(accel_norm - 9.80665))

        # 5. Rolling IMU variances
        w_start = max(0, i - window)
        imu_accel_var = float(np.var(np.sqrt(imu_ax[w_start:i+1]**2 + imu_ay[w_start:i+1]**2 + imu_az[w_start:i+1]**2)) if i > 0 else 0.01)
        imu_gyro_var = float(np.var(imu_gz[w_start:i+1]) if i > 0 else 0.005)

        # 6. LiDAR NIS & Z residual
        lidar_err_z = float(abs(lidar_z[i] - vision_z[i]))
        lidar_nis = float((lidar_err_z / 0.05)**2)

        # 7. Discrepancies between sensors
        vis_gps_disc = pos_err
        vis_lidar_disc = lidar_err_z

        # 8. Accel vs GPS velocity derivative consistency
        if i > 0:
            dv_dt_x = (gps_vx[i] - gps_vx[i-1]) / dt
            dv_dt_y = (gps_vy[i] - gps_vy[i-1]) / dt
            accel_kinematic_consistency = float(np.sqrt((imu_ax[i] - dv_dt_x)**2 + (imu_ay[i] - dv_dt_y)**2))
        else:
            accel_kinematic_consistency = 0.1

        # 9. Consecutive anomaly tracking & trust score
        is_sample_anomaly = (gps_nis > 16.27) or (imu_accel_norm_diff > 1.5) or (lidar_nis > 10.83)
        if is_sample_anomaly:
            consecutive_alarm += 1
            trust_score = max(0.0, trust_score - 0.15)
        else:
            consecutive_alarm = max(0, consecutive_alarm - 1)
            trust_score = min(1.0, trust_score + 0.05)

        features.append({
            "gps_nis": gps_nis,
            "gps_pos_residual_norm": pos_err,
            "gps_vel_residual_norm": vel_err,
            "imu_accel_norm_diff": imu_accel_norm_diff,
            "imu_accel_var": imu_accel_var,
            "imu_gyro_var": imu_gyro_var,
            "lidar_nis": lidar_nis,
            "lidar_residual_z": lidar_err_z,
            "vision_gps_discrepancy": vis_gps_disc,
            "vision_lidar_z_discrepancy": vis_lidar_disc,
            "accel_kinematic_consistency": accel_kinematic_consistency,
            "consecutive_anomalies": consecutive_alarm,
            "sensor_trust_composite": trust_score,
            "label": class_id,
            "label_name": LABEL_NAMES[class_id]
        })

    return pd.DataFrame(features)


def generate_full_dataset(samples_per_class: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generates a balanced dataset across all 4 flight classes."""
    dfs = []
    for cid in range(4):
        print(f"[*] Generating {samples_per_class} samples for class {cid} ({LABEL_NAMES[cid]})...")
        df_c = simulate_scenario(class_id=cid, num_samples=samples_per_class, seed=seed + cid * 100)
        dfs.append(df_c)

    full_df = pd.concat(dfs, ignore_index=True)
    # Shuffle
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return full_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cyber-attack flight dataset.")
    parser.add_argument("--samples-per-class", type=int, default=3000, help="Number of samples per attack class.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8).")
    parser.add_argument("--output-dir", type=str, default="ml/data", help="Output directory for datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = generate_full_dataset(samples_per_class=args.samples_per_class, seed=args.seed)

    # Chronological/stratified split
    split_idx = int(len(full_df) * args.train_ratio)
    train_df = full_df.iloc[:split_idx].copy()
    test_df = full_df.iloc[split_idx:].copy()

    train_path = out_dir / "train_dataset.csv"
    test_path = out_dir / "test_dataset.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 50)
    print("      DATASET GENERATION COMPLETED")
    print("=" * 50)
    print(f"Total Samples: {len(full_df)}")
    print(f"Train Samples: {len(train_df)} -> {train_path}")
    print(f"Test Samples:  {len(test_df)}  -> {test_path}")
    print(f"Class Distribution:\n{full_df['label_name'].value_counts()}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
