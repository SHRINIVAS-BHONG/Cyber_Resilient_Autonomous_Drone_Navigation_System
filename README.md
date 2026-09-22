# Cyber-Resilient Autonomous Drone Navigation System

Technical System Design & Implementation Specification for a simulated autonomous drone resilient against navigation cyberattacks.

---

## 1. Project Goal
Build a fully simulated autonomous drone that can:
- **Navigate normally** using PX4 SITL, Gazebo, ROS 2, and onboard sensors.
- **Detect navigation cyberattacks** such as GPS spoofing, sensor manipulation, and communication disruption.
- **Isolate corrupted navigation data** immediately upon attack detection.
- **Switch to trusted sensor inputs** (IMU, Optical Flow, LiDAR/Vision).
- **Generate a safe alternative trajectory** to continue mission or land safely.
- **Recover normal navigation** after the attack ends.
- **Log measurable evidence** showing improved cyber-resilience.

> **Note:** All attack simulations take place strictly inside the PX4/Gazebo virtual simulation environment. Never test against real drones, physical systems, or live infrastructure.

---

## 2. System Architecture

```text
Gazebo World
│
├── Simulated GPS
├── IMU
├── Camera / Depth Sensor
├── LiDAR / Range Sensor
└── Wind and obstacle models
│
▼
PX4 SITL Autopilot
│
├── PX4 EKF2 State Estimator
├── Position Controller
├── Mission Manager
└── MAVLink / microRTPS / XRCE-DDS bridge
│
▼
ROS 2 DDS Network
│
├── Sensor Bridge (`sensor_bridge_node`)
├── Attack Simulator (`attack_injector_node`)
├── Cyber Monitor & Telemetry Logger (`telemetry_logger`)
│
▼
State Estimation & Detection Layer
│
├── Custom Extended Kalman Filter (`state_estimator_node`)
├── Innovation / NIS Residual Detector (`residual_detector_node`)
└── ML-based Multi-class Classifier (`ml_detector_node`)
│
▼
Resilience & Recovery Manager (`resilience_manager_node`)
│
├── Sensor Trust Scoring & Isolation
├── Degraded / Safe Mode Failsafe
└── Sensor Reintegration & Recovery Manager
│
▼
Path Planning Layer (`path_planner_node`)
│
├── Risk-aware A* Grid Planner (Phase 1)
└── 3D RRT* Dynamic Planner (Phase 2)
│
▼
PX4 Offboard Setpoints & Control
```

---

## 3. Core Software Components

### ROS 2 Nodes (`ros2_ws/src/`)
- **`sensor_bridge` (`sensor_bridge_node`)**: Ingests, normalizes, validates timestamps, and tags source IDs for all PX4 and Gazebo sensor streams (`/sensor/imu`, `/sensor/gps`, `/sensor/lidar`, etc.).
- **`attack_simulator` (`attack_injector_node`)**: Simulates realistic cyber threats:
  - *GPS Spoofing*: Slow drift, abrupt jump, false velocity, elevation bias.
  - *Sensor Manipulation*: IMU bias/noise injection, LiDAR range corruption, magnetometer distortion.
  - *Communication Disruption*: Packet drops, latency injection (e.g. 500ms+), burst loss, topic starvation.
- **`state_estimator` (`state_estimator_node`)**: Transparent Python/C++ EKF tracking 10-DOF state `[x, y, z, vx, vy, vz, ax_bias, ay_bias, az_bias, yaw]`, exposing residuals and innovation covariances ($S, NIS$).
- **`residual_detector` (`residual_detector_node`)**: First line of defense using Normalized Innovation Squared (NIS) hypothesis testing and sliding window hysteresis.
- **`ml_detector` (`ml_detector_node`)**: Multi-class Random Forest / XGBoost model predicting attack classifications (`Normal`, `GPS Spoofing`, `IMU Manipulation`, `LiDAR Corruption`, `DoS/Communication Disruption`).
- **`resilience_manager` (`resilience_manager_node`)**: Autonomous decision engine: manages sensor trust scores, isolates compromised sensors, falls back to optical/inertial odometry, and issues failsafe flight commands.
- **`path_planner` (`path_planner_node`)**: Calculates obstacle-free alternative trajectories taking cyber risk, sensor uncertainty, and safe landing zones into account.
- **`px4_controller` (`px4_controller_node`)**: Issues offboard position/velocity setpoints to the PX4 autopilot.
- **`telemetry_logger` (`telemetry_logger_node`)**: Records rosbags, residual telemetry, and state outputs for evaluation.

---

## 4. Repository Structure

```text
├── config/
│   ├── sensors.yaml
│   ├── estimator.yaml
│   ├── detector.yaml
│   ├── resilience.yaml
│   ├── planner.yaml
│   └── attacks/
│       ├── gps_spoofing.yaml
│       ├── imu_manipulation.yaml
│       ├── lidar_corruption.yaml
│       └── communication_disruption.yaml
├── docker/
│   ├── Dockerfile
│   └── compose.yaml
├── ml/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── train_random_forest.py
│   ├── evaluate.py
│   └── models/
├── reports/
│   ├── figures/
│   └── experiment_results/
├── ros2_ws/
│   └── src/
│       ├── attack_simulator/
│       ├── drone_interfaces/
│       ├── ml_detector/
│       ├── path_planner/
│       ├── px4_controller/
│       ├── residual_detector/
│       ├── resilience_manager/
│       ├── sensor_bridge/
│       ├── state_estimator/
│       └── telemetry_logger/
├── simulation/
│   ├── launch/
│   ├── models/
│   ├── scenarios/
│   └── worlds/
├── tests/
│   ├── integration/
│   ├── regression/
│   ├── scenario/
│   └── unit/
├── visualization/
│   ├── dashboard.py
│   ├── plot_telemetry.py
│   └── rviz/
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 5. Development Phases & Roadmap

1. **Phase 0: Environment Setup**: PX4 SITL + Gazebo + ROS 2 microXRCE-DDS bridge validation.
2. **Phase 1: Normal Navigation Baseline**: Waypoint missions, stable offboard control, and telemetry logging.
3. **Phase 2: Transparent Sensor Fusion**: EKF implementation with innovation and residual exposure.
4. **Phase 3: Attack Injection**: Reproducible GPS, IMU, and communication disruption scenarios via YAML configs.
5. **Phase 4: Residual Detection**: Statistical NIS-based threshold detection and state machine.
6. **Phase 5: Machine Learning Classifier**: Supervised classification of attack fingerprints.
7. **Phase 6: Resilience & Fail-Safe Manager**: Dynamic sensor isolation, trust weighting, and automatic degraded recovery.
8. **Phase 7: Cyber-Aware Safe Path Planning**: A* replanning under sensor uncertainty.
9. **Phase 8: Evaluation & Telemetry Dashboard**: Streamlit / Plotly interactive metrics and scenario comparisons.

---

## 6. Evaluation Metrics
- **Detection**: Accuracy, Precision, Recall, F1-Score, False Alarm Rate, Detection Latency.
- **Resilience**: Time to Detect (TTD), Time to Contain (TTC), Time to Recover (TTR), Safe Trajectory Completion Rate.
- **Flight Navigation**: Position RMSE, Velocity Drift, Maximum Cross-track Error, Emergency Landing Count.
