"""
Metrics Evaluation Engine
Calculates Detection, Navigation, Resilience, and Mission metrics from run telemetry and events.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class MetricsEvaluator:
    @staticmethod
    def evaluate(
        telemetry_df: pd.DataFrame,
        events: List[Dict[str, Any]],
        scenario_cfg: Dict[str, Any],
        defense_on: bool,
        obstacles: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Computes all standard evaluation metrics from run data.
        """
        attack_cfg = scenario_cfg.get("attack", {})
        attack_enabled = bool(attack_cfg.get("enabled", False))
        attack_start = float(attack_cfg.get("start_time", 30.0))
        attack_end = attack_start + float(attack_cfg.get("duration", 20.0))
        goal_pos = np.array(scenario_cfg.get("mission", {}).get("goal_point", [60.0, 40.0, 10.0]))
        obstacles = obstacles or []

        # 1. Detection Metrics
        first_detected_event = next((e for e in events if e.get("event") in ["ATTACK_CONFIRMED", "SUSPICIOUS_ANOMALY_DETECTED"]), None)
        confirmed_event = next((e for e in events if e.get("event") == "ATTACK_CONFIRMED"), None)

        time_to_detect: Optional[float] = None
        false_alarms = 0
        missed_detection = False

        if attack_enabled:
            if confirmed_event:
                time_to_detect = round(confirmed_event["timestamp"] - attack_start, 3)
                if time_to_detect < 0: # Fired before attack started!
                    false_alarms += 1
                    time_to_detect = None
            else:
                missed_detection = True
        else:
            # Baseline clean run
            if confirmed_event or first_detected_event:
                false_alarms += 1

        # 2. Resilience Metrics
        isolated_event = next((e for e in events if "ISOLATED" in e.get("event", "") or "REJECTED" in e.get("event", "")), None)
        time_to_contain: Optional[float] = None
        if confirmed_event and isolated_event:
            time_to_contain = max(0.0, round(isolated_event["timestamp"] - confirmed_event["timestamp"], 3))

        recovered_event = next((e for e in events if e.get("event") == "SYSTEM_FULLY_RECOVERED"), None)
        time_to_recover: Optional[float] = None
        if recovered_event and attack_enabled:
            time_to_recover = max(0.0, round(recovered_event["timestamp"] - attack_end, 3))

        # Time in degraded mode
        time_in_degraded_sec = 0.0
        if "state" in telemetry_df.columns:
            degraded_mask = telemetry_df["state"].isin(["SUSPICIOUS", "ATTACK_CONFIRMED", "CONTAINMENT", "SAFE_NAVIGATION", "RECOVERY"])
            dt = 0.05
            if len(telemetry_df) > 1:
                dt = float(telemetry_df["time"].iloc[1] - telemetry_df["time"].iloc[0])
            time_in_degraded_sec = round(float(degraded_mask.sum() * dt), 2)

        sensors_rejected = list(set([
            e.get("details", {}).get("sensor", "gps")
            for e in events if "ISOLATED" in e.get("event", "") or "REJECTED" in e.get("event", "")
        ]))

        # 3. Navigation Metrics
        pos_errors = []
        for _, row in telemetry_df.iterrows():
            truth = np.array([row["truth_x"], row["truth_y"], row["truth_z"]])
            active_est = np.array([row["active_est_x"], row["active_est_y"], row["active_est_z"]])
            pos_errors.append(np.linalg.norm(truth - active_est))

        pos_errors = np.array(pos_errors) if pos_errors else np.array([0.0])
        rmse_error = round(float(np.sqrt(np.mean(pos_errors ** 2))), 3)
        max_error = round(float(np.max(pos_errors)), 3)

        # Final goal error
        last_row = telemetry_df.iloc[-1] if not telemetry_df.empty else None
        if last_row is not None:
            last_truth = np.array([last_row["truth_x"], last_row["truth_y"], last_row["truth_z"]])
            final_goal_error = round(float(np.linalg.norm(last_truth[:2] - goal_pos[:2])), 3)
        else:
            final_goal_error = 999.0

        # 4. Mission Metrics: Collisions and Completion
        collisions = 0
        min_obstacle_distance = 999.0
        for _, row in telemetry_df.iterrows():
            drone_xy = np.array([row["truth_x"], row["truth_y"]])
            for obs in obstacles:
                obs_xy = np.array([obs["x"], obs["y"]])
                dist = np.linalg.norm(drone_xy - obs_xy)
                clearance = dist - obs.get("radius", 3.0)
                min_obstacle_distance = min(min_obstacle_distance, clearance)
                if clearance <= 0.0:
                    collisions += 1

        mission_completed = (final_goal_error <= 3.5 and collisions == 0)
        emergency_landing = any("EMERGENCY" in e.get("event", "") for e in events)

        return {
            "scenario_id": scenario_cfg.get("id", "unknown"),
            "defense_on": defense_on,
            "detection": {
                "time_to_detect_s": time_to_detect,
                "false_alarms": false_alarms,
                "missed_detection": missed_detection
            },
            "navigation": {
                "position_rmse_m": rmse_error,
                "max_position_error_m": max_error,
                "final_goal_error_m": final_goal_error,
                "min_obstacle_clearance_m": round(max(0.0, min_obstacle_distance), 2)
            },
            "resilience": {
                "time_to_contain_s": time_to_contain,
                "time_to_recover_s": time_to_recover,
                "time_in_degraded_sec": time_in_degraded_sec,
                "sensors_rejected": sensors_rejected
            },
            "mission": {
                "completed": bool(mission_completed),
                "collisions": int(collisions),
                "emergency_landing": bool(emergency_landing),
                "status": "SUCCESS" if (mission_completed and collisions == 0) else "FAILED"
            }
        }
