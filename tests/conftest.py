import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "sensor_bridge"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "state_estimator"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "residual_detector"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "resilience_manager"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "path_planner"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "ml_detector"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "px4_controller"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "attack_simulator"))
sys.path.insert(0, str(root_dir / "ros2_ws" / "src" / "telemetry_logger"))
