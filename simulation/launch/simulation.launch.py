"""
ROS 2 Launch script for Cyber-Resilient Autonomous Drone Simulation Stack.
Brings up MicroXRCEAgent, Gazebo simulation world, and cyber-resilience node pipeline.
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    workspace_dir = Path(__file__).resolve().parent.parent.parent
    world_path = str(workspace_dir / "simulation" / "worlds" / "drone_cyber_world.sdf")

    # Declare arguments
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="true",
        description="Run Gazebo in headless mode (no GUI window)"
    )

    port_arg = DeclareLaunchArgument(
        "dds_port",
        default_value="8888",
        description="UDP port for Micro-XRCE-DDS Agent"
    )

    # 1. Micro-XRCE-DDS-Agent process (PX4 to ROS 2 bridge)
    xrce_agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", LaunchConfiguration("dds_port")],
        output="screen"
    )

    # 2. Gazebo Simulator process
    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_path],
        output="screen"
    )

    # 3. ROS 2 Sensor Bridge Node
    sensor_bridge_node = Node(
        package="sensor_bridge",
        executable="sensor_bridge_node",
        name="sensor_bridge_node",
        output="screen"
    )

    # 4. State Estimator Node (10-DOF EKF)
    state_estimator_node = Node(
        package="state_estimator",
        executable="state_estimator_node",
        name="state_estimator_node",
        output="screen"
    )

    # 5. Residual Detection Node
    residual_detector_node = Node(
        package="residual_detector",
        executable="residual_detector_node",
        name="residual_detector_node",
        output="screen"
    )

    # 6. Resilience Manager Node
    resilience_manager_node = Node(
        package="resilience_manager",
        executable="resilience_manager_node",
        name="resilience_manager_node",
        output="screen"
    )

    # 7. Path Planner Node (A*)
    path_planner_node = Node(
        package="path_planner",
        executable="path_planner_node",
        name="path_planner_node",
        output="screen"
    )

    # 8. Machine Learning Attack Detector Node
    ml_detector_node = Node(
        package="ml_detector",
        executable="ml_detector_node",
        name="ml_detector_node",
        output="screen"
    )

    # 9. PX4 Offboard Controller Node
    px4_controller_node = Node(
        package="px4_controller",
        executable="px4_controller_node",
        name="px4_controller_node",
        output="screen"
    )

    return LaunchDescription([
        headless_arg,
        port_arg,
        xrce_agent,
        gazebo,
        sensor_bridge_node,
        state_estimator_node,
        residual_detector_node,
        ml_detector_node,
        resilience_manager_node,
        path_planner_node,
        px4_controller_node
    ])

