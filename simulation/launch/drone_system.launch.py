import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    attack_type_arg = DeclareLaunchArgument('attack_type', default_value='gps_drift')
    start_time_arg = DeclareLaunchArgument('start_time', default_value='30.0')
    duration_arg = DeclareLaunchArgument('duration', default_value='20.0')

    return LaunchDescription([
        attack_type_arg,
        start_time_arg,
        duration_arg,

        # 1. Kinematic / Fake Drone simulator (50 Hz)
        Node(package='fake_drone', executable='fake_drone_node', name='fake_drone_node', output='screen'),

        # 2. Sensor Simulator
        Node(package='sensor_sim', executable='sensor_sim_node', name='sensor_sim_node', output='screen'),

        # 3. Attack Injector
        Node(
            package='attack_injector',
            executable='attack_injector_node',
            name='attack_injector_node',
            parameters=[{
                'attack_type': LaunchConfiguration('attack_type'),
                'start_time': LaunchConfiguration('start_time'),
                'duration': LaunchConfiguration('duration')
            }],
            output='screen'
        ),

        # 4. Main Estimator (GPS active)
        Node(
            package='estimator',
            executable='estimator_node',
            name='main_estimator_node',
            parameters=[{'is_trusted': False}],
            output='screen'
        ),

        # 5. Trusted Estimator (GPS permanently disabled)
        Node(
            package='estimator',
            executable='estimator_node',
            name='trusted_estimator_node',
            parameters=[{'is_trusted': True}],
            output='screen'
        ),

        # 6. Anomaly Detector (10 Hz)
        Node(package='detector', executable='detector_node', name='detector_node', output='screen'),

        # 7. Resilience Manager
        Node(package='resilience_manager', executable='resilience_manager_node', name='resilience_manager_node', output='screen'),

        # 8. Path Planner (A*)
        Node(package='path_planner', executable='planner_node', name='planner_node', output='screen'),

        # 9. Controller (20 Hz)
        Node(package='px4_controller', executable='controller_node', name='controller_node', output='screen'),

        # 10. Telemetry Logger
        Node(package='telemetry_logger', executable='logger_node', name='logger_node', output='screen')
    ])
