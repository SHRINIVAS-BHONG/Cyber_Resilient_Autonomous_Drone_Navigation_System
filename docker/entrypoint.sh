#!/bin/bash
set -e

# Source ROS 2 Humble installation
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
fi

# Source workspace if built
if [ -f "$WORKSPACE/ros2_ws/install/setup.bash" ]; then
    source "$WORKSPACE/ros2_ws/install/setup.bash"
fi

exec "$@"
