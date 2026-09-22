# SafeTrajectory.msg
# Alternative safe trajectory under cyber attack or degraded conditions

geometry_msgs/PoseStamped[] waypoints
float32 maximum_velocity
float32 estimated_risk
string planner
bool emergency
