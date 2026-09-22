# AttackStatus.msg
# Cyber attack status and detection metadata

string attack_type
bool detected
float32 confidence
float32 risk_score
builtin_interfaces/Time start_time
string[] compromised_sensors
