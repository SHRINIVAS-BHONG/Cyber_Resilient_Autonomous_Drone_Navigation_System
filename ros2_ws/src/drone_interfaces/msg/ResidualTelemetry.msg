# ResidualTelemetry.msg
# Real-time state estimator innovation and residual telemetry for cyber attack detection

builtin_interfaces/Time timestamp
string sensor_name
float32[] innovation
float32[] normalized_residual
float32 nis
float32 threshold
bool anomaly_flag
