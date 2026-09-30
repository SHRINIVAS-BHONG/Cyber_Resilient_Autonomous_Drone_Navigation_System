@echo off
echo ==========================================================
echo  Cyber-Resilient Autonomous Drone Navigation System
echo  Automated End-to-End Suite Execution (Windows)
echo ==========================================================

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo [1/4] Running Unit and Scenario Tests...
pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pytest failed!
    exit /b %ERRORLEVEL%
)

echo [2/4] Running Baseline Clean Flight (S0)...
python tools\run_scenario.py --scenario s0_normal --defense on --seed 42

echo [3/4] Running Attack Demonstration (S1: GPS Drift)...
python tools\run_scenario.py --scenario s1_gps_drift --defense off --seed 42
python tools\run_scenario.py --scenario s1_gps_drift --defense on --seed 42

echo [4/4] Executing Comparative Batch Scenarios...
python tools\run_batch.py --seeds 42 101 --scenarios s0_normal s1_gps_drift s2_gps_jump s5_comm_disrupt

echo ==========================================================
echo  All runs and validations completed!
echo  Launch interactive dashboard with:
echo  streamlit run dashboard\app.py
echo ==========================================================
