#!/bin/bash
set -e

echo "=========================================================="
echo " Cyber-Resilient Autonomous Drone Navigation System"
echo " Automated End-to-End Suite Execution"
echo "=========================================================="

echo "[1/4] Running Unit & Scenario Tests..."
pytest tests/ -v

echo "[2/4] Running Baseline Clean Flight (S0)..."
python3 tools/run_scenario.py --scenario s0_normal --defense on --seed 42

echo "[3/4] Running Attack Demonstration (S1: GPS Drift)..."
python3 tools/run_scenario.py --scenario s1_gps_drift --defense off --seed 42
python3 tools/run_scenario.py --scenario s1_gps_drift --defense on --seed 42

echo "[4/4] Executing Comparative Batch Scenarios..."
python3 tools/run_batch.py --seeds 42 101 --scenarios s0_normal s1_gps_drift s2_gps_jump s5_comm_disrupt

echo "=========================================================="
echo " Complete! Launch dashboard with:"
echo " streamlit run dashboard/app.py"
echo "=========================================================="
