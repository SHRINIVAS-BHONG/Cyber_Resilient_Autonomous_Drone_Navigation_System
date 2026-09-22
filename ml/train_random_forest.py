"""
Cyber-Attack Random Forest Classifier Training Pipeline.

Follows strict ML Best Practices:
1. Loads training dataset (ml/data/train_dataset.csv)
2. Checks and validates data hygiene (NaN/Inf handling)
3. Establishes a simple baseline model (Logistic Regression)
4. Fits StandardScaler strictly on training split
5. Trains balanced multi-class Random Forest classifier
6. Performs 5-Fold Stratified Cross-Validation
7. Saves artifacts to ml/models/:
   - random_forest_detector.joblib
   - scaler.joblib
   - baseline_model.joblib
   - feature_names.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Add core path
root_dir = Path(__file__).resolve().parent.parent

FEATURE_COLUMNS = [
    "gps_nis",
    "gps_pos_residual_norm",
    "gps_vel_residual_norm",
    "imu_accel_norm_diff",
    "imu_accel_var",
    "imu_gyro_var",
    "lidar_nis",
    "lidar_residual_z",
    "vision_gps_discrepancy",
    "vision_lidar_z_discrepancy",
    "accel_kinematic_consistency",
    "consecutive_anomalies",
    "sensor_trust_composite"
]


def train():
    data_dir = root_dir / "ml" / "data"
    train_file = data_dir / "train_dataset.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_file}. Run generate_dataset.py first.")

    print(f"[*] Loading training dataset from: {train_file}")
    df = pd.read_csv(train_file)

    # 1. Data hygiene check
    missing_counts = df[FEATURE_COLUMNS].isna().sum().sum()
    if missing_counts > 0:
        print(f"[!] Warning: Found {missing_counts} missing values. Imputing with median.")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    # Replace any infinite values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    print(f"[*] Total Training Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"[*] Class Counts: {dict(pd.Series(y).value_counts())}")

    # 2. Strict Train/Validation Split (80/20) BEFORE feature scaling
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Fit scaler strictly on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Baseline Model (Logistic Regression)
    print("\n[*] Training Baseline Model (Logistic Regression)...")
    baseline = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    baseline.fit(X_train_scaled, y_train)
    y_pred_base = baseline.predict(X_val_scaled)
    base_f1 = f1_score(y_val, y_pred_base, average="macro")
    print(f"[+] Baseline Validation Macro F1: {base_f1:.4f}")

    # 5. Production Model (Balanced Random Forest)
    print("\n[*] Training Production Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_val_scaled)
    rf_f1 = f1_score(y_val, y_pred_rf, average="macro")
    print(f"[+] Random Forest Validation Macro F1: {rf_f1:.4f} (Improvement over baseline: +{(rf_f1 - base_f1)*100:.2f}%)")

    # 6. Stratified 5-Fold Cross-Validation on Full Training Set
    print("\n[*] Running 5-Fold Stratified Cross-Validation on Full Training Set...")
    X_scaled_full = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_scaled_full, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[+] 5-Fold Cross-Validation Macro F1 Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"[+] Mean CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 7. Final Model Retraining on Full Scaled Training Data
    rf.fit(X_scaled_full, y)

    # 8. Save Artifacts to ml/models/
    models_dir = root_dir / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rf_path = models_dir / "random_forest_detector.joblib"
    scaler_path = models_dir / "scaler.joblib"
    baseline_path = models_dir / "baseline_model.joblib"
    feat_path = models_dir / "feature_names.json"

    joblib.dump(rf, rf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(baseline, baseline_path)

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print("\n" + "=" * 55)
    print("       MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print(f"Random Forest Model: {rf_path}")
    print(f"Feature Scaler:      {scaler_path}")
    print(f"Baseline Model:      {baseline_path}")
    print(f"Features JSON:       {feat_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train()
