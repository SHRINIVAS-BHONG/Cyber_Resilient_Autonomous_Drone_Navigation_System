"""
Independent Evaluation Pipeline for Trained Cyber-Attack Classifiers.

Evaluates the trained Random Forest and baseline models on the held-out test dataset (ml/data/test_dataset.csv):
1. Calculates multi-class performance metrics (Accuracy, Precision, Recall, F1-Score).
2. Benchmarks inference latency per sample.
3. Generates Confusion Matrix plot -> reports/figures/confusion_matrix.png
4. Generates Feature Importance plot -> reports/figures/feature_importance.png
5. Exports comprehensive quantitative metrics -> reports/experiment_results/ml_metrics.json
"""

import json
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, precision_score, recall_score

root_dir = Path(__file__).resolve().parent.parent

LABEL_MAP = {
    0: "NORMAL",
    1: "GPS_SPOOFING",
    2: "IMU_MANIPULATION",
    3: "LIDAR_CORRUPTION"
}


def evaluate():
    data_dir = root_dir / "ml" / "data"
    models_dir = root_dir / "ml" / "models"
    figures_dir = root_dir / "reports" / "figures"
    results_dir = root_dir / "reports" / "experiment_results"

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    test_file = data_dir / "test_dataset.csv"
    rf_file = models_dir / "random_forest_detector.joblib"
    scaler_file = models_dir / "scaler.joblib"
    feat_file = models_dir / "feature_names.json"

    if not test_file.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_file}.")
    if not rf_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(f"Model artifacts not found in {models_dir}. Run train_random_forest.py first.")

    print(f"[*] Loading test dataset from: {test_file}")
    df_test = pd.read_csv(test_file)

    with open(feat_file, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    rf_model = joblib.load(rf_file)
    scaler = joblib.load(scaler_file)

    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # Preprocess with training scaler
    X_test_scaled = scaler.transform(X_test)

    # 1. Benchmark Inference Latency
    print("[*] Benchmarking inference latency over test set...")
    t0 = time.perf_counter()
    y_pred = rf_model.predict(X_test_scaled)
    total_time = time.perf_counter() - t0
    latency_ms_per_sample = (total_time / len(X_test)) * 1000.0

    # Probabilities
    y_probs = rf_model.predict_proba(X_test_scaled)

    # 2. Compute Performance Metrics
    accuracy = float(np.mean(y_pred == y_test))
    macro_precision = float(precision_score(y_test, y_pred, average="macro"))
    macro_recall = float(recall_score(y_test, y_pred, average="macro"))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    cls_report = classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in range(4)],
        output_dict=True
    )

    print("\n" + "=" * 65)
    print("       TEST SET CLASSIFICATION REPORT (RANDOM FOREST)")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=[LABEL_MAP[i] for i in range(4)]))
    print(f"Overall Test Accuracy:    {accuracy * 100:.2f}%")
    print(f"Macro Precision:          {macro_precision:.4f}")
    print(f"Macro Recall:             {macro_recall:.4f}")
    print(f"Macro F1-Score:           {macro_f1:.4f}")
    print(f"Inference Latency:        {latency_ms_per_sample:.3f} ms / sample ({1000.0 / latency_ms_per_sample:.1f} Hz throughput)")
    print("=" * 65)

    # 3. Generate & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[LABEL_MAP[i] for i in range(4)]
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Cyber-Attack Detection Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"[+] Saved Confusion Matrix to: {cm_path}")
    plt.close()

    # 4. Generate & Save Feature Importance Plot
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    ax.barh(range(len(feature_cols)), sorted_importances[::-1], color="#2b5c8f", align="center")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10)
    ax.set_title("Random Forest Feature Importance Ranking", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    feat_path = figures_dir / "feature_importance.png"
    plt.savefig(feat_path, dpi=300)
    print(f"[+] Saved Feature Importance to: {feat_path}")
    plt.close()

    # 5. Export JSON Evaluation Report
    metrics_summary = {
        "model_type": "RandomForestClassifier",
        "num_test_samples": int(len(X_test)),
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "inference_latency_ms": round(latency_ms_per_sample, 3),
        "throughput_hz": round(1000.0 / latency_ms_per_sample, 1),
        "per_class_metrics": {
            LABEL_MAP[i]: {
                "precision": round(cls_report[LABEL_MAP[i]]["precision"], 4),
                "recall": round(cls_report[LABEL_MAP[i]]["recall"], 4),
                "f1_score": round(cls_report[LABEL_MAP[i]]["f1-score"], 4),
                "support": int(cls_report[LABEL_MAP[i]]["support"])
            }
            for i in range(4)
        },
        "top_features": [
            {"feature": sorted_features[i], "importance": round(float(sorted_importances[i]), 4)}
            for i in range(len(sorted_features))
        ]
    }

    metrics_json_path = results_dir / "ml_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[+] Saved Quantitative ML Metrics to: {metrics_json_path}\n")


if __name__ == "__main__":
    evaluate()
