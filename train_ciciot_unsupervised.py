"""Unsupervised Network Anomaly Detection on CICIoT2023 Dataset.

This script implements a zero-day anomaly detection pipeline using the highly-complex
CICIoT2023 dataset. Instead of using labels during training, models are trained
exclusively on benign (normal) traffic to learn its boundaries. 

During testing, the models attempt to detect 33 distinct classes of attacks 
as anomalies based purely on their deviation from the learned benign profile.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelResult:
    name: str
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("CSV"),
        help="Path to the directory containing the CICIoT2023 class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/Test_Metrics"),
        help="Directory where metrics and predictions are written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and model initialization.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly fraction for Isolation Forest.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.95,
        help="Benign quantile used to convert anomaly scores into binary labels.",
    )
    parser.add_argument(
        "--max-rows-per-class",
        type=int,
        default=20000,
        help="Maximum number of rows to load per class. Set to 0 for no limit.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the Isolation Forest.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Number of neighbors for Local Outlier Factor.",
    )
    return parser.parse_args()


def load_ciciot_data(data_dir: Path, max_rows_per_class: int) -> pd.DataFrame:
    """Load data from the CICIoT2023 folder structure and map to binary target.
    
    Benign_Final -> 0
    Everything else -> 1
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    log(f"Loading CICIoT2023 data from {data_dir} (max {max_rows_per_class} rows per class)...")
    
    all_data = []
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not class_folders:
        raise ValueError(f"No class subdirectories found in {data_dir}")
        
    for i, folder in enumerate(class_folders, 1):
        class_name = folder.name
        csv_files = sorted(folder.glob("*.csv"))
        
        if not csv_files:
            continue
            
        class_df_list = []
        rows_loaded = 0
        
        for csv_file in csv_files:
            if max_rows_per_class > 0 and rows_loaded >= max_rows_per_class:
                break
                
            rows_to_read = (max_rows_per_class - rows_loaded) if max_rows_per_class > 0 else None
            try:
                df = pd.read_csv(csv_file, nrows=rows_to_read)
                class_df_list.append(df)
                rows_loaded += len(df)
            except Exception as e:
                log(f"    Error reading {csv_file.name}: {e}")
                
        if class_df_list:
            combined_class_df = pd.concat(class_df_list, ignore_index=True)
            # Binary mapping
            binary_label = 0 if "Benign_Final" in class_name else 1
            combined_class_df["label"] = binary_label
            # Optional: keep original class name for detailed analysis
            combined_class_df["original_class"] = class_name
            all_data.append(combined_class_df)
            log(f"  [{i}/{len(class_folders)}] {class_name}: {len(combined_class_df)} rows (Mapped to {binary_label})")

    if not all_data:
        raise ValueError("Failed to load any data.")
        
    final_df = pd.concat(all_data, ignore_index=True)
    log(f"Total dataset loaded: {len(final_df)} rows, {len(final_df.columns)} columns")
    return final_df


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_unsupervised(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Train preprocessor ONLY on benign data, then transform train/test."""
    benign_mask = y_train.to_numpy() == 0
    x_train_benign = x_train[benign_mask]
    
    preprocessor = build_preprocessor()
    preprocessor.fit(x_train_benign)
    
    x_train_benign_transformed = preprocessor.transform(x_train_benign)
    x_test_transformed = preprocessor.transform(x_test)

    return (
        x_train_benign_transformed,
        x_test_transformed,
        preprocessor,
    )


def evaluate_scores(
    y_true: np.ndarray,
    score_values: np.ndarray,
    predicted_labels: np.ndarray,
    model_name: str,
) -> ModelResult:
    return ModelResult(
        name=model_name,
        roc_auc=roc_auc_score(y_true, score_values),
        average_precision=average_precision_score(y_true, score_values),
        precision=precision_score(y_true, predicted_labels, zero_division=0),
        recall=recall_score(y_true, predicted_labels, zero_division=0),
        f1=f1_score(y_true, predicted_labels, zero_division=0),
    )


def threshold_by_benign_quantile(
    anomaly_scores: np.ndarray,
    benign_reference_scores: np.ndarray,
    quantile: float,
) -> np.ndarray:
    threshold = np.quantile(benign_reference_scores, quantile)
    return (anomaly_scores >= threshold).astype(int)


def fit_isolation_forest(
    benign_train: np.ndarray,
    x_test: np.ndarray,
    y_test: pd.Series,
    contamination: float,
    prediction_quantile: float,
    random_state: int,
    n_estimators: int = 300,
) -> tuple[ModelResult, np.ndarray]:
    print("Fitting IsolationForest...")
    t0 = time.perf_counter()

    detector = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    detector.fit(benign_train)

    train_scores = -detector.decision_function(benign_train)
    test_scores = -detector.decision_function(x_test)
    predicted_labels = threshold_by_benign_quantile(test_scores, train_scores, prediction_quantile)
    result = evaluate_scores(y_test.to_numpy(), test_scores, predicted_labels, "IsolationForest")
    
    elapsed = time.perf_counter() - t0
    print(f"IsolationForest complete in {elapsed:.1f}s")
    return result, test_scores


def fit_pca_reconstruction_model(
    benign_train: np.ndarray,
    x_test: np.ndarray,
    y_test: pd.Series,
    prediction_quantile: float,
) -> tuple[ModelResult, np.ndarray]:
    print("Fitting PCA reconstruction anomaly detector...")
    t0 = time.perf_counter()

    n_components = max(1, min(20, benign_train.shape[1], max(1, benign_train.shape[0] - 1)))
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(benign_train)

    benign_reconstruction = pca.inverse_transform(pca.transform(benign_train))
    benign_scores = np.mean((benign_train - benign_reconstruction) ** 2, axis=1)

    test_reconstruction = pca.inverse_transform(pca.transform(x_test))
    test_scores = np.mean((x_test - test_reconstruction) ** 2, axis=1)
    predicted_labels = threshold_by_benign_quantile(test_scores, benign_scores, prediction_quantile)

    result = evaluate_scores(y_test.to_numpy(), test_scores, predicted_labels, "PCAReconstruction")
    elapsed = time.perf_counter() - t0
    print(f"PCA reconstruction complete in {elapsed:.1f}s")
    return result, test_scores


def fit_one_class_svm(
    benign_train: np.ndarray,
    x_test: np.ndarray,
    y_test: pd.Series,
    prediction_quantile: float,
    random_state: int,
) -> tuple[ModelResult, np.ndarray]:
    print("Fitting One-Class SVM (SGDOneClassSVM)...")
    t0 = time.perf_counter()
    
    detector = SGDOneClassSVM(random_state=random_state)
    detector.fit(benign_train)
    
    train_scores = detector.decision_function(benign_train)
    train_anomaly_scores = -train_scores
    test_scores = detector.decision_function(x_test)
    test_anomaly_scores = -test_scores
    
    predicted_labels = threshold_by_benign_quantile(test_anomaly_scores, train_anomaly_scores, prediction_quantile)
    result = evaluate_scores(y_test.to_numpy(), test_anomaly_scores, predicted_labels, "OneClassSVM")
    elapsed = time.perf_counter() - t0
    print(f"One-Class SVM complete in {elapsed:.1f}s")
    return result, test_anomaly_scores


def fit_local_outlier_factor(
    benign_train: np.ndarray,
    x_test: np.ndarray,
    y_test: pd.Series,
    prediction_quantile: float,
    n_neighbors: int = 20,
) -> tuple[ModelResult, np.ndarray]:
    print("Fitting Local Outlier Factor (LOF)...")
    t0 = time.perf_counter()
    
    detector = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, algorithm="ball_tree", n_jobs=-1)
    detector.fit(benign_train)
    
    train_scores = detector.decision_function(benign_train)
    train_anomaly_scores = -train_scores
    test_scores = detector.decision_function(x_test)
    test_anomaly_scores = -test_scores
    
    predicted_labels = threshold_by_benign_quantile(test_anomaly_scores, train_anomaly_scores, prediction_quantile)
    result = evaluate_scores(y_test.to_numpy(), test_anomaly_scores, predicted_labels, "LocalOutlierFactor")
    elapsed = time.perf_counter() - t0
    print(f"Local Outlier Factor complete in {elapsed:.1f}s")
    return result, test_anomaly_scores


def write_outputs(output_dir: Path, results: Sequence[ModelResult], y_test: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame([result.__dict__ for result in results])
    results_frame = results_frame.sort_values(by="roc_auc", ascending=False)
    results_frame.to_csv(output_dir / "unsupervised_metrics.csv", index=False)

    (output_dir / "unsupervised_metrics.json").write_text(
        json.dumps(results_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    predictions = pd.DataFrame({"y_true": y_test})
    for name, score_values in scores.items():
        predictions[f"{name}_score"] = score_values
    predictions.to_csv(output_dir / "test_scores.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_frame = load_ciciot_data(args.data_dir, args.max_rows_per_class)
    
    y_raw = raw_frame["label"]
    original_classes = raw_frame["original_class"]
    x_raw = raw_frame.drop(columns=["label", "original_class"])
    
    x_raw.columns = x_raw.columns.str.replace(r'[<\[\]]', '', regex=True)
    
    log("Converting features to numeric types...")
    for col in x_raw.columns:
        x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce")
        
    x_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    x_train, x_test, y_train, y_test, class_train, class_test = train_test_split(
        x_raw, y_raw, original_classes, test_size=args.test_size, random_state=args.random_state, stratify=y_raw
    )
    
    print("\n=== Class Distribution ===")
    print(f"Original Training set (Mixed):\n{y_train.value_counts().to_dict()}")
    print(f"Test set (Mixed):\n{y_test.value_counts().to_dict()}")

    print("\nFiltering training set to BENIGN traffic only for Unsupervised Learning...")
    benign_train, test_transformed, preprocessor = preprocess_unsupervised(
        x_train,
        y_train,
        x_test,
    )
    
    results: list[ModelResult] = []
    score_registry: dict[str, np.ndarray] = {}

    print("\n=== Training Unsupervised Models ===")

    isolation_result, isolation_scores = fit_isolation_forest(
        benign_train,
        test_transformed,
        y_test,
        contamination=args.contamination,
        prediction_quantile=args.prediction_threshold,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    results.append(isolation_result)
    score_registry[isolation_result.name] = isolation_scores

    pca_result, pca_scores = fit_pca_reconstruction_model(
        benign_train,
        test_transformed,
        y_test,
        prediction_quantile=args.prediction_threshold,
    )
    results.append(pca_result)
    score_registry[pca_result.name] = pca_scores

    ocsvm_result, ocsvm_scores = fit_one_class_svm(
        benign_train,
        test_transformed,
        y_test,
        prediction_quantile=args.prediction_threshold,
        random_state=args.random_state,
    )
    results.append(ocsvm_result)
    score_registry[ocsvm_result.name] = ocsvm_scores

    lof_result, lof_scores = fit_local_outlier_factor(
        benign_train,
        test_transformed,
        y_test,
        prediction_quantile=args.prediction_threshold,
        n_neighbors=args.n_neighbors,
    )
    results.append(lof_result)
    score_registry[lof_result.name] = lof_scores

    write_outputs(args.output_dir, results, y_test.to_numpy(), score_registry)
    
    # Save test original classes so we can do per-class evaluation later
    class_test.to_csv(args.output_dir / "test_original_classes.csv", index=False)

    summary = pd.DataFrame([result.__dict__ for result in results]).sort_values(by="roc_auc", ascending=False)
    
    print("\n" + "="*80)
    print("UNSUPERVISED MODEL COMPARISON (Ranked by ROC-AUC)")
    print("="*80 + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_name = summary.iloc[0]["name"]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_name} (ROC-AUC: {summary.iloc[0]['roc_auc']:.4f})")
    print(f"{'='*80}\n")
    
    print("Classification Report (Held-Out Test Set):\n")
    best_scores = score_registry[best_name]
    best_predictions = (best_scores >= np.quantile(score_registry[best_name][y_test == 0], args.prediction_threshold)).astype(int)
    print(
        classification_report(
            y_test.to_numpy(),
            best_predictions,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        )
    )

if __name__ == "__main__":
    main()
