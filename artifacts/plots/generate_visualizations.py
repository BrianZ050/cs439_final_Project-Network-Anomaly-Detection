"""Generate Visual Plots for CICIoT2023 Unsupervised Run."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def main():
    plots_dir = Path(__file__).parent
    run_dir = plots_dir.parent / "Test_Metrics"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {run_dir}...")
    scores_df = pd.read_csv(run_dir / "test_scores.csv")
    classes_df = pd.read_csv(run_dir / "test_original_classes.csv")
    
    y_true = scores_df["y_true"].values
    
    # Extract model names from columns (e.g., 'IsolationForest_score')
    model_columns = [col for col in scores_df.columns if col.endswith("_score")]
    
    # Use a modern, academic style
    plt.style.use('seaborn-v0_8-whitegrid')
    # If the above fails in older matplotlib versions, it will fall back to default,
    # but we can try setting some default parameters manually just in case.
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # ---------------------------------------------------------
    # 1. ROC Curves
    # ---------------------------------------------------------
    print("Generating ROC Curves...")
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(model_columns):
        model_name = col.replace("_score", "")
        scores = scores_df[col].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Benign misclassified as Attack)', fontsize=12)
    plt.ylabel('True Positive Rate (Attack correctly identified)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. Precision-Recall Curves
    # ---------------------------------------------------------
    print("Generating Precision-Recall Curves...")
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(model_columns):
        model_name = col.replace("_score", "")
        scores = scores_df[col].values
        precision, recall, _ = precision_recall_curve(y_true, scores)
        # PR AUC is approximately the area under this curve, or we can just plot it
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2, label=model_name)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "pr_curves.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. Anomaly Score Distributions (Best Model: LocalOutlierFactor)
    # ---------------------------------------------------------
    print("Generating Score Distributions...")
    best_model_col = "LocalOutlierFactor_score"
    if best_model_col in scores_df.columns:
        plt.figure(figsize=(12, 6))
        
        benign_scores = scores_df[y_true == 0][best_model_col]
        malicious_scores = scores_df[y_true == 1][best_model_col]
        
        # ---------------------------------------------------------
        # We apply a log transformation to the scores to handle extreme outliers
        # Shift scores so the minimum is 1 (to avoid log(<=0))
        # ---------------------------------------------------------
        min_score = min(benign_scores.min(), malicious_scores.min())
        shift = 1.0 - min_score if min_score <= 0 else 0.0
        
        benign_log = np.log10(benign_scores + shift)
        malicious_log = np.log10(malicious_scores + shift)
        
        threshold = np.percentile(benign_scores, 95)
        threshold_log = np.log10(threshold + shift)
        
        plt.hist(benign_log, bins=50, alpha=0.6, density=True, color='green', label='Benign Traffic')
        plt.hist(malicious_log, bins=50, alpha=0.6, density=True, color='red', label='Malicious Traffic')
        
        plt.axvline(x=threshold_log, color='black', linestyle='dashed', linewidth=2, label=f'Decision Threshold')
        
        plt.xlabel('Log10(Anomaly Score) (Higher = More Anomalous)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Anomaly Scores (Local Outlier Factor)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "score_distributions.png")
        plt.close()

    # ---------------------------------------------------------
    # 4. Detection Rate by Class (Best Model: LocalOutlierFactor)
    # ---------------------------------------------------------
    print("Generating Detection Rate by Class...")
    if best_model_col in scores_df.columns:
        # Merge scores with original classes
        df_eval = pd.DataFrame({
            "y_true": y_true,
            "score": scores_df[best_model_col],
            "attack_class": classes_df.iloc[:, 0].values  # Assuming first column is the class name
        })
        
        benign_scores = df_eval[df_eval["y_true"] == 0]["score"]
        threshold = np.percentile(benign_scores, 95)
        df_eval["predicted_anomaly"] = (df_eval["score"] >= threshold).astype(int)
        
        # Calculate recall (TPR) per malicious class
        malicious_df = df_eval[df_eval["y_true"] == 1]
        
        detection_rates = {}
        for attack_name, group in malicious_df.groupby("attack_class"):
            # TPR = Total predicted as anomaly / Total actual
            tpr = group["predicted_anomaly"].mean()
            detection_rates[attack_name] = tpr
            
        # Sort by detection rate
        sorted_rates = sorted(detection_rates.items(), key=lambda x: x[1], reverse=False)
        classes = [x[0] for x in sorted_rates]
        rates = [x[1] * 100 for x in sorted_rates] # convert to percentage
        
        plt.figure(figsize=(12, 14))
        
        # Color code: Green > 90%, Yellow > 50%, Red < 50%
        bar_colors = ['#d62728' if r < 50 else '#ff7f0e' if r < 90 else '#2ca02c' for r in rates]
        
        bars = plt.barh(classes, rates, color=bar_colors)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1 if width < 95 else width - 5
            align = 'left' if width < 95 else 'right'
            color = 'black' if width < 95 else 'white'
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                     va='center', ha=align, color=color, fontweight='bold', fontsize=9)
            
        plt.xlabel('Detection Rate (Recall) %', fontsize=12)
        plt.title('Zero-Day Detection Rate by Attack Class (Local Outlier Factor)', fontsize=14, fontweight='bold')
        plt.xlim(0, 105)
        
        # Add a vertical line for overall average
        avg_rate = np.mean(rates)
        plt.axvline(x=avg_rate, color='black', linestyle='--', alpha=0.7, label=f'Average Rate ({avg_rate:.1f}%)')
        plt.legend(loc='lower right')
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "detection_by_class.png")
        plt.close()

    print(f"All plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
