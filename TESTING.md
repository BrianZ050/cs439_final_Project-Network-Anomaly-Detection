# CICIoT2023 Intrusion Detection - Testing & Evaluation Guide

This document explains how to test and interpret the results of the Unsupervised Machine Learning pipeline for the CICIoT2023 dataset.

## Running the Pipeline

### 1. Train and Evaluate Models
Execute the core unsupervised pipeline. This automatically loads the `CSV/` directory, filters for benign data to train on, and evaluates the models against a mixed test set containing 33 unseen zero-day attacks.

```powershell
.venv\Scripts\python.exe train_ciciot_unsupervised.py --data-dir CSV --output-dir artifacts/ciciot_unsupervised_run
```

*Note: The default is **20,000 rows per class**, which is safe for 16 GB of RAM and finishes in **under 1 minute**. You can override this with `--max-rows-per-class <N>`. Set it to `0` to load the **entire dataset** with no limit. Some rough timing estimates on a 16 GB machine:*

| `--max-rows-per-class` | Approx. Runtime | RAM Usage |
|---|---|---|
| `5000` (smoke test) | ~10 seconds | Low |
| `20000` (default) | ~1 minute | Safe |
| `50000` | ~5 minutes | Moderate |
| `220000` | ~1 hour | High |
| `500000` | ~5 hours | Very High |
| `0` (no limit, ~47M rows) | Days | Will crash on 16 GB |

*You can also experiment with model hyperparameters by passing `--n-estimators` (for Isolation Forest, default 300) and `--n-neighbors` (for Local Outlier Factor, default 20).*

### 2. Generate Visual Plots
After a successful training run, run the visualization script to generate publication-ready Matplotlib/Seaborn graphs:

```powershell
.venv\Scripts\python.exe artifacts/plots/generate_visualizations.py
```

These plots will be saved to `artifacts/plots/` as `.png` files.

---

## Interpreting the Metrics (CSV Results)

After training, you can view the `unsupervised_metrics.csv` file. Because this is an Unsupervised problem with massive class imbalance, we prioritize different metrics than standard supervised classification.

### 1. ROC-AUC (Area Under ROC Curve)
- **What it measures**: The model's ability to rank anomalous/malicious flows with a higher anomaly score than benign flows.
- **Why it is the Primary Metric**: ROC-AUC is threshold-agnostic. It proves that the underlying mathematics of the model successfully separated the classes, regardless of where we draw the binary line.
- **Interpretation**: 
  - `> 0.90`: Excellent. The model easily identifies the attacks as outliers. (e.g. Local Outlier Factor).
  - `~ 0.50`: The model failed completely and is guessing at random. (e.g. One-Class SVM).

### 2. Precision and Recall
- **Precision**: When the model flags a flow as an anomaly, how often is it actually an attack? (Helps measure false alarms).
- **Recall (Detection Rate)**: Out of all the real attacks in the test set, how many did the model successfully catch?
- **Trade-off**: Because we use the 95th percentile of benign training scores as our threshold, our models are naturally highly sensitive. 

---

## Interpreting the Visual Plots

The `generate_visualizations.py` script creates 4 critical graphs for your academic paper.

### 1. `detection_by_class.png` (The "Money Shot")
This bar chart breaks down the True Positive Rate (Recall) for each of the 33 distinct attack classes using the best-performing model (LOF).
- **What to look for**: Notice how volumetric and brute-force attacks (like `DDoS-UDP_Flood`, `DictionaryBruteForce`) have near 100% detection rates. 
- **The Academic Narrative**: The model struggles heavily with stealthy application-layer attacks (like `XSS` and `SqlInjection`) because they are single, small HTTP requests that look mathematically identical to normal web browsing. This proves the limitation of network flow statistics against Layer-7 attacks.

### 2. `score_distributions.png`
A density histogram showing the anomaly scores assigned to Benign vs Malicious traffic.
- **What to look for**: You want to see the green distribution (Benign) tightly clustered on the left, and the red distribution (Malicious) shoved far to the right, crossing the black dotted decision threshold. This visually proves the model "understands" the difference.

### 3. `roc_curves.png`
Overlays the ROC curves for all 4 models (LOF, Isolation Forest, PCA, One-Class SVM).
- **What to look for**: The curve that hugs the top-left corner is the best. The curve that rides the diagonal dotted line is the worst. This plot proves why Density-Based detection (LOF) is vastly superior to Boundary-Based detection (One-Class SVM) in this specific IoT environment.

### 4. `pr_curves.png`
Precision-Recall curves are highly sensitive to class imbalance.
- **What to look for**: A curve that stays high near the top-right indicates that the model can maintain high precision (few false alarms) even as it catches more anomalies (high recall).
