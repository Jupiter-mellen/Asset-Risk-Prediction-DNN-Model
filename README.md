# Asset Risk Prediction — TensorFlow Deep Neural Network

## Overview
This project implements a predictive maintenance prototype that estimates the likelihood of asset failure from operational data. The goal is to produce ranked risk scores so maintenance teams can prioritise inspections and interventions. The model is a TensorFlow/Keras deep neural network trained on the AI4I 2020 Predictive Maintenance dataset.

---

## Data
**Dataset:** AI4I 2020 Predictive Maintenance (binary failure label)

Key fields used:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Machine type (categorical)
- Failure label (0 = no failure, 1 = failure)

**Engineered features:**
- `temp_diff_k` = Process temperature − Air temperature  
- `power_proxy` = Rotational speed × Torque  
- `wear_per_power` = Torque / (Rotational speed + 1e−6)

---

## Approach

### 1) Preparation
- Cleaned and standardised column names.
- One-hot encoded the `Type` categorical feature.
- Scaled numeric features with StandardScaler.

### 2) Train/Validation/Test split
- Stratified split: 64% train, 16% validation, 20% test.

### 3) Model
- Fully connected DNN (ReLU activations).
- Batch normalisation and dropout for regularisation.
- Sigmoid output for binary classification.
- Optimiser: Adam. Loss: Binary cross-entropy.
- Early stopping and ReduceLROnPlateau on validation loss.

### 4) Calibration
- Post-training probability calibration with Isotonic Regression using the validation set.
- Compared raw vs calibrated probabilities on the held-out test set.

---

## Results (test set)

| Metric | Raw model | Calibrated model |
| --- | ---: | ---: |
| ROC AUC | 0.886 | 0.868 |
| PR AUC (Average Precision) | 0.534 | 0.453 |

**Confusion matrix (threshold = 0.50)**

Raw model

[[1854 78]
[ 23 45]]

Calibrated model

[[1913 19]
[ 35 33]]

Classification report (calibrated, threshold = 0.50)

precision recall f1-score support
0 0.982 0.990 0.986 1932
1 0.635 0.485 0.550 68
accuracy 0.973 2000
macro avg 0.808 0.738 0.768 2000
weighted avg 0.970 0.973 0.971 2000


**Interpretation**
- The model separates classes well (high ROC AUC).
- Precision–recall is modest because failures are rare; class imbalance makes high precision at high recall challenging.
- Calibration reduces over-confidence and improves probability reliability, but can lower PR AUC because it smooths extreme scores. Whether to use calibrated or raw scores depends on the downstream decision policy.

---

## Visualisations

Precision–Recall Curve  
![Precision–Recall Curve](pr_curve.png)

ROC Curve  
![ROC Curve](roc_curve.png)

Calibration Curve  
![Calibration Curve](calibration_curve.png)

> If these images are not yet in your repo, save the figures from the notebook as:
> `pr_curve.png`, `roc_curve.png`, and `calibration_curve.png` in the project root (or update the paths above).

---

## How to run

```bash
# optional: create a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate

# install requirements
pip install -r requirements.txt

# launch notebook
jupyter notebook
```
Raw model

