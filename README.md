# Predictive Maintenance Using Machine Learning

**Author:** Kevin Danh  
**Program:** Master of Data Science – Bellevue University  
**Course:** DSC 680: Applied Data Science  

## Overview
This project develops a machine learning model to predict rare machine failures using sensor data. The goal is to enable proactive maintenance and reduce unexpected downtime.

---

## Business Problem
Machine failures are rare but costly. Traditional maintenance strategies are inefficient, either missing failures or performing unnecessary servicing. This project aims to detect early warning signs of failure to support better maintenance decisions.

---

## Dataset
- Source: Kaggle Predictive Maintenance Dataset  
- ~124,000 observations with only 106 failure cases (~0.085%)  
- Features include:
  - Sensor metrics (metric1–metric9)
  - Timestamp (date)
  - Device ID  

⚠️ Key challenge: **extreme class imbalance**

---

## Methods

### Feature Engineering
- Lag features (previous values)
- Rolling averages (short-term trends)
- Log transformations for skewed variables

### Models Tested
- Logistic Regression
- Random Forest
- XGBoost (final model)

### Handling Imbalance
- `class_weight='balanced'`
- `scale_pos_weight` for XGBoost

### Evaluation Metrics
- Recall (primary focus)
- PR-AUC (important for rare events)
- ROC-AUC

---

## Results

| Model               | ROC-AUC | PR-AUC  | Notes |
|--------------------|--------|--------|------|
| Logistic Regression| 0.68   | 0.004  | High recall, many false positives |
| Random Forest      | 0.70   | 0.022  | Failed to detect failures |
| XGBoost            | 0.76   | 0.0175 | Best overall performance |

### Final Model (XGBoost + Threshold Tuning)
- Threshold: **0.01**
- Recall: ~24%
- Precision: ~3%

👉 The model detects some failures but produces false positives  
👉 Tradeoff is acceptable because missing failures is more costly

---

## Key Insights
- Temporal features (lag & rolling) are critical
- Failure patterns are difficult to separate from normal behavior
- Precision–recall tradeoff is unavoidable in imbalanced problems

---

## Tools Used
- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---

## Project Structure

```text
predictive-maintenance-ml/
├── predictive_maintenance_ml.ipynb     # Main analysis notebook
├── predictive_maintenance_dataset.csv  # Dataset
└── README.md
```

---

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
3. Open `predictive_maintenance_ml.ipynb` in Jupyter and run all cells

---

## Future Improvements
- Collect more failure data
- Apply anomaly detection methods
- Deploy real-time monitoring system
- Add an AI-enhanced explanation layer (see [predictive-maintenance-genai](https://github.com/Cronosus99/predictive-maintenance-genai))

---

## References
Kaggle. (2023). Predictive maintenance dataset.  
https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset
