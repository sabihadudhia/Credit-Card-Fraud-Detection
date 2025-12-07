# Credit Card Fraud Detection (Machine Learning)

## Overview
This project applies supervised machine learning methods to detect fraudulent credit card transactions. The goal is to identify rare fraud cases in a highly imbalanced dataset using classification models, evaluation metrics, and techniques to handle imbalance. The implementation is written in Python and saves evaluation plots automatically.

---

## Dataset
- **Source:** Kaggle — "Credit Card Fraud Detection"
- **Type:** Supervised binary classification
- **Classes:**
  - `0` - Valid transaction
  - `1` - Fraudulent transaction
- **Features:**
  - Numerical PCA-transformed components (`V1`–`V28`)
  - `Time` and `Amount`

> Fraud cases represent a very small percentage of the dataset, so imbalance handling is essential.

---

## Features of This Project
- Exploratory Data Analysis (EDA)
- Scaling numeric features
- Train/test splitting
- Oversampling with **SMOTE**
- Class weighting
- Model training using multiple algorithms
- ROC and confusion matrix visualizations
- Automatic saving of plots

---

## Models Used
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

Each model is trained twice:
1. Using SMOTE-balanced data  
2. Using class weights on original data  

---

## Evaluation Metrics
- Accuracy
- Precision and recall
- F1-score
- ROC-AUC score
- Confusion matrix
- Feature importance

Plots are generated and saved automatically during execution.

---

## Quick usage
1. Put `creditcard.csv` in the expected path or update the path inside `Credit_Fraud_Code.py`.
2. Run the script from PowerShell:

```powershell
python "c:...\Credit Card Fraud Detection\Credit_Fraud_Code.py"
```

The script prints progress and saves PNG plots to a `plots/` directory next to the script.

---



