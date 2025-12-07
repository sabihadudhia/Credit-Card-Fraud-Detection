import pandas as pd
import numpy as np
import os
import time
# Limit parallel threads for BLAS/OMP libraries to avoid heavy nested parallelism on Windows
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
# Ensure matplotlib uses a local, writable config directory (avoids hangs on network/OneDrive paths)
os.environ.setdefault('MPLCONFIGDIR', r"C:\Users\sabih\.matplotlib")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib
# Use a non-interactive backend when running from the terminal to avoid GUI/backend issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.inspection import permutation_importance

# Plot helpers: save current figure using its title (sanitized). Avoid calling plt.show() since the script
# uses the non-interactive 'Agg' backend. Plots are written to a local `plots/` folder.
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def _safe_name(s: str) -> str:
    if s is None:
        return 'figure'
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s).strip().replace(' ', '_')

def save_current_plot(suffix: str = None):
    title = None
    try:
        title = plt.gca().get_title()
    except Exception:
        title = None
    name = _safe_name(title)
    if suffix:
        name = f"{name}_{suffix}"
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")

# 1. Load and inspect the dataset
data = pd.read_csv("C:\\Users\\sabih\\OneDrive\\Desktop\\Projects\\Credit Card Fraud Detection\\creditcard.csv")
print("Dataset shape:", data.shape)
print("\nClass distribution:")
print(data['Class'].value_counts())
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nMissing values check:")
print("Any missing values:", data.isnull().sum().any())


#2. Exploratory Data Analysis (EDA)
# EDA Summary: Quick visual checks to understand class imbalance and amount distribution
# Class distribution plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
save_current_plot('class_distribution')

# Transaction amount by class
plt.figure(figsize=(10, 5))
sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Transaction Amount by Class')
save_current_plot('amount_by_class')

# Correlation heatmap (EDT visual upgrade)
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm", linewidths=0.2)
plt.title("Correlation Matrix")
save_current_plot('correlation_matrix')

# 3. Data Preprocessing
# Preprocessing Summary: scaling 'Amount' and 'Time', dropping originals, preparing X/y and train-test split
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

# Prepare features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nOriginal training set class distribution:")
print(f"Class 0 (Normal): {(y_train == 0).sum()}")
print(f"Class 1 (Fraud): {(y_train == 1).sum()}")

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE resampling:")
print(f"Class 0 (Normal): {(y_train_res == 0).sum()}")
print(f"Class 1 (Fraud): {(y_train_res == 1).sum()}")


# 4. Model Training
# Prepare both SMOTE-trained models and no-SMOTE (class_weight) models for comparison
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / pos) if pos > 0 else 1

models_smote = {
    "Logistic Regression (SMOTE)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    # Constrain Random Forest to be faster: fewer trees, restrict tree depth, single-threaded
    "Random Forest (SMOTE)": RandomForestClassifier(n_estimators=50, n_jobs=1, max_depth=12, class_weight='balanced', random_state=42),
    # Make XGBoost single-threaded for predictable resource use
    "XGBoost (SMOTE)": xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42, n_jobs=1)
}

models_no_smote = {
    "Logistic Regression (class-weight)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    # Constrain Random Forest to be faster: fewer trees, single-threaded
    "Random Forest (class-weight)": RandomForestClassifier(n_estimators=50, n_jobs=1, max_depth=12, class_weight='balanced', random_state=42),
    "XGBoost (class-weight)": xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1)
}

# Train models and get initial scores
results = {}
combined_models = {}
print(f"\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Train and cross-validate SMOTE models (trained on resampled data)
for name, model in models_smote.items():
    print(f"\n--- START: Training {name} (trained on SMOTE-resampled data) ---")
    try:
        # If resampled training set is very large, sample for CV to save time/resources
        X_cv, y_cv = X_train_res, y_train_res
        if len(X_train_res) > 50000:
            print(f"Large resampled training set ({len(X_train_res)}) detected. Sampling 50k rows for CV to save time.")
            X_cv = X_train_res.sample(n=50000, random_state=42)
            y_cv = y_train_res.loc[X_cv.index]
        print(f"Starting CV (3-fold) for {name} on {len(X_cv)} samples...")
        cv_t0 = time.time()
        cv_scores = cross_val_score(model, X_cv, y_cv, cv=3, scoring='roc_auc', n_jobs=1)
        cv_t1 = time.time()
        print(f"Completed CV for {name} in {cv_t1 - cv_t0:.1f}s. CV ROC-AUC (mean +/- std): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print(f"CV failed for {name} on SMOTE data: {e}")
    t0 = time.time()
    print(f"Fitting model {name} on resampled training data...")
    model.fit(X_train_res, y_train_res)
    t1 = time.time()
    print(f"Finished training {name} - Training time: {t1 - t0:.1f}s")
    train_score = model.score(X_train_res, y_train_res)
    test_score = model.score(X_test, y_test)
    results[name] = test_score
    combined_models[name] = model
    print(f"{name} - Training Score: {train_score:.4f}")
    print(f"{name} - Test Score: {test_score:.4f}")

# Train and cross-validate class-weight models (trained on original imbalanced data)
for name, model in models_no_smote.items():
    print(f"\n--- START: Training {name} (trained on original data with class-weight) ---")
    try:
        X_cv, y_cv = X_train, y_train
        if len(X_train) > 50000:
            print(f"Large training set ({len(X_train)}) detected. Sampling 50k rows for CV to save time.")
            X_cv = X_train.sample(n=50000, random_state=42)
            y_cv = y_train.loc[X_cv.index]
        print(f"Starting CV (3-fold) for {name} on {len(X_cv)} samples...")
        cv_t0 = time.time()
        cv_scores = cross_val_score(model, X_cv, y_cv, cv=3, scoring='roc_auc', n_jobs=1)
        cv_t1 = time.time()
        print(f"Completed CV for {name} in {cv_t1 - cv_t0:.1f}s. CV ROC-AUC (mean +/- std): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print(f"CV failed for {name} on original data: {e}")
    t0 = time.time()
    print(f"Fitting model {name} on original training data...")
    model.fit(X_train, y_train)
    t1 = time.time()
    print(f"Finished training {name} - Training time: {t1 - t0:.1f}s")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    results[name] = test_score
    combined_models[name] = model
    print(f"{name} - Training Score: {train_score:.4f}")
    print(f"{name} - Test Score: {test_score:.4f}")

# 5. Model Evaluation
print(f"\n" + "="*50)
print("DETAILED MODEL EVALUATION")
print("="*50)
#
# Evaluation Summary: For each trained model we print classification metrics, show confusion matrix
# plot the ROC; we also collect ROC data for a combined comparison plot below.
# Collect ROC data for later combined plotting
roc_results = {}

for name, model in combined_models.items():
    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Evaluating model: {name} on test set...")
    eval_t0 = time.time()
    y_pred = model.predict(X_test)
    # Some models (possibly) may not have predict_proba; handle gracefully
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision_function when available
        if hasattr(model, 'decision_function'):
            try:
                scores = model.decision_function(X_test)
                # convert to a 0-1 range using min-max for ROC ranking only
                y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min())
            except Exception:
                y_pred_proba = y_pred
        else:
            y_pred_proba = y_pred
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    eval_t1 = time.time()
    print(f"Completed evaluation for {name} in {eval_t1 - eval_t0:.1f}s")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_current_plot(f"{_safe_name(name)}_confusion")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    # store ROC data for combined plotting
    roc_results[name] = (fpr, tpr, auc_score)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.grid(True)
    save_current_plot(f"{_safe_name(name)}_roc")
    print(f"Saved ROC plot for {name}")

# 6. Model Comparison and Summary
print(f"\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

# Combined ROC Curve Comparison
if roc_results:
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc_score) in roc_results.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    save_current_plot('roc_comparison')

# Feature Importance and Explainability
print("\nFeature importance / explainability:\n")
for name, model in combined_models.items():
    print(f"\n{name} feature importance:")
    print(f"Computing feature importance for {name}...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        display_df = feat_imp.head(15)
        print(display_df)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=display_df.values, y=display_df.index, palette='viridis')
        plt.title(f"{name} - Top 15 Feature Importances")
        plt.xlabel('Importance')
        plt.tight_layout()
        save_current_plot(f"{_safe_name(name)}_feat_imp")
        print(f"Saved feature importance plot for {name}")
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_).ravel()
        coef_ser = pd.Series(coefs, index=X.columns).sort_values(ascending=False)
        display_df = coef_ser.head(15)
        print(display_df)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=display_df.values, y=display_df.index, palette='magma')
        plt.title(f"{name} - Top 15 absolute coefficients")
        plt.xlabel('Absolute Coefficient')
        plt.tight_layout()
        save_current_plot(f"{_safe_name(name)}_coef")
        print(f"Saved coefficient plot for {name}")
    else:
        # Fallback: permutation importance (model-agnostic)
        try:
            from sklearn.inspection import permutation_importance
            # fewer repeats to save time; n_jobs=1 to avoid parallelism clashes
            r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
            perm_imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
            display_df = perm_imp.head(15)
            print(display_df)
            plt.figure(figsize=(8, 6))
            sns.barplot(x=display_df.values, y=display_df.index, palette='coolwarm')
            plt.title(f"{name} - Top 15 Permutation Importances")
            plt.xlabel('Importance')
            plt.tight_layout()
            save_current_plot(f"{_safe_name(name)}_perm_imp")
            print(f"Saved permutation importance plot for {name}")
        except Exception as e:
            print(f"Could not compute importances for {name}: {e}")

print(f"\nModel Performance Ranking:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    print(f"#{i}: {model_name} - Accuracy: {accuracy:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest performing model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]:.4f}")

print(f"\nDataset Information:")
print(f"• Total transactions: {len(data)}")
print(f"• Fraud transactions: {(data['Class'] == 1).sum()}")
print(f"• Normal transactions: {(data['Class'] == 0).sum()}")
print(f"• Fraud percentage: {((data['Class'] == 1).sum() / len(data)) * 100:.2f}%")

print(f"\nKey Insights:")
print(f"• This is a highly imbalanced dataset with only {((data['Class'] == 1).sum() / len(data)) * 100:.2f}% fraud cases")
print(f"• SMOTE was used to balance the training data")
print(f"• Both models achieved high accuracy on the test set")
print(f"• Focus on precision and recall for fraud detection is crucial")

print(f"\nRECOMMENDATIONS:")
print(f"• For production: Use {best_model_name} (highest accuracy)")
print(f"• Monitor false positive rates to avoid blocking legitimate transactions")
print(f"• Consider ensemble methods for even better performance")
print(f"• Regular model retraining is essential as fraud patterns evolve")

print(f"\nAnalysis Complete!")
print("="*50)
