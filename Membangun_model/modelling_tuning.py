"""
modelling_tuning.py
Melatih model dengan hyperparameter tuning.
Menggunakan manual MLflow logging (bukan autolog).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub

# ── DagsHub init ─────────────────────────────────────────────────────────────
dagshub.init(repo_owner='squishydal', repo_name='Submission', mlflow=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
train_df = pd.read_csv('dreaddit_preprocessing/dreaddit-train-preprocessed.csv')
test_df  = pd.read_csv('dreaddit_preprocessing/dreaddit-test-preprocessed.csv')

X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test  = test_df.drop(columns=['label'])
y_test  = test_df['label']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Hyperparameter tuning ─────────────────────────────────────────────────────
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"Best params: {best_params}")

# ── Prediksi ──────────────────────────────────────────────────────────────────
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# ── Manual MLflow logging ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomForest_tuned"):

    # params
    mlflow.log_params(best_params)
    mlflow.log_param('cv_folds', 5)
    mlflow.log_param('scoring', 'f1')

    # metrics (sama dengan autolog + tambahan)
    mlflow.log_metric('accuracy',         accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision',        precision_score(y_test, y_pred))
    mlflow.log_metric('recall',           recall_score(y_test, y_pred))
    mlflow.log_metric('f1_score',         f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc',          roc_auc_score(y_test, y_pred_prob))
    mlflow.log_metric('log_loss',         log_loss(y_test, y_pred_prob))
    mlflow.log_metric('best_cv_f1',       grid_search.best_score_)

    # Artefak 1: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Stress', 'Stress'],
                yticklabels=['Not Stress', 'Stress'], ax=ax)
    ax.set_title('Confusion Matrix - RF Tuned')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tuned.png', dpi=100)
    mlflow.log_artifact('confusion_matrix_tuned.png')
    plt.close()

    # Artefak 2: feature importance
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top20.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Top 20 Feature Importances - RF Tuned')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_tuned.png', dpi=100)
    mlflow.log_artifact('feature_importance_tuned.png')
    plt.close()

    # Artefak 3: classification report sebagai teks
    report = classification_report(y_test, y_pred, target_names=['Not Stress', 'Stress'])
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')

    # Log model
    mlflow.sklearn.log_model(best_model, artifact_path='model')

    print("\nClassification Report:")
    print(report)
    print("\nRun selesai. Cek DagsHub Experiments tab.")
