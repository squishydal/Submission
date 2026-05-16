"""
modelling.py
Melatih model Random Forest untuk Dreaddit stress detection.
Menggunakan MLflow autolog + DagsHub tracking.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

# ── MLflow autolog ────────────────────────────────────────────────────────────
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_baseline"):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Artefak tambahan 1: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Stress', 'Stress'],
                yticklabels=['Not Stress', 'Stress'], ax=ax)
    ax.set_title('Confusion Matrix - Random Forest Baseline')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

    # Artefak tambahan 2: feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top20.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Top 20 Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    mlflow.log_artifact('feature_importance.png')
    plt.close()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Stress', 'Stress']))
    print("Run selesai. Cek DagsHub Experiments tab.")
