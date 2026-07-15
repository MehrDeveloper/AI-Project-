"""
model.py -- Core Logic Module
Keeps all AI/ML logic completely separate from the UI (app.py), as required
by the project guide's "Core Logic Module" and "Suggested Function Structure".
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score
)

FEATURES = [
    "study_hours_daily", "attendance_percent",
    "previous_score", "sleep_hours", "extra_classes"
]
TARGET = "passed"


def load_data(path):
    """Load the dataset from CSV. Raises a clear error if the file is missing."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. Run data/generate_data.py first."
        )


def preprocess_data(df):
    """Split into features/target, scale features, and do a train/test split."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_model_or_algorithm(X_train, y_train, model_type="Random Forest"):
    """Train the selected model. Two options so results can be compared."""
    if model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return the 3+ performance indicators required by the Evaluation Module."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "predictions": preds,
    }


def generate_explanation(model, model_type, feature_values, prediction):
    """
    Explainability Module: plain-language explanation of WHY the model
    produced this result, plus the key contributing factors.
    """
    result_text = "PASS" if prediction == 1 else "AT RISK OF FAILING"

    if model_type == "Random Forest":
        importances = dict(zip(FEATURES, model.feature_importances_))
    else:
        # Logistic Regression: use absolute coefficient size as importance
        importances = dict(zip(FEATURES, np.abs(model.coef_[0])))

    top_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]

    readable_names = {
        "study_hours_daily": "daily study hours",
        "attendance_percent": "class attendance",
        "previous_score": "previous exam score",
        "sleep_hours": "sleep hours",
        "extra_classes": "taking extra classes",
    }

    explanation_lines = [
        f"Predicted outcome: {result_text}.",
        f"The model weighs these factors most heavily for this prediction:"
    ]
    for feat, weight in top_factors:
        explanation_lines.append(f"  - {readable_names[feat]} (influence score: {weight:.3f})")

    return "\n".join(explanation_lines), importances


def create_visuals_data(importances):
    """Prepare feature importance data in a chart-ready format for the UI."""
    readable_names = {
        "study_hours_daily": "Study Hours",
        "attendance_percent": "Attendance %",
        "previous_score": "Previous Score",
        "sleep_hours": "Sleep Hours",
        "extra_classes": "Extra Classes",
    }
    labels = [readable_names[f] for f in importances.keys()]
    values = list(importances.values())
    return labels, values
