from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH = ROOT_DIR / "models" / "student_mental_health_pipeline.joblib"


def load_processed_data():
    """Load the four processed CSV files created by Notebook 02."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Tune Logistic Regression and Random Forest using training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = {
        "Logistic Regression": GridSearchCV(
            Pipeline(
                [("model", LogisticRegression(max_iter=2000, random_state=42))]
            ),
            {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]},
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        ),
        "Random Forest": GridSearchCV(
            Pipeline([("model", RandomForestClassifier(random_state=42))]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        results[name] = search

    best_name = max(results, key=lambda name: results[name].best_score_)
    return best_name, results[best_name].best_estimator_, results


def evaluate_model(model, X_test, y_test):
    """Calculate final metrics and the confusion matrix."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro"),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1": f1_score(y_test, predictions, average="macro"),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=classes,
        ),
    }
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    return metrics, matrix


def save_model(model):
    """Save the complete trained pipeline."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def run_pipeline():
    """Run training, evaluation, and model saving."""
    X_train, X_test, y_train, y_test = load_processed_data()
    best_name, model, searches = train_model(X_train, y_train)
    metrics, matrix = evaluate_model(model, X_test, y_test)
    save_model(model)
    return {
        "model_name": best_name,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "cv_scores": {
            name: search.best_score_ for name, search in searches.items()
        },
    }


if __name__ == "__main__":
    print(run_pipeline())
