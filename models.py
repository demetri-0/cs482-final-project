# models.py
"""
Model training, hyperparameter tuning, and evaluation
for the roadway crash injury project.
"""

from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
import numpy as np
from sklearn.preprocessing import label_binarize


# =======================================================
# 1. BUILD MODEL PIPELINES
# =======================================================

def build_model_pipelines(preprocessor: Optional[ColumnTransformer] = None):
    steps_logreg = []
    steps_ridge = []
    steps_knn = []

    if preprocessor is not None:
        steps_logreg.append(("preprocessor", preprocessor))
        steps_ridge.append(("preprocessor", preprocessor))
        steps_knn.append(("preprocessor", preprocessor))

    steps_logreg.append((
        "clf",
        LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1,
        ),
    ))

    steps_ridge.append(("clf", RidgeClassifier()))
    steps_knn.append(("clf", KNeighborsClassifier()))

    return (
        Pipeline(steps_logreg),
        Pipeline(steps_ridge),
        Pipeline(steps_knn),
    )


# =======================================================
# 2. PARAMETER GRIDS
# =======================================================

def get_param_grids():
    param_grid_log_reg = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    param_grid_ridge = {
        "clf__alpha": [0.1, 1.0, 10.0],
    }

    param_grid_knn = {
        "clf__n_neighbors": [5, 11, 21],
        "clf__weights": ["uniform", "distance"],
    }

    return param_grid_log_reg, param_grid_ridge, param_grid_knn


# =======================================================
# 3. GRID SEARCH WRAPPER
# =======================================================

def run_grid_search(model_name, pipeline, param_grid, X_train, y_train, scoring="f1_weighted"):
    print(f"\n=== Grid search for {model_name} ===")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"Best params for {model_name}: {grid.best_params_}")
    print(f"Best CV {scoring} for {model_name}: {grid.best_score_:.4f}")
    return grid.best_estimator_


# =======================================================
# 4. EVALUATION HELPERS
# =======================================================

def evaluate_model(model_name, model, X, y_true, average="weighted"):
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"\n=== Evaluation: {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion_matrix(model, X, y_true, title="Confusion Matrix"):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =======================================================
# 🔥 NEW — MULTICLASS ROC SUPPORT (One-vs-Rest)
# =======================================================

def compute_and_plot_roc(model, X, y_true, title="ROC Curve (OvR)"):
    """
    Multiclass ROC AUC using One-vs-Rest.
    Plots one ROC curve for each class: 0 vs rest, 1 vs rest, 2 vs rest.
    """

    if not hasattr(model, "predict_proba"):
        print("Model does not support probability outputs. Skipping ROC.")
        return

    y_score = model.predict_proba(X)
    classes = np.unique(y_true)

    # Binarize multiclass labels for OvR ROC
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))
    aucs = {}

    for i, cls in enumerate(classes):
        try:
            auc = roc_auc_score(y_bin[:, i], y_score[:, i])
            aucs[int(cls)] = auc
            RocCurveDisplay.from_predictions(
                y_bin[:, i],
                y_score[:, i],
                name=f"Class {cls} vs Rest (AUC={auc:.3f})",
            )
        except Exception:
            print(f"Could not compute ROC for class {cls}")

    print("\nMulticlass ROC AUC (One-vs-Rest):")
    for cls, auc in aucs.items():
        print(f"  Class {cls}: AUC = {auc:.4f}")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# =======================================================
# 5. HIGH-LEVEL TRAIN/EVAL FUNCTION
# =======================================================

def train_and_evaluate_all_models(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    preprocessor=None,
    scoring="f1_weighted",
):
    log_reg_pipe, ridge_pipe, knn_pipe = build_model_pipelines(preprocessor)
    param_grid_log_reg, param_grid_ridge, param_grid_knn = get_param_grids()

    best_log_reg = run_grid_search("Logistic Regression", log_reg_pipe, param_grid_log_reg, X_train, y_train, scoring)
    best_ridge = run_grid_search("Ridge Classifier", ridge_pipe, param_grid_ridge, X_train, y_train, scoring)
    best_knn = run_grid_search("KNN", knn_pipe, param_grid_knn, X_train, y_train, scoring)

    metrics_val = [
        evaluate_model("LogReg (val)", best_log_reg, X_val, y_val),
        evaluate_model("Ridge (val)", best_ridge, X_val, y_val),
        evaluate_model("KNN (val)", best_knn, X_val, y_val),
    ]

    best_on_val = max(metrics_val, key=lambda m: m["f1"])
    best_name = best_on_val["model"]

    if "LogReg" in best_name:
        best_model = best_log_reg
    elif "Ridge" in best_name:
        best_model = best_ridge
    else:
        best_model = best_knn

    test_metrics = evaluate_model(best_name.replace("(val)", "(test)"), best_model, X_test, y_test)

    plot_confusion_matrix(best_model, X_test, y_test, title=f"Confusion Matrix – {best_name} (Test)")
    compute_and_plot_roc(best_model, X_test, y_test, title=f"ROC – {best_name} (Test)")

    print("\nFinal test metrics:", test_metrics)
    return best_model, test_metrics
