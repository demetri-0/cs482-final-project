# models.py
"""
This file handles training a few different ML models, tuning them with grid search,
and evaluating them for the roadway crash injury project.
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


# Build model pipelines

def build_model_pipelines(preprocessor: Optional[ColumnTransformer] = None):
    # We build 3 separate pipelines: logistic regression, ridge classifier, and KNN
    steps_logreg = []
    steps_ridge = []
    steps_knn = []

    if preprocessor is not None:
        steps_logreg.append(("preprocessor", preprocessor))
        steps_ridge.append(("preprocessor", preprocessor))
        steps_knn.append(("preprocessor", preprocessor))

    # Add the actual classifier for logistic regression
    steps_logreg.append((
        "clf",
        LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1,
        ),
    ))

    # RidgeClassifier is another linear model 
    steps_ridge.append(("clf", RidgeClassifier()))

    # KNN is a distance-based model that predicts based on nearest neighbors
    steps_knn.append(("clf", KNeighborsClassifier()))

    # Return the three pipelines so the rest of the code can tune/evaluate them
    return (
        Pipeline(steps_logreg),
        Pipeline(steps_ridge),
        Pipeline(steps_knn),
    )


# Parameter grids
def get_param_grids():
    # These dictionaries define what hyperparameters GridSearchCV will try for each model

    # Logistic regression grid: different regularization strengths (C)
    # penalty is fixed to L2 since that's the common/default here
    param_grid_log_reg = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    # RidgeClassifier grid: alpha controls strength of regularization
    param_grid_ridge = {
        "clf__alpha": [0.1, 1.0, 10.0],
    }

    # KNN grid: try a few neighbor counts and whether votes are weighted by distance
    param_grid_knn = {
        "clf__n_neighbors": [5, 11, 21],
        "clf__weights": ["uniform", "distance"],
    }

    # Return all three so train and evaluate all models can plug them in
    return param_grid_log_reg, param_grid_ridge, param_grid_knn


# Grid search wrapper
def run_grid_search(model_name, pipeline, param_grid, X_train, y_train, scoring="f1_weighted"):
    # Wrapper around GridSearchCV so every model gets tuned the same way
    print(f"\n=== Grid search for {model_name} ===")

    # cv=5 means 5-fold cross validation on the training set
    # scoring defaults to weighted F1 to account for class imbalance
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )

    # Fit grid search: tries all combinations of params across the CV folds
    grid.fit(X_train, y_train)

    # Print best settings found and the best cross-validated score
    print(f"Best params for {model_name}: {grid.best_params_}")
    print(f"Best CV {scoring} for {model_name}: {grid.best_score_:.4f}")

    # Return the best pipeline (preprocessor + tuned model) ready to use
    return grid.best_estimator_


# Evaluation helpers

def evaluate_model(model_name, model, X, y_true, average="weighted"):
    # Predict labels for the given dataset split (val or test)
    y_pred = model.predict(X)

    # Basic classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Print metrics so it's easy to compare models in the console output
    print(f"\n=== Evaluation: {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Full per-class breakdown
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Return a dict so we can rank models and save metrics
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion_matrix(model, X, y_true, title="Confusion Matrix"):
    # Confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# One vs rest ROC 

def compute_and_plot_roc(model, X, y_true, title="ROC Curve (OvR)"):
    """
    Multiclass ROC AUC using One-vs-Rest.
    Plots one ROC curve for each class: 0 vs rest, 1 vs rest, 2 vs rest.
    """

    if not hasattr(model, "predict_proba"):
        print("Model does not support probability outputs. Skipping ROC.")
        return

    # Probability scores for each class 
    y_score = model.predict_proba(X)

    # Get the unique class labels
    classes = np.unique(y_true)

    # Convert multiclass labels into a binary matrix for One-vs-Rest ROC calculations
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))
    aucs = {}

    # Loop through each class and compute "this class vs all other classes" ROC AUC
    for i, cls in enumerate(classes):
        try:
            # Compute AUC for the current class using its binary labels and probability scores
            auc = roc_auc_score(y_bin[:, i], y_score[:, i])
            aucs[int(cls)] = auc

            # Plot ROC curve for that class
            RocCurveDisplay.from_predictions(
                y_bin[:, i],
                y_score[:, i],
                name=f"Class {cls} vs Rest (AUC={auc:.3f})",
            )
        except Exception:
            # If something goes wrong (like missing class column), don't crash the whole run
            print(f"Could not compute ROC for class {cls}")

    # Print the AUC values so you can report them or compare across models
    print("\nMulticlass ROC AUC (One-vs-Rest):")
    for cls, auc in aucs.items():
        print(f"  Class {cls}: AUC = {auc:.4f}")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# High level train/eval

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
    # Build the three pipelines 
    log_reg_pipe, ridge_pipe, knn_pipe = build_model_pipelines(preprocessor)

    # Get the hyperparameter grids for each model
    param_grid_log_reg, param_grid_ridge, param_grid_knn = get_param_grids()

    # Tune each model using grid search on the training set
    best_log_reg = run_grid_search("Logistic Regression", log_reg_pipe, param_grid_log_reg, X_train, y_train, scoring)
    best_ridge = run_grid_search("Ridge Classifier", ridge_pipe, param_grid_ridge, X_train, y_train, scoring)
    best_knn = run_grid_search("KNN", knn_pipe, param_grid_knn, X_train, y_train, scoring)

    # Evaluate all tuned models on the validation set so we can pick the best one
    metrics_val = [
        evaluate_model("LogReg (val)", best_log_reg, X_val, y_val),
        evaluate_model("Ridge (val)", best_ridge, X_val, y_val),
        evaluate_model("KNN (val)", best_knn, X_val, y_val),
    ]

    # Pick the best model based on validation weighted F1
    best_on_val = max(metrics_val, key=lambda m: m["f1"])
    best_name = best_on_val["model"]

    # Figure out which actual fitted pipeline corresponds to that best validation name
    if "LogReg" in best_name:
        best_model = best_log_reg
    elif "Ridge" in best_name:
        best_model = best_ridge
    else:
        best_model = best_knn

    # Final evaluation on the test set for an unbiased estimate of performance
    test_metrics = evaluate_model(best_name.replace("(val)", "(test)"), best_model, X_test, y_test)

    # Plot helpful visuals for the final selected model
    plot_confusion_matrix(best_model, X_test, y_test, title=f"Confusion Matrix – {best_name} (Test)")
    compute_and_plot_roc(best_model, X_test, y_test, title=f"ROC – {best_name} (Test)")

    # Print + return the final model and its test results
    print("\nFinal test metrics:", test_metrics)
    return best_model, test_metrics
