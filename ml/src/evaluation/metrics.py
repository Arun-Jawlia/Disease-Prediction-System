import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_probability,
):
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
    }