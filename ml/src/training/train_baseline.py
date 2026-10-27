from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocessor import (
    get_preprocessor,
)

from src.training.datasets import (
    load_dataset,
)

from src.training.models import (
    get_baseline_models,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

ML_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ML_ROOT
    / "artifacts"
    / "baseline"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

MODELS_DIR = (
    RESULTS_DIR
    / "models"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    dataset_name,
    model_name,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
    )

    plt.title(
        f"{dataset_name.title()} - "
        f"{model_name.replace('_', ' ').title()}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["0", "1"],
    )

    plt.yticks(
        [0, 1],
        ["0", "1"],
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    output_dir = (
        FIGURES_DIR
        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    dataset_name,
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    preprocessor = get_preprocessor(
        dataset_name
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                clone(model),
            ),
        ]
    )

    print(
        f"\nTraining: "
        f"{dataset_name} | "
        f"{model_name}"
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        dataset_name,
        model_name,
    )

    return pipeline, metrics


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():

    all_results = []

    models = get_baseline_models()

    for dataset_name in [
        "diabetes",
        "heart",
        "liver",
    ]:

        print("\n" + "=" * 70)

        print(
            f"DATASET: "
            f"{dataset_name.upper()}"
        )

        print("=" * 70)

        X, y = load_dataset(
            dataset_name
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        )

        print(
            f"Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"Testing samples: "
            f"{len(X_test)}"
        )

        for model_name, model in models.items():

            pipeline, metrics = train_model(
                dataset_name,
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )

            result = {
                "dataset": dataset_name,
                "model": model_name,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                **metrics,
            }

            all_results.append(
                result
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        [
            "dataset",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    )

    csv_path = (
        METRICS_DIR
        / "baseline_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        METRICS_DIR
        / "baseline_results.json"
    )

    json_path.write_text(
        json.dumps(
            all_results,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:"
        f"\n{csv_path}"
    )


if __name__ == "__main__":
    main()