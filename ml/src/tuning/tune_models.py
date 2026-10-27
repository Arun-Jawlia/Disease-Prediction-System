from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import (
    calculate_classification_metrics,
)

from src.preprocessing.preprocessor import (
    get_preprocessor,
)

from src.training.datasets import (
    load_dataset,
)

from src.tuning.param_grids import (
    MODEL_CONFIGS,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ML_ROOT = Path(__file__).resolve().parents[2]

TUNING_DIR = (
    ML_ROOT
    / "artifacts"
    / "tuning"
)

METRICS_DIR = (
    TUNING_DIR
    / "metrics"
)

MODELS_DIR = (
    TUNING_DIR
    / "models"
)

CV_RESULTS_DIR = (
    TUNING_DIR
    / "cv_results"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CV_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CROSS VALIDATION
# ============================================================

CV = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(
    dataset_name: str,
    model,
) -> Pipeline:

    preprocessor = get_preprocessor(
        dataset_name
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


# ============================================================
# TUNE ONE MODEL
# ============================================================

def tune_model(
    dataset_name: str,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
):

    config = MODEL_CONFIGS[
        model_name
    ]

    pipeline = build_pipeline(
        dataset_name,
        config["model"],
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Tuning: "
        f"{dataset_name.upper()} | "
        f"{model_name}"
    )

    print(
        f"{'=' * 70}"
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="f1",
        cv=CV,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV F1:"
    )

    print(
        grid_search.best_score_
    )

    # --------------------------------------------------------
    # FINAL HELD-OUT TEST SET
    # --------------------------------------------------------

    best_pipeline = (
        grid_search.best_estimator_
    )

    y_pred = best_pipeline.predict(
        X_test
    )

    y_probability = (
        best_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = (
        calculate_classification_metrics(
            y_test,
            y_pred,
            y_probability,
        )
    )

    # --------------------------------------------------------
    # SAVE CV RESULTS
    # --------------------------------------------------------

    cv_results = pd.DataFrame(
        grid_search.cv_results_
    )

    cv_path = (
        CV_RESULTS_DIR
        / f"{dataset_name}_{model_name}.csv"
    )

    cv_results.to_csv(
        cv_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_dir = (
        MODELS_DIR
        / dataset_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / f"{model_name}.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path,
    )

    # --------------------------------------------------------
    # RESULT OBJECT
    # --------------------------------------------------------

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "best_cv_f1": float(
            grid_search.best_score_
        ),
        "best_params": (
            grid_search.best_params_
        ),
        "test_metrics": test_metrics,
        "model_path": str(
            model_path
        ),
        "cv_results_path": str(
            cv_path
        ),
    }

    print(
        "\nHeld-out test metrics:"
    )

    for metric, value in test_metrics.items():

        print(
            f"{metric:12}: "
            f"{value:.4f}"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    for dataset_name in [
        "diabetes",
        "heart",
        "liver",
    ]:

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
            f"\n{dataset_name.upper()}"
        )

        print(
            f"Train: {X_train.shape}"
        )

        print(
            f"Test: {X_test.shape}"
        )

        for model_name in MODEL_CONFIGS:

            result = tune_model(
                dataset_name=dataset_name,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            all_results.append(
                result
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_records = []

    for result in all_results:

        metrics = result[
            "test_metrics"
        ]

        summary_records.append(
            {
                "dataset": result[
                    "dataset"
                ],

                "model": result[
                    "model"
                ],

                "best_cv_f1": result[
                    "best_cv_f1"
                ],

                "accuracy": metrics[
                    "accuracy"
                ],

                "precision": metrics[
                    "precision"
                ],

                "recall": metrics[
                    "recall"
                ],

                "f1": metrics[
                    "f1"
                ],

                "roc_auc": metrics[
                    "roc_auc"
                ],

                "best_params": json.dumps(
                    result[
                        "best_params"
                    ]
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df = summary_df.sort_values(
        [
            "dataset",
            "f1",
        ],
        ascending=[
            True,
            False,
        ],
    )

    summary_path = (
        METRICS_DIR
        / "tuned_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    json_path = (
        METRICS_DIR
        / "tuned_results.json"
    )

    json_path.write_text(
        json.dumps(
            all_results,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HYPERPARAMETER TUNING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSummary:"
        f"\n{summary_path}"
    )


if __name__ == "__main__":
    main()