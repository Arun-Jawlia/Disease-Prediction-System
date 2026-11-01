from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DISEASES = [
    "diabetes",
    "heart",
    "liver",
]

MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
]


# ============================================================
# PATHS
# ============================================================

# ml/
ML_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ML_DIR / "data" / "raw"

TUNED_MODELS_DIR = (
    ML_DIR
    / "artifacts"
    / "tuning"
    / "models"
)

OUTPUT_DIR = (
    ML_DIR
    / "artifacts"
    / "evaluation"
)

METRICS_DIR = OUTPUT_DIR / "metrics"

FIGURES_DIR = OUTPUT_DIR / "figures"

CONFUSION_DIR = (
    FIGURES_DIR
    / "confusion_matrices"
)


# Create output directories
METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONFUSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_FILES = {
    "diabetes": DATA_DIR / "diabetes" / "pima-indians-diabetes.csv",
    "heart": DATA_DIR / "heart" / "heart_disease_cleveland_new.csv",
    "liver": DATA_DIR / "liver" / "indian_liver_patient.csv",
}


# ============================================================
# TARGET COLUMN CONFIGURATION
# ============================================================

TARGET_COLUMNS = {
    "diabetes": "Outcome",
    "heart": "target",
    "liver": "Dataset",
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(disease):

    path = DATASET_FILES[disease]

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found for {disease}:\n"
            f"{path}\n"
        )

    print(
        f"\nLoading {disease} dataset..."
    )

    df = pd.read_csv(path)

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(df, disease):

    target_column = TARGET_COLUMNS[disease]

    if target_column not in df.columns:

        raise ValueError(
            f"\nTarget column '{target_column}' "
            f"was not found in {disease} dataset.\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if disease == "diabetes":

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        return X, y

    # --------------------------------------------------------
    # Heart
    # --------------------------------------------------------

    elif disease == "heart":

        X = df.drop(
            columns=[target_column]
        )

        y = df[target_column]

        return X, y

    # --------------------------------------------------------
    # Liver
    # --------------------------------------------------------

    elif disease == "liver":

        # ILPD commonly contains:
        #
        # Dataset
        # 1 = liver disease
        # 2 = no liver disease
        #
        # Convert to:
        # 1 = high risk
        # 0 = low risk

        y = df[target_column].copy()

        y = y.map({
            1: 1,
            2: 0,
        })

        if y.isna().any():

            raise ValueError(
                "Unexpected values found in "
                "liver target column."
            )

        X = df.drop(
            columns=[target_column]
        )

        return X, y

    raise ValueError(
        f"Unsupported disease: {disease}"
    )


# ============================================================
# CREATE TRAIN / TEST SPLIT
# ============================================================

def create_test_split(X, y):

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# FIND MODEL FILE
# ============================================================

def get_model_path(
    disease,
    model_name,
):

    path = (
        TUNED_MODELS_DIR
        / disease
        / f"{model_name}.joblib"
    )

    return path


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        y_probability = (
            model
            .predict_proba(X_test)[:, 1]
        )

    elif hasattr(
        model,
        "decision_function",
    ):

        y_probability = (
            model
            .decision_function(X_test)
        )

    else:

        y_probability = y_pred

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
    }


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    disease,
    model_name,
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Low Risk",
            "High Risk",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"{disease.title()} - "
        f"{model_name.replace('_', ' ').title()}"
    )

    output_path = (
        CONFUSION_DIR
        / f"{disease}_{model_name}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all():

    all_results = []

    print("\n")
    print("=" * 70)
    print("PHASE 7 - MODEL EVALUATION")
    print("=" * 70)

    for disease in DISEASES:

        print("\n")
        print("-" * 70)
        print(
            f"EVALUATING: {disease.upper()}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_dataset(
            disease
        )

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X, y = prepare_dataset(
            df,
            disease,
        )

        # ----------------------------------------------------
        # Same test split as Phase 6
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = create_test_split(
            X,
            y,
        )

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Test samples: {len(X_test)}"
        )

        # ----------------------------------------------------
        # Evaluate five models
        # ----------------------------------------------------

        for model_name in MODELS:

            print("\n")
            print(
                f"Model: {model_name}"
            )

            model_path = get_model_path(
                disease,
                model_name,
            )

            if not model_path.exists():

                print(
                    "WARNING: Model file not found:"
                )

                print(
                    model_path
                )

                continue

            # Load tuned model
            model = joblib.load(
                model_path
            )

            # Evaluate
            metrics = evaluate_model(
                model,
                X_test,
                y_test,
            )

            # Save confusion matrix
            cm_path = (
                save_confusion_matrix(
                    metrics[
                        "confusion_matrix"
                    ],
                    disease,
                    model_name,
                )
            )

            cm = metrics[
                "confusion_matrix"
            ]

            # Store result
            result = {

                "disease": disease,

                "model": model_name,

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

                "tn": int(cm[0, 0]),

                "fp": int(cm[0, 1]),

                "fn": int(cm[1, 0]),

                "tp": int(cm[1, 1]),

                "confusion_matrix": str(
                    cm.tolist()
                ),

                "confusion_matrix_file": str(
                    cm_path
                ),
            }

            all_results.append(
                result
            )

            print(
                f"Accuracy : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"Recall   : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"F1       : "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"ROC-AUC  : "
                f"{metrics['roc_auc']:.4f}"
            )

    return pd.DataFrame(
        all_results
    )


# ============================================================
# SELECT FINAL MODELS
# ============================================================

def select_final_models(
    results_df,
):

    final_models = {}

    for disease in DISEASES:

        disease_results = (
            results_df[
                results_df["disease"]
                == disease
            ]
            .copy()
        )

        if disease_results.empty:

            print(
                f"\nNo results found for "
                f"{disease}."
            )

            continue

        # ----------------------------------------------------
        # Ranking strategy
        #
        # Screening system:
        # Recall → F1 → ROC-AUC → Precision → Accuracy
        # ----------------------------------------------------

        disease_results = (
            disease_results
            .sort_values(
                by=[
                    "recall",
                    "f1",
                    "roc_auc",
                    "precision",
                    "accuracy",
                ],
                ascending=False,
            )
        )

        winner = (
            disease_results
            .iloc[0]
        )

        final_models[
            disease
        ] = {

            "model": winner[
                "model"
            ],

            "accuracy": float(
                winner["accuracy"]
            ),

            "precision": float(
                winner["precision"]
            ),

            "recall": float(
                winner["recall"]
            ),

            "f1": float(
                winner["f1"]
            ),

            "roc_auc": float(
                winner["roc_auc"]
            ),

            "tn": int(
                winner["tn"]
            ),

            "fp": int(
                winner["fp"]
            ),

            "fn": int(
                winner["fn"]
            ),

            "tp": int(
                winner["tp"]
            ),
        }

    return final_models


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    final_models,
):

    # --------------------------------------------------------
    # Remove complex column for CSV
    # --------------------------------------------------------

    csv_df = results_df.copy()

    csv_path = (
        METRICS_DIR
        / "test_set_results.csv"
    )

    csv_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranking_df = (
        results_df
        .copy()
    )

    ranking_df["rank"] = (
        ranking_df
        .groupby("disease")["f1"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking_df = (
        ranking_df
        .sort_values(
            [
                "disease",
                "rank",
            ]
        )
    )

    ranking_path = (
        METRICS_DIR
        / "model_ranking.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final models JSON
    # --------------------------------------------------------

    final_path = (
        METRICS_DIR
        / "final_models.json"
    )

    with open(
        final_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_models,
            file,
            indent=4,
        )

    print("\n")
    print("=" * 70)
    print("RESULT FILES CREATED")
    print("=" * 70)

    print(
        f"\nTest results:\n{csv_path}"
    )

    print(
        f"\nModel ranking:\n{ranking_path}"
    )

    print(
        f"\nFinal models:\n{final_path}"
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    final_models,
):

    print("\n")
    print("=" * 70)
    print("FINAL MODEL SELECTION")
    print("=" * 70)

    for disease, result in (
        final_models.items()
    ):

        print("\n")

        print(
            f"{disease.upper()}"
        )

        print(
            f"Selected Model: "
            f"{result['model']}"
        )

        print(
            f"Accuracy : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{result['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{result['roc_auc']:.4f}"
        )

        print(
            f"TN={result['tn']} "
            f"FP={result['fp']} "
            f"FN={result['fn']} "
            f"TP={result['tp']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "DISEASE PREDICTION SYSTEM"
    )
    print(
        "PHASE 7 - MODEL EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nML directory:"
        f"\n{ML_DIR}"
    )

    print(
        f"\nModels directory:"
        f"\n{TUNED_MODELS_DIR}"
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results_df = evaluate_all()

    if results_df.empty:

        raise RuntimeError(
            "\nNo models were evaluated.\n"
            "Check whether Phase 6 generated "
            "the .joblib files."
        )

    # --------------------------------------------------------
    # Save test results
    # --------------------------------------------------------

    final_models = (
        select_final_models(
            results_df
        )
    )

    save_results(
        results_df,
        final_models,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_final_summary(
        final_models
    )

    print("\n")
    print("=" * 70)
    print("PHASE 7 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()