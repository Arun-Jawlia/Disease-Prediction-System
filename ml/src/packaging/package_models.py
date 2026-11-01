from pathlib import Path
import hashlib
import json
import shutil

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DISEASES = [
    "diabetes",
    "heart",
    "liver",
]


# ============================================================
# PATHS
# ============================================================

# Project root
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

ML_DIR = PROJECT_ROOT / "ml"

TUNED_MODELS_DIR = (
    ML_DIR
    / "artifacts"
    / "tuning"
    / "models"
)

FINAL_MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

FINAL_SELECTION_FILE = (
    ML_DIR
    / "artifacts"
    / "evaluation"
    / "metrics"
    / "final_models.json"
)

PACKAGING_REPORT_DIR = (
    ML_DIR
    / "artifacts"
    / "packaging"
)

PACKAGING_REPORT_FILE = (
    PACKAGING_REPORT_DIR
    / "packaging_report.json"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

FINAL_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PACKAGING_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SHA-256 CHECKSUM
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# LOAD FINAL MODEL SELECTION
# ============================================================

def load_final_selection():

    if not FINAL_SELECTION_FILE.exists():

        raise FileNotFoundError(
            "\nPhase 7 final model selection "
            "was not found.\n\n"
            f"Expected file:\n"
            f"{FINAL_SELECTION_FILE}\n\n"
            "Run Phase 7 first."
        )

    with open(
        FINAL_SELECTION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# GET MODEL NAME
# ============================================================

def get_selected_model(
    selection,
    disease,
):

    if disease not in selection:

        raise ValueError(
            f"No selected model found for "
            f"{disease}."
        )

    model_name = selection[
        disease
    ]["model"]

    return model_name


# ============================================================
# CREATE METADATA
# ============================================================

def create_metadata(
    disease,
    model_name,
    final_model_path,
    selection,
):

    metadata = {

        "project": (
            "Disease Prediction System"
        ),

        "disease": disease,

        "model": model_name,

        "artifact": "model.joblib",

        "framework": (
            "scikit-learn"
        ),

        "serialization": (
            "joblib"
        ),

        "model_type": (
            "complete preprocessing "
            "and classification pipeline"
        ),

        "selection_stage": (
            "Phase 7"
        ),

        "evaluation": {

            "accuracy": selection[
                disease
            ].get("accuracy"),

            "precision": selection[
                disease
            ].get("precision"),

            "recall": selection[
                disease
            ].get("recall"),

            "f1": selection[
                disease
            ].get("f1"),

            "roc_auc": selection[
                disease
            ].get("roc_auc"),
        },

        "confusion_matrix": {

            "true_negative": selection[
                disease
            ].get("tn"),

            "false_positive": selection[
                disease
            ].get("fp"),

            "false_negative": selection[
                disease
            ].get("fn"),

            "true_positive": selection[
                disease
            ].get("tp"),
        },

        "file": {

            "name": (
                final_model_path.name
            ),

            "sha256": (
                calculate_sha256(
                    final_model_path
                )
            ),
        },
    }

    return metadata


# ============================================================
# VERIFY MODEL
# ============================================================

def verify_model(
    model_path,
):

    print(
        f"  Loading model: "
        f"{model_path}"
    )

    model = joblib.load(
        model_path
    )

    # Check basic sklearn API
    if not hasattr(
        model,
        "predict",
    ):

        raise ValueError(
            "Model does not have "
            "predict() method."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):

        print(
            "  WARNING: Model does not "
            "provide predict_proba()."
        )

    return model


# ============================================================
# PACKAGE ONE MODEL
# ============================================================

def package_model(
    disease,
    model_name,
    selection,
):

    print("\n")
    print(
        f"Packaging: "
        f"{disease.upper()}"
    )

    # --------------------------------------------------------
    # Source model
    # --------------------------------------------------------

    source_model = (
        TUNED_MODELS_DIR
        / disease
        / f"{model_name}.joblib"
    )

    if not source_model.exists():

        raise FileNotFoundError(
            "\nSelected model does not exist:\n"
            f"{source_model}"
        )

    # --------------------------------------------------------
    # Destination directory
    # --------------------------------------------------------

    disease_dir = (
        FINAL_MODELS_DIR
        / disease
    )

    disease_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Destination model
    # --------------------------------------------------------

    final_model = (
        disease_dir
        / "model.joblib"
    )

    # --------------------------------------------------------
    # Copy model
    # --------------------------------------------------------

    shutil.copy2(
        source_model,
        final_model,
    )

    print(
        f"  Source:"
        f"\n  {source_model}"
    )

    print(
        f"  Final:"
        f"\n  {final_model}"
    )

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    verify_model(
        final_model
    )

    # --------------------------------------------------------
    # Checksum
    # --------------------------------------------------------

    checksum = calculate_sha256(
        final_model
    )

    checksum_file = (
        disease_dir
        / "checksum.sha256"
    )

    checksum_file.write_text(
        checksum,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = create_metadata(
        disease=disease,
        model_name=model_name,
        final_model_path=final_model,
        selection=selection,
    )

    metadata_file = (
        disease_dir
        / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"  SHA-256: {checksum}"
    )

    print(
        f"  Metadata:"
        f"\n  {metadata_file}"
    )

    return {
        "disease": disease,
        "model": model_name,
        "model_path": str(
            final_model
        ),
        "metadata_path": str(
            metadata_file
        ),
        "checksum_path": str(
            checksum_file
        ),
        "sha256": checksum,
        "status": "success",
    }


# ============================================================
# PACKAGE ALL MODELS
# ============================================================

def package_all():

    print("\n")
    print("=" * 70)
    print(
        "DISEASE PREDICTION SYSTEM"
    )
    print(
        "PHASE 8 - MODEL PACKAGING"
    )
    print("=" * 70)

    print(
        f"\nProject root:"
        f"\n{PROJECT_ROOT}"
    )

    print(
        f"\nFinal models directory:"
        f"\n{FINAL_MODELS_DIR}"
    )

    # --------------------------------------------------------
    # Load selection
    # --------------------------------------------------------

    selection = (
        load_final_selection()
    )

    print(
        "\nFinal model selection:"
    )

    for disease in DISEASES:

        model_name = (
            get_selected_model(
                selection,
                disease,
            )
        )

        print(
            f"  {disease}: "
            f"{model_name}"
        )

    # --------------------------------------------------------
    # Package
    # --------------------------------------------------------

    results = []

    for disease in DISEASES:

        model_name = (
            get_selected_model(
                selection,
                disease,
            )
        )

        result = package_model(
            disease=disease,
            model_name=model_name,
            selection=selection,
        )

        results.append(
            result
        )

    return results


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results):

    with open(
        PACKAGING_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        "\nPackaging report:"
        f"\n{PACKAGING_REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    results = package_all()

    save_report(
        results
    )

    print("\n")
    print("=" * 70)
    print(
        "PHASE 8 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        "\nFinal production models:"
    )

    for result in results:

        print(
            f"\n{result['disease'].upper()}"
        )

        print(
            f"Model: "
            f"{result['model']}"
        )

        print(
            f"Path: "
            f"{result['model_path']}"
        )

        print(
            f"SHA-256: "
            f"{result['sha256']}"
        )


if __name__ == "__main__":
    main()