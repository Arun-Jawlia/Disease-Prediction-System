from pathlib import Path

import joblib


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)


DISEASES = [
    "diabetes",
    "heart",
    "liver",
]


def main():

    print("\n")
    print("=" * 60)
    print("PRODUCTION MODEL VERIFICATION")
    print("=" * 60)

    for disease in DISEASES:

        model_path = (
            MODELS_DIR
            / disease
            / "model.joblib"
        )

        print(
            f"\nChecking {disease}..."
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found:\n"
                f"{model_path}"
            )

        model = joblib.load(
            model_path
        )

        print(
            "  ✓ Model file exists"
        )

        print(
            "  ✓ Model loaded"
        )

        print(
            f"  ✓ Model type: "
            f"{type(model).__name__}"
        )

        if hasattr(
            model,
            "predict",
        ):

            print(
                "  ✓ predict() available"
            )

        if hasattr(
            model,
            "predict_proba",
        ):

            print(
                "  ✓ predict_proba() available"
            )

    print("\n")
    print("=" * 60)
    print(
        "ALL PRODUCTION MODELS VERIFIED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()