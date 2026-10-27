from pathlib import Path

import pandas as pd

from src.preprocessing.preprocessor import (
    prepare_diabetes,
    prepare_heart,
    prepare_liver,
)


ML_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "diabetes": {
        "path": (
            ML_ROOT
            / "data"
            / "raw"
            / "diabetes"
            / "pima-indians-diabetes.csv"
        ),
        "prepare": prepare_diabetes,
    },

    "heart": {
        "path": (
            ML_ROOT
            / "data"
            / "raw"
            / "heart"
            / "heart_disease_cleveland_new.csv"
        ),
        "prepare": prepare_heart,
    },

    "liver": {
        "path": (
            ML_ROOT
            / "data"
            / "raw"
            / "liver"
            / "indian_liver_patient.csv"
        ),
        "prepare": prepare_liver,
    },
}


def load_dataset(dataset_name: str):

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    config = DATASETS[dataset_name]

    df = pd.read_csv(config["path"])

    X, y = config["prepare"](df)

    return X, y