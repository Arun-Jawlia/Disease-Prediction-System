from pathlib import Path

import pandas as pd

from preprocessor import (
    prepare_diabetes,
    prepare_heart,
    prepare_liver,
    build_diabetes_preprocessor,
    build_heart_preprocessor,
    build_liver_preprocessor,
    split_dataset,
)


ML_ROOT = Path(__file__).resolve().parents[2]

DIABETES_PATH = (
    ML_ROOT
    / "data"
    / "raw"
    / "diabetes"
    / "pima-indians-diabetes.csv"
)

HEART_PATH = (
    ML_ROOT
    / "data"
    / "raw"
    / "heart"
    / "heart_disease_cleveland_new.csv"
)

LIVER_PATH = (
    ML_ROOT
    / "data"
    / "raw"
    / "liver"
    / "indian_liver_patient.csv"
)


def test_diabetes():

    df = pd.read_csv(DIABETES_PATH)

    X, y = prepare_diabetes(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    preprocessor = build_diabetes_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nDIABETES")
    print("-" * 50)
    print("Original:", X.shape)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print(
        "Processed train:",
        X_train_processed.shape,
    )
    print(
        "Processed test:",
        X_test_processed.shape,
    )


def test_heart():

    df = pd.read_csv(HEART_PATH)

    X, y = prepare_heart(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    preprocessor = build_heart_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nHEART")
    print("-" * 50)
    print("Original:", X.shape)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print(
        "Processed train:",
        X_train_processed.shape,
    )
    print(
        "Processed test:",
        X_test_processed.shape,
    )


def test_liver():

    df = pd.read_csv(LIVER_PATH)

    X, y = prepare_liver(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    preprocessor = build_liver_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nLIVER")
    print("-" * 50)
    print("Original after duplicate removal:", X.shape)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print(
        "Processed train:",
        X_train_processed.shape,
    )
    print(
        "Processed test:",
        X_test_processed.shape,
    )


if __name__ == "__main__":

    test_diabetes()
    test_heart()
    test_liver()

    print("\nPreprocessing validation completed.")