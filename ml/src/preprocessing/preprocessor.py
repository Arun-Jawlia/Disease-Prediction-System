from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DIABETES_TARGET = "Outcome"


HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

HEART_TARGET = "target"


LIVER_FEATURES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

LIVER_TARGET = "Dataset"


# ============================================================
# DATASET-SPECIFIC ZERO VALUE RULES
# ============================================================

DIABETES_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert clinically invalid zero representations
    into NaN for selected diabetes measurements.
    """

    df = df.copy()

    for column in DIABETES_ZERO_AS_MISSING:
        df[column] = df[column].replace(0, np.nan)

    return df


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heart dataset does not require zero-value replacement.
    """

    return df.copy()


def clean_liver_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records.

    Raw dataset remains untouched because this function
    operates on a copy.
    """

    df = df.copy()

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    return df


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def prepare_diabetes(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    df = clean_diabetes_data(df)

    X = df[DIABETES_FEATURES].copy()
    y = df[DIABETES_TARGET].copy()

    return X, y


def prepare_heart(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    df = clean_heart_data(df)

    X = df[HEART_FEATURES].copy()
    y = df[HEART_TARGET].copy()

    return X, y


def prepare_liver(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    df = clean_liver_data(df)

    X = df[LIVER_FEATURES].copy()
    y = df[LIVER_TARGET].copy()

    # Original dataset:
    # 1 = liver disease
    # 2 = no liver disease
    #
    # Normalize to:
    # 1 = positive/high risk
    # 0 = negative/low risk

    y = y.map({
        1: 1,
        2: 0,
    })

    if y.isna().any():
        raise ValueError(
            "Unexpected target value found in liver dataset."
        )

    return X, y.astype(int)


# ============================================================
# PREPROCESSOR BUILDERS
# ============================================================

def build_diabetes_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                DIABETES_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_heart_preprocessor() -> ColumnTransformer:

    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_liver_preprocessor() -> ColumnTransformer:

    numeric_features = [
        "Age",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Protiens",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]

    categorical_features = [
        "Gender",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Stratified 80/20 train-test split.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# PREPROCESSOR FIT / TRANSFORM
# ============================================================

def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit preprocessing ONLY on training data.

    Then transform both training and test data.
    """

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    return (
        X_train_processed,
        X_test_processed,
    )