from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def inspect_dataset(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print("\n" + "=" * 80)
    print(f"DATASET: {path.name}")
    print("=" * 80)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nNumerical Summary:")
    print(df.describe())


if __name__ == "__main__":
    datasets = [
        "data/raw/heart/heart_disease_cleveland_new.csv",
        "data/raw/diabetes/pima-indians-diabetes.csv",
        "data/raw/liver/indian_liver_patient.csv",
    ]

    for dataset in datasets:
        inspect_dataset(dataset)