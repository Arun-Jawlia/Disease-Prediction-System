from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# PATH CONFIGURATION
# ============================================================

ML_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = ML_ROOT / "data" / "raw"
FIGURES_DIR = ML_ROOT / "artifacts" / "figures"
REPORTS_DIR = ML_ROOT / "artifacts" / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "diabetes": {
        "path": RAW_DATA_DIR / "diabetes" / "pima-indians-diabetes.csv",
        "target": "Outcome",
    },
    "heart": {
        "path": RAW_DATA_DIR / "heart" / "heart_disease_cleveland_new.csv",
        "target": "target",
    },
    "liver": {
        "path": RAW_DATA_DIR / "liver" / "indian_liver_patient.csv",
        "target": "Dataset",
    },
}


# ============================================================
# GENERAL SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_target_distribution(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
) -> None:

    output_dir = FIGURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=target,
    )

    plt.title(f"{dataset_name.title()} - Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        output_dir / "target_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
) -> None:

    output_dir = FIGURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_features.remove(target)

    for feature in numeric_features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
        )

        plt.title(
            f"{dataset_name.title()} - {feature} Distribution"
        )

        plt.tight_layout()

        safe_name = feature.lower().replace(" ", "_")

        plt.savefig(
            output_dir / f"{safe_name}_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# BOX PLOTS
# ============================================================

def plot_boxplots(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
) -> None:

    output_dir = FIGURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_features.remove(target)

    for feature in numeric_features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=feature,
        )

        plt.title(
            f"{dataset_name.title()} - {feature} Box Plot"
        )

        plt.tight_layout()

        safe_name = feature.lower().replace(" ", "_")

        plt.savefig(
            output_dir / f"{safe_name}_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# CORRELATION MATRIX
# ============================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:

    output_dir = FIGURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    plt.figure(
        figsize=(
            max(10, len(correlation.columns) * 0.8),
            max(8, len(correlation.columns) * 0.7),
        )
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title(
        f"{dataset_name.title()} - Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FEATURE VS TARGET
# ============================================================

def plot_feature_target_relationships(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
) -> None:

    output_dir = FIGURES_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_features.remove(target)

    for feature in numeric_features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=target,
            y=feature,
        )

        plt.title(
            f"{dataset_name.title()} - "
            f"{feature} vs {target}"
        )

        plt.tight_layout()

        safe_name = feature.lower().replace(" ", "_")

        plt.savefig(
            output_dir / f"{safe_name}_vs_target.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# ZERO VALUE ANALYSIS
# ============================================================

def get_zero_value_summary(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:

    numeric_features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_features.remove(target)

    records = []

    for feature in numeric_features:

        zero_count = (df[feature] == 0).sum()

        records.append(
            {
                "feature": feature,
                "zero_count": zero_count,
                "zero_percentage": round(
                    zero_count / len(df) * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# SKEWNESS ANALYSIS
# ============================================================

def get_skewness_summary(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:

    numeric_features = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_features.remove(target)

    result = df[numeric_features].skew()

    return (
        result
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
        .rename("skewness")
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )


# ============================================================
# DATASET REPORT
# ============================================================

def generate_dataset_report(
    df: pd.DataFrame,
    dataset_name: str,
    target: str,
) -> str:

    target_counts = df[target].value_counts().sort_index()

    target_percentages = (
        df[target]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    zero_summary = get_zero_value_summary(
        df,
        target,
    )

    skewness_summary = get_skewness_summary(
        df,
        target,
    )

    report = []

    report.append(
        f"# {dataset_name.title()} Dataset EDA\n"
    )

    report.append(
        f"## Dataset Shape\n\n"
        f"- Rows: {df.shape[0]}\n"
        f"- Columns: {df.shape[1]}\n"
    )

    report.append("## Missing Values\n")

    missing = df.isna().sum()

    for column, count in missing.items():

        if count > 0:
            report.append(
                f"- `{column}`: {count}\n"
            )

    if missing.sum() == 0:
        report.append("- No missing values detected.\n")

    report.append("\n## Duplicate Rows\n\n")

    report.append(
        f"- Duplicate rows: {df.duplicated().sum()}\n"
    )

    report.append("\n## Target Distribution\n\n")

    report.append("| Target | Count | Percentage |\n")
    report.append("|---:|---:|---:|\n")

    for value in target_counts.index:

        report.append(
            f"| {value} | "
            f"{target_counts[value]} | "
            f"{target_percentages[value]:.2f}% |\n"
        )

    report.append("\n## Zero Values\n\n")

    report.append(
        "| Feature | Zero Count | Zero Percentage |\n"
    )

    report.append("|---|---:|---:|\n")

    for _, row in zero_summary.iterrows():

        report.append(
            f"| {row['feature']} | "
            f"{row['zero_count']} | "
            f"{row['zero_percentage']:.2f}% |\n"
        )

    report.append("\n## Feature Skewness\n\n")

    report.append("| Feature | Skewness |\n")
    report.append("|---|---:|\n")

    for _, row in skewness_summary.iterrows():

        report.append(
            f"| {row['feature']} | "
            f"{row['skewness']:.3f} |\n"
        )

    return "".join(report)


# ============================================================
# MAIN
# ============================================================

def main():

    complete_report = []

    for dataset_name, config in DATASETS.items():

        print(
            f"\nRunning EDA for: {dataset_name}"
        )

        df = load_dataset(config["path"])

        target = config["target"]

        print(f"Shape: {df.shape}")
        print(
            f"Target distribution:\n"
            f"{df[target].value_counts()}"
        )

        print(
            f"Missing values:\n"
            f"{df.isna().sum()}"
        )

        print(
            f"Duplicate rows: "
            f"{df.duplicated().sum()}"
        )

        plot_target_distribution(
            df,
            dataset_name,
            target,
        )

        plot_feature_distributions(
            df,
            dataset_name,
            target,
        )

        plot_boxplots(
            df,
            dataset_name,
            target,
        )

        plot_correlation_matrix(
            df,
            dataset_name,
        )

        plot_feature_target_relationships(
            df,
            dataset_name,
            target,
        )

        complete_report.append(
            generate_dataset_report(
                df,
                dataset_name,
                target,
            )
        )

    report_path = (
        REPORTS_DIR / "eda_report.md"
    )

    report_path.write_text(
        "\n\n".join(complete_report),
        encoding="utf-8",
    )

    print(
        f"\nEDA completed."
        f"\nReport: {report_path}"
    )


if __name__ == "__main__":
    main()