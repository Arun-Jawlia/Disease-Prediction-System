import pandas as pd


def select_best_models(
    results_df: pd.DataFrame,
):

    selected = {}

    for disease in results_df["dataset"].unique():

        disease_results = results_df[
            results_df["dataset"] == disease
        ].copy()

        # Screening-oriented ordering.
        disease_results = disease_results.sort_values(
            by=[
                "recall",
                "f1",
                "roc_auc",
                "precision",
                "accuracy",
            ],
            ascending=False,
        )

        selected[disease] = (
            disease_results.iloc[0]
        )

    return selected


def generate_ranking_table(
    results_df: pd.DataFrame,
):

    ranking = results_df.copy()

    ranking["rank"] = (
        ranking
        .groupby("dataset")["f1"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    return ranking.sort_values(
        ["dataset", "rank"]
    )