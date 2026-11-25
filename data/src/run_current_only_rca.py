from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data")


def signed_log(x):
    """
    Compress very large positive/negative percentage changes.

    sign(x) * log(1 + abs(x))
    """
    return np.sign(x) * np.log1p(abs(x))


def normalize_scores(series):
    """
    Convert scores to 0-1 range within one incident.
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def calculate_score(group):

    group = group.copy()

    # ---------------------------------------------------------
    # TRANSFORM EVIDENCE
    # ---------------------------------------------------------

    group["log_signal"] = signed_log(
        group["log_change_pct"]
    )

    group["trace_signal"] = signed_log(
        group["trace_change_pct"]
    )

    group["median_signal"] = signed_log(
        group["median_change_pct"]
    )

    group["p95_signal"] = signed_log(
        group["p95_change_pct"]
    )

    # ---------------------------------------------------------
    # ABSOLUTE ANOMALY EVIDENCE
    # ---------------------------------------------------------

    group["log_signal"] = group[
        "log_signal"
    ].abs()

    group["trace_signal"] = group[
        "trace_signal"
    ].abs()

    group["median_signal"] = group[
        "median_signal"
    ].abs()

    group["p95_signal"] = group[
        "p95_signal"
    ].abs()

    # ---------------------------------------------------------
    # CURRENT-ONLY SCORE
    # ---------------------------------------------------------

    group["raw_score"] = (
        group["log_signal"]
        + group["trace_signal"]
        + group["median_signal"]
        + group["p95_signal"]
    )

    # Normalize for easier comparison
    group["score"] = normalize_scores(
        group["raw_score"]
    )

    return group


def main():

    features_file = (
        DATA_DIR
        / "incident_service_features.parquet"
    )

    df = pd.read_parquet(
        features_file
    )

    results = []

    # ---------------------------------------------------------
    # PROCESS EACH INCIDENT
    # ---------------------------------------------------------

    for case, group in df.groupby(
        "case"
    ):

        group = calculate_score(
            group
        )

        # -----------------------------------------------------
        # GROUND TRUTH
        # -----------------------------------------------------

        parts = case.replace(
            "re2ob_",
            ""
        ).split("_")

        root_service = "_".join(
            parts[:-2]
        )

        group["ground_truth"] = (
            group["service"]
            == root_service
        )

        # -----------------------------------------------------
        # RANK
        # -----------------------------------------------------

        group = group.sort_values(
            "score",
            ascending=False
        ).reset_index(drop=True)

        group["rank"] = (
            np.arange(len(group)) + 1
        )

        group["case_root_service"] = (
            root_service
        )

        results.append(group)

    results = pd.concat(
        results,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    case_groups = results.groupby(
        "case"
    )

    top1 = 0
    top3 = 0
    top5 = 0

    reciprocal_ranks = []

    for case, group in case_groups:

        group = group.sort_values(
            "rank"
        )

        root_service = (
            group["case_root_service"]
            .iloc[0]
        )

        ranked_services = (
            group["service"]
            .tolist()
        )

        if (
            len(ranked_services) > 0
            and ranked_services[0]
            == root_service
        ):
            top1 += 1

        if root_service in ranked_services[:3]:
            top3 += 1

        if root_service in ranked_services[:5]:
            top5 += 1

        if root_service in ranked_services:

            rank = (
                ranked_services.index(
                    root_service
                )
                + 1
            )

            reciprocal_ranks.append(
                1 / rank
            )

    number_cases = (
        results["case"]
        .nunique()
    )

    top1_accuracy = (
        top1 / number_cases
    )

    top3_accuracy = (
        top3 / number_cases
    )

    top5_accuracy = (
        top5 / number_cases
    )

    mrr = np.mean(
        reciprocal_ranks
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output = (
        DATA_DIR
        / "current_only_rca_results.parquet"
    )

    results.to_parquet(
        output,
        index=False
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("CURRENT-ONLY RCA BASELINE")
    print("=" * 70)

    print(
        "Cases:",
        number_cases
    )

    print(
        "Top-1 Accuracy:",
        round(top1_accuracy, 4)
    )

    print(
        "Top-3 Accuracy:",
        round(top3_accuracy, 4)
    )

    print(
        "Top-5 Accuracy:",
        round(top5_accuracy, 4)
    )

    print(
        "MRR:",
        round(mrr, 4)
    )

    print()
    print(
        "Results saved to:"
    )

    print(output)


if __name__ == "__main__":
    main()
