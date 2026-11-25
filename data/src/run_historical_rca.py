from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")

FEATURES = [
    "log_change_pct",
    "trace_change_pct",
    "median_change_pct",
    "p95_change_pct",
]


def signed_log(x):
    """
    Compress very large percentage changes.
    """
    return np.sign(x) * np.log1p(abs(x))


def transform_features(df):
    """
    Transform highly skewed percentage changes.
    """

    df = df.copy()

    for feature in FEATURES:
        df[feature + "_transformed"] = signed_log(
            df[feature].fillna(0)
        )

    return df


def create_case_vectors(df):
    """
    Convert service-level evidence into
    one vector per incident.

    Services are sorted so that every
    incident has the same vector layout.
    """

    services = sorted(
        df["service"].dropna().unique()
    )

    transformed_features = [
        feature + "_transformed"
        for feature in FEATURES
    ]

    vectors = {}
    metadata = {}

    for case, group in df.groupby("case"):

        group = group.set_index("service")

        values = []

        for service in services:

            if service in group.index:

                for feature in transformed_features:

                    values.append(
                        group.loc[
                            service,
                            feature
                        ]
                    )

            else:

                values.extend(
                    [0.0] * len(transformed_features)
                )

        vectors[case] = np.array(
            values,
            dtype=float
        )

        # Root service is derived from
        # the case name and used only for evaluation.
        parts = case.replace(
            "re2ob_",
            ""
        ).split("_")

        root_service = "_".join(
            parts[:-2]
        )

        metadata[case] = {
            "root_service": root_service
        }

    return vectors, metadata


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def historical_neighbors(
    current_case,
    vectors,
    metadata,
    k=5
):

    current_vector = vectors[
        current_case
    ]

    similarities = []

    for case, vector in vectors.items():

        # -----------------------------------------------------
        # LEAVE-ONE-OUT
        # -----------------------------------------------------

        if case == current_case:
            continue

        similarity = cosine_similarity(
            current_vector,
            vector
        )

        similarities.append(
            (
                case,
                similarity,
                metadata[case]["root_service"]
            )
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:k]


def historical_root_scores(
    neighbors
):

    scores = {}

    for case, similarity, root in neighbors:

        # Only positive similarity contributes.
        weight = max(similarity, 0)

        scores[root] = (
            scores.get(root, 0)
            + weight
        )

    return scores


def main():

    feature_file = (
        DATA_DIR
        / "incident_service_features.parquet"
    )

    df = pd.read_parquet(
        feature_file
    )

    print(
        "Loaded service features:",
        df.shape
    )

    # ---------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------

    df = transform_features(df)

    # ---------------------------------------------------------
    # CASE VECTORS
    # ---------------------------------------------------------

    vectors, metadata = create_case_vectors(
        df
    )

    print(
        "Historical incidents:",
        len(vectors)
    )

    # ---------------------------------------------------------
    # RUN LEAVE-ONE-OUT MATCHING
    # ---------------------------------------------------------

    result_rows = []

    cases = sorted(vectors.keys())

    for i, current_case in enumerate(
        cases,
        start=1
    ):

        neighbors = historical_neighbors(
            current_case,
            vectors,
            metadata,
            k=5
        )

        root_scores = historical_root_scores(
            neighbors
        )

        ranked_roots = sorted(
            root_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        root_ranking = [
            root
            for root, score in ranked_roots
        ]

        true_root = metadata[
            current_case
        ]["root_service"]

        # -----------------------------------------------------
        # HISTORICAL TOP-1
        # -----------------------------------------------------

        if len(root_ranking) > 0:

            predicted_root = (
                root_ranking[0]
            )

        else:

            predicted_root = None

        # -----------------------------------------------------
        # SAVE NEIGHBORS
        # -----------------------------------------------------

        row = {
            "case": current_case,
            "true_root": true_root,
            "historical_predicted_root":
                predicted_root,
        }

        for rank, (
            neighbor_case,
            similarity,
            neighbor_root
        ) in enumerate(
            neighbors,
            start=1
        ):

            row[
                f"neighbor_{rank}"
            ] = neighbor_case

            row[
                f"similarity_{rank}"
            ] = similarity

            row[
                f"neighbor_root_{rank}"
            ] = neighbor_root

        # Root support
        for root in [
            "checkoutservice",
            "currencyservice",
            "emailservice",
            "productcatalogservice",
            "recommendationservice",
        ]:

            row[
                f"historical_score_{root}"
            ] = root_scores.get(
                root,
                0
            )

        result_rows.append(row)

        if i <= 5:

            print()
            print(
                f"[{i}] {current_case}"
            )

            print(
                "True root:",
                true_root
            )

            print(
                "Top historical root:",
                predicted_root
            )

            print(
                "Neighbors:"
            )

            for neighbor in neighbors:

                print(
                    "  ",
                    neighbor[0],
                    "| similarity =",
                    round(
                        neighbor[1],
                        4
                    ),
                    "| root =",
                    neighbor[2]
                )

    results = pd.DataFrame(
        result_rows
    )

    # ---------------------------------------------------------
    # EVALUATE HISTORICAL-ONLY METHOD
    # ---------------------------------------------------------

    top1 = 0
    top3 = 0

    reciprocal_ranks = []

    for _, row in results.iterrows():

        current_case = row["case"]
        true_root = row["true_root"]

        # Reconstruct historical root ranking
        root_scores = {}

        for root in [
            "checkoutservice",
            "currencyservice",
            "emailservice",
            "productcatalogservice",
            "recommendationservice",
        ]:

            root_scores[root] = row[
                f"historical_score_{root}"
            ]

        ranked_roots = [
            root
            for root, score in sorted(
                root_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        if ranked_roots[0] == true_root:
            top1 += 1

        if true_root in ranked_roots[:3]:
            top3 += 1

        rank = (
            ranked_roots.index(
                true_root
            )
            + 1
        )

        reciprocal_ranks.append(
            1 / rank
        )

    number_cases = len(results)

    top1_accuracy = (
        top1 / number_cases
    )

    top3_accuracy = (
        top3 / number_cases
    )

    mrr = np.mean(
        reciprocal_ranks
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output = (
        DATA_DIR
        / "historical_rca_results.parquet"
    )

    results.to_parquet(
        output,
        index=False
    )

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("HISTORICAL RCA — LEAVE-ONE-OUT")
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
