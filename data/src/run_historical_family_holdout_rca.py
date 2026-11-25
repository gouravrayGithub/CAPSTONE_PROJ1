from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


FEATURES = [
    "log_change_pct",
    "trace_change_pct",
    "median_change_pct",
    "p95_change_pct",
]

DATA_PATH = Path("data/incident_service_features.parquet")
OUTPUT_PATH = Path("data/historical_family_holdout_rca_results.parquet")


def parse_case(case_name):
    """
    Example:
    re2ob_checkoutservice_cpu_1

    Returns:
        root_service = checkoutservice
        fault_type = cpu
    """
    match = re.match(
        r"re2ob_(.+)_(cpu|delay|disk|loss|mem|socket)_\d+$",
        case_name
    )

    if not match:
        raise ValueError(f"Could not parse case name: {case_name}")

    return match.group(1), match.group(2)


def signed_log_transform(x):
    """
    Compress very large percentage changes while preserving direction.
    """
    return np.sign(x) * np.log1p(np.abs(x))


print("Loading incident service features...")

df = pd.read_parquet(DATA_PATH)

print("Rows:", len(df))
print("Cases:", df["case"].nunique())

# ---------------------------------------------------------
# 1. Add root service and fault type
# ---------------------------------------------------------

case_metadata = pd.DataFrame({
    "case": df["case"].drop_duplicates()
})

case_metadata[["root_service", "fault_type"]] = (
    case_metadata["case"]
    .apply(parse_case)
    .apply(pd.Series)
)

df = df.merge(
    case_metadata,
    on="case",
    how="left"
)



# ---------------------------------------------------------
# 2. Transform feature values
# ---------------------------------------------------------

for feature in FEATURES:
    df[feature] = df[feature].fillna(0)

    df[feature + "_transformed"] = (
        df[feature]
        .apply(signed_log_transform)
    )


TRANSFORMED_FEATURES = [
    f + "_transformed"
    for f in FEATURES
]


# ---------------------------------------------------------
# 3. Create one vector per incident
# ---------------------------------------------------------

services = sorted(df["service"].unique())

cases = sorted(df["case"].unique())

vectors = {}
metadata_lookup = {}

for case in cases:

    case_df = df[df["case"] == case].copy()

    # Stable service ordering
    case_df = (
        case_df
        .set_index("service")
        .reindex(services)
        .fillna(0)
    )

    vector = case_df[TRANSFORMED_FEATURES].to_numpy().flatten()

    vectors[case] = vector

    row = case_metadata[
        case_metadata["case"] == case
    ].iloc[0]

    metadata_lookup[case] = {
        "root_service": row["root_service"],
        "fault_type": row["fault_type"],
    }


# ---------------------------------------------------------
# 4. Leave-One-Fault-Family-Out RCA
# ---------------------------------------------------------

results = []

for current_case in cases:

    current_root = metadata_lookup[current_case]["root_service"]
    current_fault = metadata_lookup[current_case]["fault_type"]

    current_vector = vectors[current_case].reshape(1, -1)

    candidate_cases = []

    for historical_case in cases:

        if historical_case == current_case:
            continue

        historical_root = metadata_lookup[
            historical_case
        ]["root_service"]

        historical_fault = metadata_lookup[
            historical_case
        ]["fault_type"]

        # -------------------------------------------------
        # Critical holdout:
        # exclude the complete root-service + fault family
        # -------------------------------------------------

        if (
            historical_root == current_root
            and historical_fault == current_fault
        ):
            continue

        candidate_cases.append(historical_case)

    historical_matrix = np.vstack(
        [vectors[c] for c in candidate_cases]
    )

    similarities = cosine_similarity(
        current_vector,
        historical_matrix
    )[0]

    similarity_df = pd.DataFrame({
        "case": candidate_cases,
        "similarity": similarities
    })

    similarity_df = similarity_df.sort_values(
        "similarity",
        ascending=False
    )

    # -----------------------------------------------------
    # Top historical neighbours
    # -----------------------------------------------------

    top_neighbors = similarity_df.head(5)

    # -----------------------------------------------------
    # Aggregate evidence by historical root service
    # -----------------------------------------------------

    root_support = {}

    for _, row in top_neighbors.iterrows():

        historical_case = row["case"]
        similarity = row["similarity"]

        historical_root = metadata_lookup[
            historical_case
        ]["root_service"]

        # Only positive similarity contributes support
        if similarity > 0:

            root_support[historical_root] = (
                root_support.get(historical_root, 0)
                + similarity
            )

    ranked_roots = sorted(
        root_support.items(),
        key=lambda x: x[1],
        reverse=True
    )

    predictions = [
        root for root, score in ranked_roots
    ]

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    top1 = (
        predictions[0]
        if len(predictions) >= 1
        else None
    )

    top3 = predictions[:3]
    top5 = predictions[:5]

    reciprocal_rank = 0.0

    if current_root in predictions:

        rank = predictions.index(current_root) + 1

        reciprocal_rank = 1.0 / rank

    results.append({
        "case": current_case,
        "true_root": current_root,
        "fault_type": current_fault,
        "predicted_root": top1,

        "top1_correct": (
            top1 == current_root
        ),

        "top3_correct": (
            current_root in top3
        ),

        "top5_correct": (
            current_root in top5
        ),

        "reciprocal_rank": reciprocal_rank,

       "top_neighbors": "; ".join(
    f"{row['case']}:{row['similarity']:.4f}"
    for _, row in top_neighbors.iterrows()
),

"ranked_roots": "; ".join(predictions),
    })


# ---------------------------------------------------------
# 5. Results dataframe
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# 6. Overall metrics
# ---------------------------------------------------------

top1_accuracy = results_df["top1_correct"].mean()
top3_accuracy = results_df["top3_correct"].mean()
top5_accuracy = results_df["top5_correct"].mean()
mrr = results_df["reciprocal_rank"].mean()


print()
print("=" * 60)
print("LEAVE-ONE-FAULT-FAMILY-OUT HISTORICAL RCA")
print("=" * 60)

print(f"Cases: {len(results_df)}")

print(
    f"Top-1 Accuracy: {top1_accuracy:.4f}"
)

print(
    f"Top-3 Accuracy: {top3_accuracy:.4f}"
)

print(
    f"Top-5 Accuracy: {top5_accuracy:.4f}"
)

print(
    f"MRR: {mrr:.4f}"
)

print()
print("Results saved to:")
print(OUTPUT_PATH)


# ---------------------------------------------------------
# 7. Show a few examples
# ---------------------------------------------------------

print()
print("Sample predictions:")
print()

for _, row in results_df.head(10).iterrows():

    print(
        f"{row['case']} | "
        f"True={row['true_root']} | "
        f"Predicted={row['predicted_root']} | "
        f"Fault={row['fault_type']}"
    )

    print(
        "  Neighbours:",
        row["top_neighbors"]
    )