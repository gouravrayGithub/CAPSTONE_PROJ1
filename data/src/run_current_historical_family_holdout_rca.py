from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

FEATURES = [
    "log_change_pct",
    "trace_change_pct",
    "median_change_pct",
    "p95_change_pct",
]

DATA_PATH = Path("data/incident_service_features.parquet")

OUTPUT_PATH = Path(
    "data/current_historical_family_holdout_rca_results.parquet"
)


# =========================================================
# 1. PARSE CASE NAME
# =========================================================

def parse_case(case_name):

    pattern = (
        r"re2ob_(.+)_(cpu|delay|disk|loss|mem|socket)_\d+$"
    )

    match = re.match(pattern, case_name)

    if not match:
        raise ValueError(f"Could not parse case: {case_name}")

    root_service = match.group(1)
    fault_type = match.group(2)

    return root_service, fault_type


# =========================================================
# 2. SIGNED LOG TRANSFORMATION
# =========================================================

def signed_log_transform(x):

    return np.sign(x) * np.log1p(np.abs(x))


# =========================================================
# 3. LOAD DATA
# =========================================================

print("Loading incident service features...")

df = pd.read_parquet(DATA_PATH)

print("Rows:", len(df))
print("Cases:", df["case"].nunique())


# =========================================================
# 4. ADD CASE METADATA
# =========================================================

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


# =========================================================
# 5. TRANSFORM CURRENT EVIDENCE
# =========================================================

for feature in FEATURES:

    transformed_name = feature + "_transformed"

    df[transformed_name] = (
        signed_log_transform(df[feature])
    )

    # Both increase and decrease can indicate abnormality.
    df[transformed_name] = (
        df[transformed_name].abs()
    )


TRANSFORMED_FEATURES = [
    f + "_transformed"
    for f in FEATURES
]


# =========================================================
# 6. BUILD CURRENT-INCIDENT VECTORS
# =========================================================

print("\nBuilding incident vectors...")

incident_vectors = {}

for case, group in df.groupby("case"):

    group = group.sort_values("service")

    vector = group[TRANSFORMED_FEATURES].fillna(0).values.flatten()

    incident_vectors[case] = vector


# =========================================================
# 7. NORMALIZE CURRENT EVIDENCE WITHIN INCIDENT
# =========================================================

def build_current_scores(group):

    scores = group[TRANSFORMED_FEATURES].sum(axis=1)

    scores = scores.astype(float)

    # Min-max normalization inside this incident
    min_score = scores.min()
    max_score = scores.max()

    if max_score > min_score:
        scores = (
            (scores - min_score)
            / (max_score - min_score)
        )
    else:
        scores[:] = 0.0

    return scores


# =========================================================
# 8. RUN RCA
# =========================================================

results = []

cases = sorted(df["case"].unique())

print("\n" + "=" * 60)
print("CURRENT + HISTORICAL FAMILY-HOLDOUT RCA")
print("=" * 60)
print("Cases:", len(cases))


for current_case in cases:

    current_group = df[
        df["case"] == current_case
    ].copy()

    true_root = current_group["root_service"].iloc[0]
    fault_type = current_group["fault_type"].iloc[0]


    # -----------------------------------------------------
    # A. CURRENT EVIDENCE
    # -----------------------------------------------------

    current_group["current_score"] = (
        build_current_scores(current_group)
    )


    current_scores = dict(
        zip(
            current_group["service"],
            current_group["current_score"]
        )
    )


    # -----------------------------------------------------
    # B. HISTORICAL SIMILARITY
    # -----------------------------------------------------

    current_vector = incident_vectors[current_case]

    historical_candidates = []

    for historical_case in cases:

        if historical_case == current_case:
            continue

        historical_meta = case_metadata[
            case_metadata["case"] == historical_case
        ].iloc[0]

        historical_root = historical_meta["root_service"]
        historical_fault = historical_meta["fault_type"]


        # -------------------------------------------------
        # LEAVE-ONE-FAULT-FAMILY-OUT
        # -------------------------------------------------

        if (
            historical_root == true_root
            and historical_fault == fault_type
        ):
            continue


        historical_vector = incident_vectors[
            historical_case
        ]


        similarity = cosine_similarity(
            current_vector.reshape(1, -1),
            historical_vector.reshape(1, -1)
        )[0][0]


        historical_candidates.append(
            {
                "case": historical_case,
                "similarity": float(similarity),
                "root_service": historical_root,
            }
        )


    historical_candidates = sorted(
        historical_candidates,
        key=lambda x: x["similarity"],
        reverse=True
    )


    # -----------------------------------------------------
    # C. HISTORICAL ROOT SUPPORT
    # -----------------------------------------------------

    root_support = {}

    for candidate in historical_candidates:

        root = candidate["root_service"]

        similarity = candidate["similarity"]

        if similarity <= 0:
            continue

        root_support[root] = (
            root_support.get(root, 0)
            + similarity
        )


    # Make sure every service exists
    all_roots = sorted(
        case_metadata["root_service"].unique()
    )

    for root in all_roots:

        root_support.setdefault(root, 0.0)


    historical_series = pd.Series(root_support)

    # Normalize historical evidence
    h_min = historical_series.min()
    h_max = historical_series.max()

    if h_max > h_min:

        historical_series = (
            (historical_series - h_min)
            / (h_max - h_min)
        )

    else:

        historical_series[:] = 0.0


    # -----------------------------------------------------
    # D. CURRENT ROOT SCORES
    # -----------------------------------------------------

    current_root_scores = {}

    for root in all_roots:

        if root in current_scores:

            current_root_scores[root] = (
                current_scores[root]
            )

        else:

            current_root_scores[root] = 0.0


    current_series = pd.Series(
        current_root_scores
    )


    # -----------------------------------------------------
    # E. COMBINE CURRENT + HISTORICAL
    # -----------------------------------------------------

    # Equal contribution initially.
    #
    # Do NOT tune this weight yet.
    # We want a clean first experiment.

    combined_series = (
        0.5 * current_series
        +
        0.5 * historical_series
    )


    ranked_roots = (
        combined_series
        .sort_values(ascending=False)
        .index
        .tolist()
    )


    # -----------------------------------------------------
    # F. METRICS
    # -----------------------------------------------------

    top1 = true_root in ranked_roots[:1]

    top3 = true_root in ranked_roots[:3]

    top5 = true_root in ranked_roots[:5]

    rank = ranked_roots.index(true_root) + 1

    reciprocal_rank = 1.0 / rank


    # -----------------------------------------------------
    # G. SAVE RESULT
    # -----------------------------------------------------

    top_neighbors = historical_candidates[:5]


    results.append(
        {
            "case": current_case,
            "true_root": true_root,
            "fault_type": fault_type,

            "predicted_root": ranked_roots[0],

            "top1": top1,
            "top3": top3,
            "top5": top5,

            "rank": rank,
            "reciprocal_rank": reciprocal_rank,

            "top_neighbors": "; ".join(
                f"{x['case']}:{x['similarity']:.4f}"
                for x in top_neighbors
            ),

            "ranked_roots": "; ".join(
                ranked_roots
            ),

            "current_root_score": float(
                current_series.get(true_root, 0)
            ),

            "historical_root_score": float(
                historical_series.get(true_root, 0)
            ),

            "combined_root_score": float(
                combined_series.get(true_root, 0)
            ),
        }
    )


# =========================================================
# 9. RESULTS
# =========================================================

results_df = pd.DataFrame(results)

top1_accuracy = results_df["top1"].mean()
top3_accuracy = results_df["top3"].mean()
top5_accuracy = results_df["top5"].mean()
mrr = results_df["reciprocal_rank"].mean()


print("\n" + "=" * 60)
print("RESULTS")
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


# =========================================================
# 10. SAVE
# =========================================================

results_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)


# =========================================================
# 11. SAMPLE PREDICTIONS
# =========================================================

print("\nSample predictions:\n")

for _, row in results_df.head(10).iterrows():

    print(
        f"{row['case']} | "
        f"True={row['true_root']} | "
        f"Predicted={row['predicted_root']} | "
        f"Fault={row['fault_type']}"
    )

    print(
        f"  Neighbours: {row['top_neighbors']}"
    )