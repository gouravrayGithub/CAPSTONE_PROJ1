from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path(
    "data/incident_service_features.parquet"
)

OUTPUT_PATH = Path(
    "data/partial_observation_rca_results.parquet"
)

FEATURES = [
    "log_change_pct",
    "trace_change_pct",
    "median_change_pct",
    "p95_change_pct",
]

OBSERVATION_LEVELS = [
    1.0,
    0.8,
    0.6,
    0.4,
    0.2,
]

RANDOM_SEED = 42


# =========================================================
# CASE PARSER
# =========================================================

def parse_case(case_name):

    pattern = (
        r"re2ob_(.+)_(cpu|delay|disk|loss|mem|socket)_\d+$"
    )

    match = re.match(pattern, case_name)

    if not match:
        raise ValueError(
            f"Could not parse case: {case_name}"
        )

    return match.group(1), match.group(2)


# =========================================================
# TRANSFORMATION
# =========================================================

def signed_log_transform(x):

    return np.sign(x) * np.log1p(np.abs(x))


# =========================================================
# LOAD DATA
# =========================================================

print("Loading incident service features...")

df = pd.read_parquet(DATA_PATH)

print("Rows:", len(df))
print("Cases:", df["case"].nunique())


# =========================================================
# CASE METADATA
# =========================================================

case_metadata = pd.DataFrame({
    "case": df["case"].drop_duplicates()
})

case_metadata[
    ["root_service", "fault_type"]
] = (
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
# TRANSFORM FEATURES
# =========================================================

for feature in FEATURES:

    transformed = feature + "_transformed"

    df[transformed] = (
        signed_log_transform(
            df[feature]
        )
        .abs()
    )


TRANSFORMED_FEATURES = [
    feature + "_transformed"
    for feature in FEATURES
]


# =========================================================
# GLOBAL SERVICE ORDER
# =========================================================
#
# IMPORTANT:
# Every incident vector must use exactly the same
# service ordering.
#
# This allows an unobserved service to be represented
# by zeros without changing vector dimensions.
# =========================================================

ALL_SERVICES = sorted(
    df["service"].unique()
)

print("\nServices:", len(ALL_SERVICES))
print("Service order:")

for service in ALL_SERVICES:
    print(" ", service)


# =========================================================
# BUILD COMPLETE INCIDENT VECTORS
# =========================================================

print("\nBuilding complete incident vectors...")

incident_vectors = {}

incident_services = {}

for case, group in df.groupby("case"):

    group = group.set_index("service")

    vector_values = []

    for service in ALL_SERVICES:

        if service in group.index:

            row = group.loc[service]

            values = [
                row[feature]
                for feature in TRANSFORMED_FEATURES
            ]

        else:

            values = [
                0.0
                for _ in TRANSFORMED_FEATURES
            ]

        vector_values.extend(values)

    incident_vectors[case] = np.array(
        vector_values,
        dtype=float
    )

    incident_services[case] = (
        group.index.tolist()
    )


# =========================================================
# CURRENT SERVICE SCORE
# =========================================================

def service_scores(group):

    scores = (
        group[TRANSFORMED_FEATURES]
        .sum(axis=1)
        .astype(float)
    )

    if scores.max() > scores.min():

        scores = (
            (scores - scores.min())
            /
            (scores.max() - scores.min())
        )

    else:

        scores[:] = 0.0

    return scores


# =========================================================
# RESULTS
# =========================================================

results = []

cases = sorted(
    df["case"].unique()
)

rng = np.random.default_rng(
    RANDOM_SEED
)


print("\n" + "=" * 60)
print("PARTIAL OBSERVATION RCA")
print("=" * 60)

print(
    "Observation levels:",
    OBSERVATION_LEVELS
)


# =========================================================
# MAIN LOOP
# =========================================================

for observation_level in OBSERVATION_LEVELS:

    print(
        f"\nRunning observation level: "
        f"{observation_level:.0%}"
    )

    for current_case in cases:

        current_group = df[
            df["case"] == current_case
        ].copy()

        true_root = (
            current_group["root_service"]
            .iloc[0]
        )

        fault_type = (
            current_group["fault_type"]
            .iloc[0]
        )


        # -------------------------------------------------
        # SELECT OBSERVED SERVICES
        # -------------------------------------------------

        all_services = (
            current_group["service"]
            .tolist()
        )

        n_services = max(
            1,
            int(
                np.ceil(
                    len(all_services)
                    * observation_level
                )
            )
        )


        # Randomly select observed services
        selected_services = rng.choice(
            all_services,
            size=n_services,
            replace=False
        ).tolist()


        # -------------------------------------------------
        # STAGE A:
        # KEEP THE TRUE ROOT VISIBLE
        # -------------------------------------------------

        if true_root not in selected_services:

            selected_services[0] = true_root


        partial_group = current_group[
            current_group["service"].isin(
                selected_services
            )
        ].copy()


        # -------------------------------------------------
        # CURRENT EVIDENCE
        # -------------------------------------------------

        partial_group["current_score"] = (
            service_scores(
                partial_group
            )
        )

        current_scores = dict(
            zip(
                partial_group["service"],
                partial_group["current_score"]
            )
        )


        # -------------------------------------------------
        # BUILD PARTIAL CURRENT VECTOR
        # -------------------------------------------------
        #
        # IMPORTANT:
        #
        # Observed service:
        #     use its actual evidence
        #
        # Unobserved service:
        #     use zeros
        #
        # The vector still contains ALL services in
        # exactly the same order as historical vectors.
        # -------------------------------------------------

        partial_lookup = (
            partial_group
            .set_index("service")
        )

        partial_values = []

        for service in ALL_SERVICES:

            if service in partial_lookup.index:

                row = partial_lookup.loc[service]

                values = [
                    row[feature]
                    for feature in TRANSFORMED_FEATURES
                ]

            else:

                # Service not observed
                values = [
                    0.0
                    for _ in TRANSFORMED_FEATURES
                ]

            partial_values.extend(values)


        current_vector = np.array(
            partial_values,
            dtype=float
        )


        # -------------------------------------------------
        # HISTORICAL SIMILARITY
        # -------------------------------------------------

        historical_candidates = []


        for historical_case in cases:

            if historical_case == current_case:
                continue


            metadata = case_metadata[
                case_metadata["case"]
                == historical_case
            ].iloc[0]


            historical_root = (
                metadata["root_service"]
            )

            historical_fault = (
                metadata["fault_type"]
            )


            # -------------------------------------------------
            # LEAVE-ONE-FAULT-FAMILY-OUT
            # -------------------------------------------------

            if (
                historical_root == true_root
                and historical_fault == fault_type
            ):
                continue


            historical_vector = (
                incident_vectors[
                    historical_case
                ]
            )


            similarity = cosine_similarity(
                current_vector.reshape(1, -1),
                historical_vector.reshape(1, -1)
            )[0][0]


            historical_candidates.append(
                {
                    "case": historical_case,
                    "similarity": float(
                        similarity
                    ),
                    "root_service":
                        historical_root,
                }
            )


        # -------------------------------------------------
        # SORT HISTORICAL NEIGHBOURS
        # -------------------------------------------------

        historical_candidates.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )


        # -------------------------------------------------
        # HISTORICAL ROOT SUPPORT
        # -------------------------------------------------

        historical_support = {}


        for candidate in historical_candidates:

            root = candidate[
                "root_service"
            ]

            similarity = candidate[
                "similarity"
            ]

            historical_support[root] = (
                historical_support.get(
                    root,
                    0.0
                )
                + similarity
            )


        all_roots = sorted(
            case_metadata[
                "root_service"
            ].unique()
        )


        for root in all_roots:

            historical_support.setdefault(
                root,
                0.0
            )


        historical_series = pd.Series(
            historical_support,
            dtype=float
        )


        # -------------------------------------------------
        # NORMALIZE HISTORICAL SUPPORT
        # -------------------------------------------------

        if (
            historical_series.max()
            >
            historical_series.min()
        ):

            historical_series = (
                (
                    historical_series
                    - historical_series.min()
                )
                /
                (
                    historical_series.max()
                    - historical_series.min()
                )
            )

        else:

            historical_series[:] = 0.0


        # -------------------------------------------------
        # CURRENT ROOT SUPPORT
        # -------------------------------------------------

        current_root_support = {}

        for root in all_roots:

            current_root_support[root] = (
                current_scores.get(
                    root,
                    0.0
                )
            )


        current_series = pd.Series(
            current_root_support,
            dtype=float
        )


        # -------------------------------------------------
        # RANK CURRENT EVIDENCE
        # -------------------------------------------------

        current_ranked = (
            current_series
            .sort_values(
                ascending=False,
                kind="stable"
            )
            .index
            .tolist()
        )


        # -------------------------------------------------
        # RANK HISTORICAL EVIDENCE
        # -------------------------------------------------

        historical_ranked = (
            historical_series
            .sort_values(
                ascending=False,
                kind="stable"
            )
            .index
            .tolist()
        )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        current_rank = (
            current_ranked.index(
                true_root
            ) + 1
        )

        historical_rank = (
            historical_ranked.index(
                true_root
            ) + 1
        )


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        results.append(
            {
                "case": current_case,

                "observation_level":
                    observation_level,

                "true_root":
                    true_root,

                "fault_type":
                    fault_type,

                "observed_services":
                    ";".join(
                        sorted(
                            selected_services
                        )
                    ),

                "n_observed_services":
                    len(
                        selected_services
                    ),

                # -----------------------------
                # Current evidence
                # -----------------------------

                "current_predicted":
                    current_ranked[0],

                "current_rank":
                    current_rank,

                "current_top1":
                    current_rank == 1,

                "current_top3":
                    current_rank <= 3,

                "current_top5":
                    current_rank <= 5,

                "current_mrr":
                    1.0 / current_rank,

                # -----------------------------
                # Historical evidence
                # -----------------------------

                "historical_predicted":
                    historical_ranked[0],

                "historical_rank":
                    historical_rank,

                "historical_top1":
                    historical_rank == 1,

                "historical_top3":
                    historical_rank <= 3,

                "historical_top5":
                    historical_rank <= 5,

                "historical_mrr":
                    1.0 / historical_rank,

                # -----------------------------
                # Neighbour information
                # -----------------------------

                "top_neighbors":
                    "; ".join(
                        f"{x['case']}:{x['similarity']:.4f}"
                        for x in historical_candidates[:5]
                    ),
            }
        )


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# SUMMARY
# =========================================================

summary_rows = []


for level, group in (
    results_df
    .groupby("observation_level")
):

    summary_rows.append(
        {
            "observation_level":
                level,

            "cases":
                len(group),

            "current_top1":
                group[
                    "current_top1"
                ].mean(),

            "current_top3":
                group[
                    "current_top3"
                ].mean(),

            "current_top5":
                group[
                    "current_top5"
                ].mean(),

            "current_mrr":
                group[
                    "current_mrr"
                ].mean(),

            "historical_top1":
                group[
                    "historical_top1"
                ].mean(),

            "historical_top3":
                group[
                    "historical_top3"
                ].mean(),

            "historical_top5":
                group[
                    "historical_top5"
                ].mean(),

            "historical_mrr":
                group[
                    "historical_mrr"
                ].mean(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("PARTIAL OBSERVATION SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "observation_level":
                lambda x: f"{x:.0%}",

            "current_top1":
                lambda x: f"{x:.4f}",

            "current_top3":
                lambda x: f"{x:.4f}",

            "current_top5":
                lambda x: f"{x:.4f}",

            "current_mrr":
                lambda x: f"{x:.4f}",

            "historical_top1":
                lambda x: f"{x:.4f}",

            "historical_top3":
                lambda x: f"{x:.4f}",

            "historical_top5":
                lambda x: f"{x:.4f}",

            "historical_mrr":
                lambda x: f"{x:.4f}",
        }
    )
)


# =========================================================
# SAVE
# =========================================================

results_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nDetailed results saved to:")
print(OUTPUT_PATH)