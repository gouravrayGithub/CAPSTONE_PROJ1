from pathlib import Path
import pickle
import math

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

PARTIAL_DIR = (
    DATA_DIR / "partial_incident_graphs"
)

FEATURE_FILE = (
    DATA_DIR / "incident_service_features.parquet"
)

OBSERVATION_LEVELS = [
    100,
    80,
    60,
    40,
    20,
]

MAX_HOPS = 3


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# CASE PARSER
# ============================================================

def parse_case(case):

    parts = case.split("_")

    root_service = parts[1]
    fault_type = parts[2]

    return root_service, fault_type


# ============================================================
# SIGNED LOG TRANSFORMATION
# ============================================================

def transform_change(x):

    if pd.isna(x):
        return 0.0

    x = float(x)

    return math.copysign(
        math.log1p(abs(x)),
        x
    )


# ============================================================
# BUILD SERVICE ANOMALY EVIDENCE
# ============================================================

def build_service_evidence(
    feature_df,
    case
):

    case_df = feature_df[
        feature_df["case"] == case
    ].copy()

    evidence = {}

    for _, row in case_df.iterrows():

        values = []

        for column in [
            "log_change_pct",
            "trace_change_pct",
            "median_change_pct",
            "p95_change_pct",
        ]:

            value = transform_change(
                row[column]
            )

            # Magnitude of change
            values.append(
                abs(value)
            )

        # ----------------------------------------------------
        # Average anomaly magnitude
        # ----------------------------------------------------

        evidence[
            row["service"]
        ] = sum(values) / len(values)

    return evidence


# ============================================================
# STRUCTURAL GRAPH SCORE
# ============================================================

def graph_match_score(
    current_graph,
    historical_graph
):

    current_nodes = set(
        current_graph.nodes()
    )

    historical_nodes = set(
        historical_graph.nodes()
    )

    current_edges = set(
        current_graph.edges()
    )

    historical_edges = set(
        historical_graph.edges()
    )


    # --------------------------------------------------------
    # Node overlap
    # --------------------------------------------------------

    if current_nodes:

        node_score = (
            len(
                current_nodes
                &
                historical_nodes
            )
            /
            len(current_nodes)
        )

    else:

        node_score = 0.0


    # --------------------------------------------------------
    # Edge overlap
    # --------------------------------------------------------

    if current_edges:

        edge_score = (
            len(
                current_edges
                &
                historical_edges
            )
            /
            len(current_edges)
        )

    else:

        edge_score = 0.0


    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    if current_edges:

        structural_score = (
            node_score
            +
            edge_score
        ) / 2.0

    else:

        structural_score = node_score


    return structural_score


# ============================================================
# FIND UPSTREAM CANDIDATES
# ============================================================

def find_upstream_candidates(
    current_graph,
    historical_graph,
    max_hops=3
):

    current_nodes = set(
        current_graph.nodes()
    )

    candidates = {}

    queue = []


    # --------------------------------------------------------
    # Start from observed services
    # --------------------------------------------------------

    for node in current_nodes:

        if node in historical_graph:

            queue.append(
                (
                    node,
                    0
                )
            )


    visited = set()


    # ========================================================
    # BACKWARD BFS
    # ========================================================

    while queue:

        node, distance = (
            queue.pop(0)
        )

        state = (
            node,
            distance
        )

        if state in visited:

            continue

        visited.add(state)


        if distance >= max_hops:

            continue


        if node not in historical_graph:

            continue


        for predecessor in (
            historical_graph.predecessors(node)
        ):

            new_distance = (
                distance + 1
            )


            # ------------------------------------------------
            # Only hidden services become candidates
            # ------------------------------------------------

            if predecessor not in current_nodes:

                if (
                    predecessor
                    not in candidates
                ):

                    candidates[
                        predecessor
                    ] = new_distance

                else:

                    candidates[
                        predecessor
                    ] = min(
                        candidates[
                            predecessor
                        ],
                        new_distance
                    )


            queue.append(
                (
                    predecessor,
                    new_distance
                )
            )


    return candidates


# ============================================================
# FIND DOWNSTREAM OBSERVED SERVICES
# ============================================================

def candidate_downstream_evidence(
    candidate,
    current_graph,
    historical_graph,
    anomaly_evidence,
    max_hops=3
):

    current_nodes = set(
        current_graph.nodes()
    )

    queue = [
        (
            candidate,
            0
        )
    ]

    visited = set()

    observed_reachable = []


    # ========================================================
    # FORWARD BFS
    # ========================================================

    while queue:

        node, distance = (
            queue.pop(0)
        )

        state = (
            node,
            distance
        )

        if state in visited:

            continue

        visited.add(state)


        if distance >= max_hops:

            continue


        if node not in historical_graph:

            continue


        for successor in (
            historical_graph.successors(node)
        ):

            new_distance = (
                distance + 1
            )


            # ------------------------------------------------
            # If successor is currently observed, it provides
            # evidence for this candidate.
            # ------------------------------------------------

            if successor in current_nodes:

                evidence = (
                    anomaly_evidence.get(
                        successor,
                        0.0
                    )
                )

                observed_reachable.append(
                    {
                        "service":
                            successor,

                        "distance":
                            new_distance,

                        "evidence":
                            evidence,
                    }
                )


            # Continue downstream
            queue.append(
                (
                    successor,
                    new_distance
                )
            )


    return observed_reachable


# ============================================================
# CANDIDATE EVIDENCE SCORE
# ============================================================

def calculate_candidate_evidence(
    downstream_services
):

    if not downstream_services:

        return 0.0


    numerator = 0.0

    denominator = 0.0


    for item in downstream_services:

        distance = item[
            "distance"
        ]

        evidence = item[
            "evidence"
        ]


        # ----------------------------------------------------
        # Closer observed descendants contribute more.
        # ----------------------------------------------------

        weight = (
            1.0
            /
            distance
        )


        numerator += (
            weight
            *
            evidence
        )

        denominator += weight


    if denominator == 0:

        return 0.0


    return (
        numerator
        /
        denominator
    )


# ============================================================
# CANDIDATE SCORE
# ============================================================

def calculate_candidate_score(
    structural_score,
    candidate_distance,
    evidence_score
):

    # --------------------------------------------------------
    # Upstream distance penalty
    # --------------------------------------------------------

    distance_factor = (
        1.0
        /
        (1.0 + candidate_distance)
    )


    # --------------------------------------------------------
    # Candidate-specific evidence
    # --------------------------------------------------------

    evidence_factor = (
        1.0
        +
        evidence_score
    )


    return (
        structural_score
        *
        distance_factor
        *
        evidence_factor
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CANDIDATE-SPECIFIC EVIDENCE RCA")
print("=" * 70)

print(
    "Maximum upstream hops:",
    MAX_HOPS
)


# ============================================================
# LOAD FEATURES
# ============================================================

feature_df = pd.read_parquet(
    FEATURE_FILE
)


# ============================================================
# FIND CASES
# ============================================================

case_dirs = sorted(
    [
        path
        for path in PARTIAL_DIR.iterdir()
        if path.is_dir()
    ]
)


print(
    "Cases:",
    len(case_dirs)
)


# ============================================================
# PRECOMPUTE EVIDENCE
# ============================================================

case_evidence = {}


for case_dir in case_dirs:

    case = case_dir.name

    case_evidence[
        case
    ] = build_service_evidence(
        feature_df,
        case
    )


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# OBSERVATION LEVELS
# ============================================================

for observation_level in (
    OBSERVATION_LEVELS
):

    print()
    print("-" * 70)

    print(
        f"Observation level: "
        f"{observation_level}%"
    )

    print("-" * 70)


    level_results = []


    # ========================================================
    # CURRENT INCIDENT
    # ========================================================

    for case_dir in case_dirs:

        case = case_dir.name

        root_service, fault_type = (
            parse_case(case)
        )


        # ----------------------------------------------------
        # Load partial current graph
        # ----------------------------------------------------

        current_graph = load_graph(
            case_dir
            /
            f"partial_{observation_level}.gpickle"
        )


        anomaly_evidence = (
            case_evidence[case]
        )


        candidate_totals = {}


        # ====================================================
        # HISTORICAL INCIDENTS
        # ====================================================

        for other_case_dir in case_dirs:

            other_case = (
                other_case_dir.name
            )


            # ------------------------------------------------
            # Leave current case out
            # ------------------------------------------------

            if other_case == case:

                continue


            other_root, other_fault = (
                parse_case(other_case)
            )


            # ------------------------------------------------
            # Leave-one-fault-family-out
            # ------------------------------------------------

            if (
                other_root == root_service
                and
                other_fault == fault_type
            ):

                continue


            # ------------------------------------------------
            # Complete historical graph
            # ------------------------------------------------

            historical_graph = load_graph(
                other_case_dir
                /
                "partial_100.gpickle"
            )


            # =================================================
            # STRUCTURAL MATCH
            # =================================================

            structural_score = (
                graph_match_score(
                    current_graph,
                    historical_graph
                )
            )


            # =================================================
            # FIND HIDDEN UPSTREAM CANDIDATES
            # =================================================

            candidates = (
                find_upstream_candidates(
                    current_graph,
                    historical_graph,
                    MAX_HOPS
                )
            )


            # =================================================
            # SCORE EACH CANDIDATE
            # =================================================

            for candidate, candidate_distance in (
                candidates.items()
            ):

                # ------------------------------------------------
                # Find observed descendants that this candidate
                # historically leads to.
                # ------------------------------------------------

                downstream_services = (
                    candidate_downstream_evidence(
                        candidate,
                        current_graph,
                        historical_graph,
                        anomaly_evidence,
                        MAX_HOPS
                    )
                )


                # ------------------------------------------------
                # Candidate-specific anomaly evidence
                # ------------------------------------------------

                evidence_score = (
                    calculate_candidate_evidence(
                        downstream_services
                    )
                )


                # ------------------------------------------------
                # Final candidate score
                # ------------------------------------------------

                score = (
                    calculate_candidate_score(
                        structural_score,
                        candidate_distance,
                        evidence_score
                    )
                )


                # ------------------------------------------------
                # Aggregate evidence across historical incidents
                # ------------------------------------------------

                candidate_totals[
                    candidate
                ] = (
                    candidate_totals.get(
                        candidate,
                        0.0
                    )
                    +
                    score
                )


        # ====================================================
        # RANK SERVICES
        # ====================================================

        ranked_candidates = sorted(
            candidate_totals.items(),

            key=lambda x: (
                -x[1],
                x[0]
            )
        )


        ranked_services = [
            candidate
            for candidate, score
            in ranked_candidates
        ]


        # ====================================================
        # ROOT RANK
        # ====================================================

        rank = None


        if root_service in ranked_services:

            rank = (
                ranked_services.index(
                    root_service
                )
                +
                1
            )


        # ====================================================
        # METRICS
        # ====================================================

        top1 = (
            rank == 1
            if rank is not None
            else False
        )

        top3 = (
            rank <= 3
            if rank is not None
            else False
        )

        top5 = (
            rank <= 5
            if rank is not None
            else False
        )

        mrr = (
            1.0 / rank
            if rank is not None
            else 0.0
        )


        # ====================================================
        # STORE
        # ====================================================

        level_results.append(
            {
                "case":
                    case,

                "root_service":
                    root_service,

                "fault_type":
                    fault_type,

                "observation_level":
                    observation_level,

                "rank":
                    rank,

                "top1":
                    int(top1),

                "top3":
                    int(top3),

                "top5":
                    int(top5),

                "mrr":
                    mrr,

                "num_observed_nodes":
                    current_graph.number_of_nodes(),

                "num_observed_edges":
                    current_graph.number_of_edges(),

                "num_candidates":
                    len(
                        candidate_totals
                    ),
            }
        )


    # ========================================================
    # LEVEL SUMMARY
    # ========================================================

    results.extend(
        level_results
    )


    level_df = pd.DataFrame(
        level_results
    )


    print(
        f"Top-1: "
        f"{level_df['top1'].mean():.4f}"
    )

    print(
        f"Top-3: "
        f"{level_df['top3'].mean():.4f}"
    )

    print(
        f"Top-5: "
        f"{level_df['top5'].mean():.4f}"
    )

    print(
        f"MRR: "
        f"{level_df['mrr'].mean():.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)


output = (
    DATA_DIR
    /
    "candidate_evidence_rca_results.parquet"
)


results_df.to_parquet(
    output,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)


summary = (
    results_df
    .groupby(
        "observation_level"
    )
    .agg(
        cases=(
            "case",
            "count"
        ),

        top1=(
            "top1",
            "mean"
        ),

        top3=(
            "top3",
            "mean"
        ),

        top5=(
            "top5",
            "mean"
        ),

        mrr=(
            "mrr",
            "mean"
        ),

        avg_nodes=(
            "num_observed_nodes",
            "mean"
        ),

        avg_edges=(
            "num_observed_edges",
            "mean"
        ),

        avg_candidates=(
            "num_candidates",
            "mean"
        ),
    )
    .reset_index()
)


print(
    summary.to_string(
        index=False
    )
)


print()
print("Saved:")
print(output)