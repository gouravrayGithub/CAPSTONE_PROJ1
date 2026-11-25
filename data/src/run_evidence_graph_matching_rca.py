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

    return (
        math.copysign(
            math.log1p(abs(float(x))),
            float(x)
        )
    )


# ============================================================
# BUILD SERVICE EVIDENCE
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

            values.append(
                abs(value)
            )

        # ----------------------------------------------------
        # Equal contribution from the four evidence sources
        # ----------------------------------------------------

        evidence[
            row["service"]
        ] = sum(values) / len(values)

    return evidence


# ============================================================
# STRUCTURAL MATCH
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
                & historical_nodes
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
                & historical_edges
            )
            /
            len(current_edges)
        )

    else:

        edge_score = 0.0


    # --------------------------------------------------------
    # Structural score
    # --------------------------------------------------------

    if current_edges:

        structural_score = (
            node_score
            +
            edge_score
        ) / 2.0

    else:

        structural_score = node_score


    return (
        structural_score,
        node_score,
        edge_score
    )


# ============================================================
# NEIGHBORHOOD MATCH
# ============================================================

def neighborhood_score(
    current_graph,
    historical_graph
):

    current_nodes = set(
        current_graph.nodes()
    )

    matched_nodes = (
        current_nodes
        &
        set(historical_graph.nodes())
    )

    if not matched_nodes:

        return 0.0


    scores = []


    for node in matched_nodes:

        # ----------------------------------------------------
        # Current neighborhood
        # ----------------------------------------------------

        current_predecessors = set(
            current_graph.predecessors(node)
        )

        current_successors = set(
            current_graph.successors(node)
        )


        # ----------------------------------------------------
        # Historical neighborhood
        # ----------------------------------------------------

        historical_predecessors = set(
            historical_graph.predecessors(node)
        )

        historical_successors = set(
            historical_graph.successors(node)
        )


        # ----------------------------------------------------
        # Predecessor similarity
        # ----------------------------------------------------

        if current_predecessors:

            pred_score = (
                len(
                    current_predecessors
                    &
                    historical_predecessors
                )
                /
                len(current_predecessors)
            )

        else:

            pred_score = 1.0


        # ----------------------------------------------------
        # Successor similarity
        # ----------------------------------------------------

        if current_successors:

            succ_score = (
                len(
                    current_successors
                    &
                    historical_successors
                )
                /
                len(current_successors)
            )

        else:

            succ_score = 1.0


        scores.append(
            (
                pred_score
                +
                succ_score
            ) / 2.0
        )


    return sum(scores) / len(scores)


# ============================================================
# MULTI-HOP UPSTREAM SEARCH
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

    for node in current_nodes:

        if node in historical_graph:

            queue.append(
                (
                    node,
                    0
                )
            )

    visited = set()


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
            # Candidate must not already be observed
            # ------------------------------------------------

            if (
                predecessor
                not in current_nodes
            ):

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
# CANDIDATE SCORE
# ============================================================

def candidate_score(
    structural_score,
    neighborhood,
    candidate_distance,
    observed_evidence
):

    # --------------------------------------------------------
    # Distance decay
    # --------------------------------------------------------

    distance_factor = (
        1.0
        /
        (1.0 + candidate_distance)
    )


    # --------------------------------------------------------
    # Evidence aggregation
    #
    # observed_evidence represents how strongly the currently
    # observed incident behaves abnormally around the matched
    # structure.
    # --------------------------------------------------------

    evidence_factor = (
        1.0
        +
        observed_evidence
    )


    score = (
        structural_score
        *
        (0.5 + 0.5 * neighborhood)
        *
        distance_factor
        *
        evidence_factor
    )


    return score


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("EVIDENCE-AWARE GRAPH MATCHING RCA")
print("=" * 70)

print(
    f"Maximum upstream hops: {MAX_HOPS}"
)


# ============================================================
# LOAD FEATURES
# ============================================================

features = pd.read_parquet(
    FEATURE_FILE
)


# ============================================================
# BUILD CASE LIST
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
# PRECOMPUTE SERVICE EVIDENCE
# ============================================================

case_evidence = {}


for case_dir in case_dirs:

    case = case_dir.name

    case_evidence[
        case
    ] = build_service_evidence(
        features,
        case
    )


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# EVALUATION
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
        # Load current partial graph
        # ----------------------------------------------------

        current_graph = load_graph(
            case_dir
            /
            f"partial_{observation_level}.gpickle"
        )


        current_evidence = (
            case_evidence[case]
        )


        # ----------------------------------------------------
        # Candidate aggregation
        # ----------------------------------------------------

        candidate_totals = {}


        # ====================================================
        # HISTORICAL INCIDENT SEARCH
        # ====================================================

        for other_case_dir in case_dirs:

            other_case = (
                other_case_dir.name
            )


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


            historical_graph = load_graph(
                other_case_dir
                /
                "partial_100.gpickle"
            )


            # =================================================
            # STRUCTURAL MATCH
            # =================================================

            (
                structural,
                node_score,
                edge_score
            ) = graph_match_score(
                current_graph,
                historical_graph
            )


            # =================================================
            # NEIGHBORHOOD MATCH
            # =================================================

            neighborhood = (
                neighborhood_score(
                    current_graph,
                    historical_graph
                )
            )


            # =================================================
            # UPSTREAM CANDIDATES
            # =================================================

            candidates = (
                find_upstream_candidates(
                    current_graph,
                    historical_graph,
                    MAX_HOPS
                )
            )


            # =================================================
            # CURRENT OBSERVED EVIDENCE
            # =================================================

            observed_services = (
                set(current_graph.nodes())
                &
                set(current_evidence.keys())
            )


            if observed_services:

                evidence_values = [
                    current_evidence[
                        service
                    ]
                    for service
                    in observed_services
                ]

                observed_evidence = (
                    sum(evidence_values)
                    /
                    len(evidence_values)
                )

            else:

                observed_evidence = 0.0


            # =================================================
            # SCORE CANDIDATES
            # =================================================

            for candidate, distance in (
                candidates.items()
            ):

                score = candidate_score(
                    structural,
                    neighborhood,
                    distance,
                    observed_evidence
                )


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
        # RANK
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
        # SAVE
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
                    len(candidate_totals),
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
    "evidence_graph_matching_rca_results.parquet"
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