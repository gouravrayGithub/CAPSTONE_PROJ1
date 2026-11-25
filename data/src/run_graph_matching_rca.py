from pathlib import Path
import pickle
import networkx as nx
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [100, 80, 60, 40, 20]

MAX_HOPS = 3


# ============================================================
# CASE PARSER
# ============================================================

def parse_case(case):

    parts = case.split("_")

    root_service = parts[1]
    fault_type = parts[2]

    return root_service, fault_type


# ============================================================
# GRAPH LOADER
# ============================================================

def load_graph(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# STRUCTURAL MATCHING
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
    # Node similarity
    # --------------------------------------------------------

    if len(current_nodes) == 0:

        node_score = 0.0

    else:

        matched_nodes = (
            current_nodes
            .intersection(
                historical_nodes
            )
        )

        node_score = (
            len(matched_nodes)
            /
            len(current_nodes)
        )

    # --------------------------------------------------------
    # Edge similarity
    # --------------------------------------------------------

    if len(current_edges) == 0:

        edge_score = 0.0

    else:

        matched_edges = (
            current_edges
            .intersection(
                historical_edges
            )
        )

        edge_score = (
            len(matched_edges)
            /
            len(current_edges)
        )

    # --------------------------------------------------------
    # Combined structural score
    # --------------------------------------------------------

    if len(current_edges) == 0:

        structural_score = node_score

    else:

        structural_score = (
            node_score
            +
            edge_score
        ) / 2.0

    return (
        structural_score,
        node_score,
        edge_score
    )


# ============================================================
# MULTI-HOP UPSTREAM CANDIDATES
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

    # --------------------------------------------------------
    # Start backward BFS from every observed node.
    # --------------------------------------------------------

    queue = []

    for observed_node in current_nodes:

        if observed_node in historical_graph:

            queue.append(
                (
                    observed_node,
                    0
                )
            )

    # --------------------------------------------------------
    # BFS traversal
    #
    # distance = number of edges travelled upstream
    #
    # Example:
    #
    # A -> B -> C
    #
    # If C is observed:
    #
    # B = 1 hop
    # A = 2 hops
    # --------------------------------------------------------

    visited = set()

    while queue:

        node, distance = queue.pop(0)

        state = (
            node,
            distance
        )

        if state in visited:
            continue

        visited.add(state)

        # Stop after MAX_HOPS
        if distance >= max_hops:
            continue

        if node not in historical_graph:
            continue

        # ----------------------------------------------------
        # Traverse backwards through historical dependencies
        # ----------------------------------------------------

        for predecessor in (
            historical_graph.predecessors(node)
        ):

            new_distance = distance + 1

            # ------------------------------------------------
            # Only unobserved services can become RCA
            # candidates.
            # ------------------------------------------------

            if predecessor not in current_nodes:

                if predecessor not in candidates:

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

            # ------------------------------------------------
            # Continue searching upstream.
            # ------------------------------------------------

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

def score_candidates(
    candidates,
    structural_score
):

    scores = {}

    for candidate, distance in (
        candidates.items()
    ):

        # ----------------------------------------------------
        # Distance decay
        #
        # 1 hop -> 1/2
        # 2 hops -> 1/3
        # 3 hops -> 1/4
        # ----------------------------------------------------

        distance_factor = (
            1.0
            /
            (1.0 + distance)
        )

        score = (
            structural_score
            *
            distance_factor
        )

        scores[candidate] = score

    return scores


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MULTI-HOP GRAPH MATCHING RCA")
print("=" * 70)

print(
    f"Maximum upstream hops: {MAX_HOPS}"
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
        # Load partial current graph
        # ----------------------------------------------------

        partial_path = (
            case_dir
            /
            f"partial_{observation_level}.gpickle"
        )

        current_graph = load_graph(
            partial_path
        )


        # ----------------------------------------------------
        # Historical evidence
        # ----------------------------------------------------

        historical_cases = []


        # ====================================================
        # SEARCH HISTORICAL INCIDENTS
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
            #
            # Example:
            #
            # Current:
            # checkoutservice_cpu_1
            #
            # Exclude:
            # checkoutservice_cpu_2
            # checkoutservice_cpu_3
            #
            # But allow:
            # checkoutservice_disk_1
            # checkoutservice_delay_1
            # etc.
            # ------------------------------------------------

            if (
                other_root == root_service
                and
                other_fault == fault_type
            ):

                continue


            # ------------------------------------------------
            # Historical graph
            #
            # IMPORTANT:
            #
            # Historical graph is complete.
            # We use partial_100 because the historical
            # knowledge bank should contain complete evidence.
            # ------------------------------------------------

            historical_path = (
                other_case_dir
                /
                "partial_100.gpickle"
            )

            historical_graph = load_graph(
                historical_path
            )


            # =================================================
            # STRUCTURAL MATCH
            # =================================================

            (
                structural_score,
                node_score,
                edge_score
            ) = graph_match_score(
                current_graph,
                historical_graph
            )


            # =================================================
            # MULTI-HOP UPSTREAM SEARCH
            # =================================================

            candidates = (
                find_upstream_candidates(
                    current_graph,
                    historical_graph,
                    max_hops=MAX_HOPS
                )
            )


            # =================================================
            # SCORE CANDIDATES
            # =================================================

            candidate_scores = (
                score_candidates(
                    candidates,
                    structural_score
                )
            )


            # =================================================
            # STORE HISTORICAL EVIDENCE
            # =================================================

            for candidate, score in (
                candidate_scores.items()
            ):

                historical_cases.append(
                    {
                        "historical_case":
                            other_case,

                        "candidate":
                            candidate,

                        "score":
                            score,

                        "structural_score":
                            structural_score,

                        "node_score":
                            node_score,

                        "edge_score":
                            edge_score,

                        "distance":
                            candidates[candidate],
                    }
                )


        # ====================================================
        # AGGREGATE CANDIDATE EVIDENCE
        # ====================================================

        candidate_totals = {}


        for item in historical_cases:

            candidate = (
                item["candidate"]
            )

            score = (
                item["score"]
            )

            candidate_totals[candidate] = (
                candidate_totals.get(
                    candidate,
                    0.0
                )
                +
                score
            )


        # ====================================================
        # RANK CANDIDATES
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
        # ROOT-CAUSE RANK
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
        # SAVE CASE RESULT
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

                "num_candidate_services":
                    len(
                        candidate_totals
                    ),
            }
        )


    # ========================================================
    # STORE LEVEL RESULTS
    # ========================================================

    results.extend(
        level_results
    )


    level_df = pd.DataFrame(
        level_results
    )


    # ========================================================
    # PRINT LEVEL RESULTS
    # ========================================================

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

    print(
        f"Average observed nodes: "
        f"{level_df['num_observed_nodes'].mean():.2f}"
    )

    print(
        f"Average observed edges: "
        f"{level_df['num_observed_edges'].mean():.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


output = (
    DATA_DIR
    /
    "graph_matching_rca_results.parquet"
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
            "num_candidate_services",
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