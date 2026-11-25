
from pathlib import Path
import pickle

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [100, 80, 60, 40, 20]
MAX_HOPS = 3

# Save separately so the original experiment remains unchanged.
OUTPUT_FILE = DATA_DIR / "graph_matching_rca_v2_results.parquet"


# ============================================================
# CASE PARSER
# ============================================================

def parse_case(case):
    """
    Example:
    re2ob_currencyservice_cpu_1

    root_service = currencyservice
    fault_type = cpu
    """

    parts = case.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected case name: {case}")

    root_service = parts[1]
    fault_type = parts[2]

    return root_service, fault_type


# ============================================================
# GRAPH LOADER
# ============================================================

def load_graph(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


# ============================================================
# STRUCTURAL MATCHING
# ============================================================

def graph_match_score(current_graph, historical_graph):
    """
    Measures how much of the current graph's nodes and edges
    are also present in a historical graph.

    The denominator is the current graph's node/edge count.
    """

    current_nodes = set(current_graph.nodes())
    historical_nodes = set(historical_graph.nodes())

    current_edges = set(current_graph.edges())
    historical_edges = set(historical_graph.edges())

    # Node similarity
    if current_nodes:
        node_score = (
            len(current_nodes & historical_nodes)
            / len(current_nodes)
        )
    else:
        node_score = 0.0

    # Edge similarity
    if current_edges:
        edge_score = (
            len(current_edges & historical_edges)
            / len(current_edges)
        )
    else:
        edge_score = 0.0

    # Combined structural score
    if not current_edges:
        structural_score = node_score
    else:
        structural_score = (
            node_score + edge_score
        ) / 2.0

    return structural_score, node_score, edge_score


# ============================================================
# MULTI-HOP UPSTREAM CANDIDATE GENERATION
# ============================================================

def find_upstream_candidates(
    current_graph,
    historical_graph,
    max_hops=3,
):
    """
    Start from observed services and traverse backwards
    through the historical graph.

    Only unobserved upstream services become candidates.

    Assumes graph edges are directed from upstream to
    downstream, e.g. A -> B -> C.
    """

    current_nodes = set(current_graph.nodes())
    candidates = {}

    # Start from every observed service that exists
    # in the historical graph.
    queue = [
        (node, 0)
        for node in current_nodes
        if node in historical_graph
    ]

    # Track the shortest distance reached for each node.
    shortest_distance = {}

    while queue:
        node, distance = queue.pop(0)

        if distance >= max_hops:
            continue

        if node not in historical_graph:
            continue

        for predecessor in historical_graph.predecessors(node):
            new_distance = distance + 1

            # Record only the shortest path to this node.
            if (
                predecessor in shortest_distance
                and shortest_distance[predecessor] <= new_distance
            ):
                continue

            shortest_distance[predecessor] = new_distance

            # Only services absent from the current graph
            # can be proposed as root-cause candidates.
            if predecessor not in current_nodes:
                candidates[predecessor] = min(
                    candidates.get(predecessor, float("inf")),
                    new_distance,
                )

            # Continue the backward search.
            queue.append((predecessor, new_distance))

    return candidates


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidates(candidates, structural_score):
    """
    Preserve the original scoring rule:

    score = structural similarity / (1 + upstream distance)
    """

    scores = {}

    for candidate, distance in candidates.items():
        distance_factor = 1.0 / (1.0 + distance)

        scores[candidate] = (
            structural_score * distance_factor
        )

    return scores


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 72)
    print("GRAPH MATCHING RCA - V2")
    print("=" * 72)

    print("Maximum upstream hops:", MAX_HOPS)
    print("Partial graph directory:", PARTIAL_DIR)

    if not PARTIAL_DIR.exists():
        raise FileNotFoundError(
            f"Partial graph directory not found: {PARTIAL_DIR}"
        )

    case_dirs = sorted(
        path
        for path in PARTIAL_DIR.iterdir()
        if path.is_dir()
    )

    if not case_dirs:
        raise RuntimeError(
            f"No incident folders found in {PARTIAL_DIR}"
        )

    print("Cases found:", len(case_dirs))

    # Load complete historical graphs once.
    # These must be the original historical graphs,
    # not the root-hidden partial_100 graphs.
    historical_graphs = {}

    for case_dir in case_dirs:
        case = case_dir.name

        historical_path = (
            DATA_DIR
            / case
            / "historical_incident_graph.gpickle"
        )

        historical_graphs[case] = load_graph(
            historical_path
        )

    print(
        "Complete historical graphs loaded:",
        len(historical_graphs),
    )

    all_results = []

    # ========================================================
    # EVALUATE EACH OBSERVATION LEVEL
    # ========================================================

    for observation_level in OBSERVATION_LEVELS:

        print()
        print("-" * 72)
        print(f"Observation level: {observation_level}%")
        print("-" * 72)

        level_results = []

        for case_dir in case_dirs:

            case = case_dir.name
            root_service, fault_type = parse_case(case)

            # ------------------------------------------------
            # Load current partial graph
            # ------------------------------------------------

            partial_path = (
                case_dir
                / f"partial_{observation_level}.gpickle"
            )

            current_graph = load_graph(partial_path)

            # ------------------------------------------------
            # Collect evidence from historical incidents
            # ------------------------------------------------

            candidate_totals = {}

            for other_case_dir in case_dirs:

                other_case = other_case_dir.name

                # Exclude the current incident itself.
                if other_case == case:
                    continue

                other_root, other_fault = parse_case(
                    other_case
                )

                # Leave the same root + fault family out.
                # This is the original experiment's protocol.
                if (
                    other_root == root_service
                    and other_fault == fault_type
                ):
                    continue

                # IMPORTANT FIX:
                # Use the complete historical graph.
                historical_graph = historical_graphs[other_case]

                # Structural similarity
                (
                    structural_score,
                    node_score,
                    edge_score,
                ) = graph_match_score(
                    current_graph,
                    historical_graph,
                )

                # Upstream candidate generation
                candidates = find_upstream_candidates(
                    current_graph,
                    historical_graph,
                    max_hops=MAX_HOPS,
                )

                # Candidate scores
                candidate_scores = score_candidates(
                    candidates,
                    structural_score,
                )

                # Aggregate scores over historical incidents
                for candidate, score in candidate_scores.items():

                    candidate_totals[candidate] = (
                        candidate_totals.get(candidate, 0.0)
                        + score
                    )

            # ------------------------------------------------
            # Rank all generated candidates
            # ------------------------------------------------

            ranked_candidates = sorted(
                candidate_totals.items(),
                key=lambda item: (-item[1], item[0]),
            )

            ranked_services = [
                service
                for service, _ in ranked_candidates
            ]

            # ------------------------------------------------
            # Evaluation only: check the ground-truth root
            # ------------------------------------------------

            root_is_candidate = (
                root_service in ranked_services
            )

            if root_is_candidate:
                rank = ranked_services.index(root_service) + 1
            else:
                rank = None

            top1 = int(rank == 1) if rank is not None else 0
            top3 = int(rank <= 3) if rank is not None else 0
            top5 = int(rank <= 5) if rank is not None else 0

            mrr = 1.0 / rank if rank is not None else 0.0

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            level_results.append({
                "case": case,
                "root_service": root_service,
                "fault_type": fault_type,
                "observation_level": observation_level,

                "rank": rank,
                "root_is_candidate": int(root_is_candidate),

                "top1": top1,
                "top3": top3,
                "top5": top5,
                "mrr": mrr,

                "num_observed_nodes":
                    current_graph.number_of_nodes(),

                "num_observed_edges":
                    current_graph.number_of_edges(),

                "num_candidate_services":
                    len(candidate_totals),
            })

        # ====================================================
        # SUMMARIZE THIS OBSERVATION LEVEL
        # ====================================================

        level_df = pd.DataFrame(level_results)
        all_results.extend(level_results)

        total_cases = len(level_df)
        candidate_cases = int(
            level_df["root_is_candidate"].sum()
        )

        candidate_coverage = (
            candidate_cases / total_cases
            if total_cases else 0.0
        )

        # Accuracy only among incidents where the true root
        # was actually generated as a candidate.
        candidate_df = level_df[
            level_df["root_is_candidate"] == 1
        ]

        conditional_top1 = (
            candidate_df["top1"].mean()
            if not candidate_df.empty else 0.0
        )

        print(f"Cases: {total_cases}")
        print(
            f"Root candidate coverage: "
            f"{candidate_coverage:.4f}"
        )
        print(
            f"No-candidate rate: "
            f"{(level_df['num_candidate_services'] == 0).mean():.4f}"
        )
        print(
            f"Overall Top-1: {level_df['top1'].mean():.4f}"
        )
        print(
            f"Overall Top-3: {level_df['top3'].mean():.4f}"
        )
        print(
            f"Overall Top-5: {level_df['top5'].mean():.4f}"
        )
        print(
            f"Overall MRR: {level_df['mrr'].mean():.4f}"
        )
        print(
            f"Conditional Top-1: {conditional_top1:.4f}"
        )
        print(
            "Average candidate count: "
            f"{level_df['num_candidate_services'].mean():.2f}"
        )
        print(
            "Average observed nodes: "
            f"{level_df['num_observed_nodes'].mean():.2f}"
        )
        print(
            "Average observed edges: "
            f"{level_df['num_observed_edges'].mean():.2f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # Also save a CSV for easy inspection.
    csv_output = OUTPUT_FILE.with_suffix(".csv")
    results_df.to_csv(csv_output, index=False)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby("observation_level")
        .agg(
            cases=("case", "count"),
            root_candidate_coverage=(
                "root_is_candidate", "mean"
            ),
            overall_top1=("top1", "mean"),
            overall_top3=("top3", "mean"),
            overall_top5=("top5", "mean"),
            mrr=("mrr", "mean"),
            avg_candidates=("num_candidate_services", "mean"),
            no_candidate_rate=(
                "num_candidate_services",
                lambda values: (values == 0).mean(),
            ),
        )
        .reset_index()
        .sort_values("observation_level", ascending=False)
    )

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print()
    print("Parquet saved to:", OUTPUT_FILE)
    print("CSV saved to:", csv_output)


if __name__ == "__main__":
    main()
