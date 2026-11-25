
from pathlib import Path
from collections import deque
import pickle

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
DIRECTIONS = ["predecessors", "successors", "undirected"]

MAX_HOPS = 3

OUTPUT_PARQUET = (
    DATA_DIR / "graph_direction_comparison_results.parquet"
)
OUTPUT_CSV = (
    DATA_DIR / "graph_direction_comparison_results.csv"
)
SUMMARY_CSV = (
    DATA_DIR / "graph_direction_comparison_summary.csv"
)


# ============================================================
# LOAD AND PARSE GRAPHS
# ============================================================

def load_graph(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing graph: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def parse_case(case):
    parts = case.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected case name: {case}")

    return parts[1], parts[2]


# ============================================================
# STRUCTURAL SIMILARITY
# ============================================================

def graph_match_score(current_graph, historical_graph):
    current_nodes = set(current_graph.nodes())
    historical_nodes = set(historical_graph.nodes())

    current_edges = set(current_graph.edges())
    historical_edges = set(historical_graph.edges())

    if current_nodes:
        node_score = (
            len(current_nodes & historical_nodes)
            / len(current_nodes)
        )
    else:
        node_score = 0.0

    if current_edges:
        edge_score = (
            len(current_edges & historical_edges)
            / len(current_edges)
        )
    else:
        edge_score = 0.0

    if not current_edges:
        structural_score = node_score
    else:
        structural_score = (node_score + edge_score) / 2.0

    return structural_score, node_score, edge_score


# ============================================================
# NEIGHBOURS IN THE CHOSEN DIRECTION
# ============================================================

def get_neighbors(graph, node, direction):
    if direction == "predecessors":
        return list(graph.predecessors(node))

    if direction == "successors":
        return list(graph.successors(node))

    if direction == "undirected":
        return list(
            set(graph.predecessors(node))
            | set(graph.successors(node))
        )

    raise ValueError(f"Unsupported direction: {direction}")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def find_candidates(
    current_graph,
    historical_graph,
    direction,
    max_hops=3,
):
    """
    Search outward from observed services.

    Only services absent from the current graph are candidates.
    Record the shortest distance found for each candidate.
    """

    observed_nodes = set(current_graph.nodes())
    candidates = {}

    queue = deque(
        (node, 0)
        for node in observed_nodes
        if node in historical_graph
    )

    shortest_distance = {
        node: 0 for node, _ in queue
    }

    while queue:
        node, distance = queue.popleft()

        if distance >= max_hops:
            continue

        neighbors = get_neighbors(
            historical_graph,
            node,
            direction,
        )

        for neighbor in neighbors:
            new_distance = distance + 1

            if neighbor not in observed_nodes:
                candidates[neighbor] = min(
                    candidates.get(neighbor, float("inf")),
                    new_distance,
                )

            # Do not repeatedly explore a node when a
            # shorter path to it has already been found.
            if (
                neighbor in shortest_distance
                and shortest_distance[neighbor] <= new_distance
            ):
                continue

            shortest_distance[neighbor] = new_distance
            queue.append((neighbor, new_distance))

    return candidates


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidates(candidates, structural_score):
    """
    Original score:
        structural similarity / (1 + distance)

    A zero-similarity historical graph contributes no evidence.
    We exclude its candidates rather than ranking zero-score
    ties alphabetically.
    """

    if structural_score <= 0:
        return {}

    scores = {}

    for candidate, distance in candidates.items():
        scores[candidate] = (
            structural_score / (1.0 + distance)
        )

    return scores


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 76)
    print("CONTROLLED GRAPH-DIRECTION RCA COMPARISON")
    print("=" * 76)

    print("Project root:", ROOT_DIR)
    print("Maximum hops:", MAX_HOPS)
    print("Directions:", DIRECTIONS)

    if not PARTIAL_DIR.exists():
        raise FileNotFoundError(
            f"Partial graph directory not found: {PARTIAL_DIR}"
        )

    case_dirs = sorted(
        path for path in PARTIAL_DIR.iterdir()
        if path.is_dir()
    )

    if not case_dirs:
        raise RuntimeError("No incident folders found.")

    print("Cases found:", len(case_dirs))

    # Load complete historical graphs once.
    historical_graphs = {}

    for case_dir in case_dirs:
        case = case_dir.name

        graph_path = (
            DATA_DIR
            / case
            / "historical_incident_graph.gpickle"
        )

        historical_graphs[case] = load_graph(graph_path)

    print(
        "Complete historical graphs loaded:",
        len(historical_graphs),
    )

    all_results = []

    # ========================================================
    # OBSERVATION LEVELS
    # ========================================================

    for level in OBSERVATION_LEVELS:

        print("\n" + "=" * 76)
        print(f"OBSERVATION LEVEL: {level}%")
        print("=" * 76)

        for direction in DIRECTIONS:

            print(f"\nDirection: {direction}")

            direction_results = []

            # ------------------------------------------------
            # Evaluate each current incident
            # ------------------------------------------------

            for case_dir in case_dirs:

                case = case_dir.name
                root_service, fault_type = parse_case(case)

                partial_path = (
                    case_dir / f"partial_{level}.gpickle"
                )

                current_graph = load_graph(partial_path)

                candidate_totals = {}

                # --------------------------------------------
                # Search the historical knowledge bank
                # --------------------------------------------

                for other_case_dir in case_dirs:

                    other_case = other_case_dir.name

                    if other_case == case:
                        continue

                    other_root, other_fault = parse_case(
                        other_case
                    )

                    # Keep the same family-exclusion protocol
                    # as the existing graph-matching experiment.
                    if (
                        other_root == root_service
                        and other_fault == fault_type
                    ):
                        continue

                    historical_graph = historical_graphs[
                        other_case
                    ]

                    # Structural evidence
                    structural_score, _, _ = graph_match_score(
                        current_graph,
                        historical_graph,
                    )

                    # Candidate generation using the chosen
                    # traversal direction
                    candidates = find_candidates(
                        current_graph,
                        historical_graph,
                        direction=direction,
                        max_hops=MAX_HOPS,
                    )

                    # Score the candidates
                    candidate_scores = score_candidates(
                        candidates,
                        structural_score,
                    )

                    # Aggregate evidence across historical cases
                    for candidate, score in (
                        candidate_scores.items()
                    ):
                        candidate_totals[candidate] = (
                            candidate_totals.get(candidate, 0.0)
                            + score
                        )

                # --------------------------------------------
                # Rank candidates
                # --------------------------------------------

                ranked_candidates = sorted(
                    candidate_totals.items(),
                    key=lambda item: (-item[1], item[0]),
                )

                ranked_services = [
                    service
                    for service, _ in ranked_candidates
                ]

                # Ground truth is used only for evaluation.
                root_is_candidate = (
                    root_service in ranked_services
                )

                if root_is_candidate:
                    rank = (
                        ranked_services.index(root_service) + 1
                    )
                else:
                    rank = None

                top1 = int(rank == 1) if rank is not None else 0
                top3 = int(rank <= 3) if rank is not None else 0
                top5 = int(rank <= 5) if rank is not None else 0
                mrr = 1.0 / rank if rank is not None else 0.0

                root_score = candidate_totals.get(
                    root_service, 0.0
                )

                direction_results.append({
                    "case": case,
                    "root_service": root_service,
                    "fault_type": fault_type,
                    "observation_level": level,
                    "direction": direction,

                    "root_is_candidate": int(root_is_candidate),
                    "rank": rank,
                    "root_score": root_score,

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

            # ------------------------------------------------
            # Print metrics for this direction and level
            # ------------------------------------------------

            df = pd.DataFrame(direction_results)
            all_results.extend(direction_results)

            n = len(df)
            coverage = df["root_is_candidate"].mean()

            candidates_only = df[
                df["root_is_candidate"] == 1
            ]

            conditional_top1 = (
                candidates_only["top1"].mean()
                if not candidates_only.empty
                else 0.0
            )

            no_candidate_rate = (
                df["num_candidate_services"] == 0
            ).mean()

            print(f"Root candidate coverage: {coverage:.4f}")
            print(f"No-candidate rate:       {no_candidate_rate:.4f}")
            print(f"Overall Top-1:            {df['top1'].mean():.4f}")
            print(f"Overall Top-3:            {df['top3'].mean():.4f}")
            print(f"Overall Top-5:            {df['top5'].mean():.4f}")
            print(f"Overall MRR:              {df['mrr'].mean():.4f}")
            print(f"Conditional Top-1:        {conditional_top1:.4f}")
            print(
                "Average candidates:      "
                f"{df['num_candidate_services'].mean():.2f}"
            )

    # ========================================================
    # SAVE CASE-LEVEL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary = (
        results_df
        .groupby(["observation_level", "direction"])
        .agg(
            cases=("case", "count"),
            root_candidate_coverage=(
                "root_is_candidate", "mean"
            ),
            overall_top1=("top1", "mean"),
            overall_top3=("top3", "mean"),
            overall_top5=("top5", "mean"),
            mrr=("mrr", "mean"),
            average_candidates=(
                "num_candidate_services", "mean"
            ),
            no_candidate_rate=(
                "num_candidate_services",
                lambda values: (values == 0).mean(),
            ),
        )
        .reset_index()
        .sort_values(
            ["observation_level", "direction"],
            ascending=[True, True],
        )
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 76)
    print("FINAL COMPARISON")
    print("=" * 76)
    print(summary.to_string(index=False))

    print("\nSaved case-level Parquet:", OUTPUT_PARQUET)
    print("Saved case-level CSV:    ", OUTPUT_CSV)
    print("Saved summary CSV:       ", SUMMARY_CSV)


if __name__ == "__main__":
    main()
