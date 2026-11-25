
from pathlib import Path
from collections import defaultdict
import json
import pickle

import networkx as nx
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("experiments/successor_v1/graphs")
RESULTS_DIR = Path("experiments/successor_v1/results")

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
MAX_HOPS = 3


# ============================================================
# CASE INFORMATION
# ============================================================

def parse_case_name(case_name):
    """
    Example:
    re2ob_currencyservice_cpu_1

    Root service: currencyservice
    Fault type: cpu
    """
    parts = case_name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected case name: {case_name}")

    return parts[1], parts[2]


def load_graph(path):
    with path.open("rb") as file:
        return pickle.load(file)


# ============================================================
# GRAPH STRUCTURAL SIMILARITY
# ============================================================

def graph_match_score(current_graph, historical_graph):
    """
    Compare the observed graph with a historical graph.

    Node score:
        Fraction of current observed nodes found historically.

    Edge score:
        Fraction of current observed directed edges found
        in the historical graph.

    Final score:
        Average of node and edge scores.
        If the current graph has no edges, use node score.
    """

    current_nodes = set(current_graph.nodes())
    historical_nodes = set(historical_graph.nodes())

    if not current_nodes:
        return 0.0

    node_score = (
        len(current_nodes & historical_nodes)
        / len(current_nodes)
    )

    current_edges = set(current_graph.edges())
    historical_edges = set(historical_graph.edges())

    if not current_edges:
        return node_score

    edge_score = (
        len(current_edges & historical_edges)
        / len(current_edges)
    )

    return (node_score + edge_score) / 2.0


# ============================================================
# SUCCESSOR-BASED CANDIDATE GENERATION
# ============================================================

def find_successor_candidates(
    observed_graph,
    historical_graph,
    max_hops=MAX_HOPS,
):
    """
    Starting from every observed service, follow outgoing
    edges in the historical graph.

    Return unobserved services reachable within max_hops,
    together with their minimum directed distance.
    """

    observed_nodes = set(observed_graph.nodes())
    distances = {}

    for source in observed_nodes:
        if source not in historical_graph:
            continue

        reachable = nx.single_source_shortest_path_length(
            historical_graph,
            source,
            cutoff=max_hops,
        )

        for candidate, distance in reachable.items():

            # Do not recommend a service already observed.
            if candidate in observed_nodes:
                continue

            # Exclude the starting node itself.
            if distance == 0:
                continue

            if (
                candidate not in distances
                or distance < distances[candidate]
            ):
                distances[candidate] = distance

    return distances


# ============================================================
# SUCCESSOR-BASED RCA RANKING
# ============================================================

def rank_root_cause_candidates(
    current_case,
    observed_graph,
    historical_cases,
):
    """
    Each eligible historical graph contributes evidence.

    Candidate contribution:
        structural similarity / (1 + successor distance)

    Contributions from different historical cases are summed.
    Candidates supported by stronger and closer structural
    matches receive higher scores.
    """

    candidate_scores = defaultdict(float)
    candidate_support = defaultdict(int)

    for historical_case, historical_graph in historical_cases:

        # Never compare a case against itself.
        if historical_case == current_case:
            continue

        # Exclude cases from the same root-service/fault family.
        current_root, current_fault = parse_case_name(current_case)
        historical_root, historical_fault = parse_case_name(
            historical_case
        )

        if (
            current_root == historical_root
            and current_fault == historical_fault
        ):
            continue

        structural_score = graph_match_score(
            observed_graph,
            historical_graph,
        )

        # Ignore unrelated graphs that have no structural match.
        if structural_score <= 0:
            continue

        candidates = find_successor_candidates(
            observed_graph,
            historical_graph,
        )

        for candidate, distance in candidates.items():

            contribution = structural_score / (1 + distance)

            candidate_scores[candidate] += contribution
            candidate_support[candidate] += 1

    # Stable tie-breaking makes runs reproducible.
    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        {
            "candidate": candidate,
            "score": round(score, 8),
            "historical_support": candidate_support[candidate],
        }
        for candidate, score in ranked
    ]


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(rows):
    total_cases = len(rows)

    if total_cases == 0:
        return {}

    root_found = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_rank_sum = 0.0
    no_candidate_count = 0
    candidate_count_sum = 0

    for row in rows:
        ranked = row["ranked_candidates"]
        root = row["ground_truth_root"]

        candidate_names = [
            item["candidate"] for item in ranked
        ]

        candidate_count_sum += len(candidate_names)

        if not candidate_names:
            no_candidate_count += 1

        if root in candidate_names:
            root_found += 1
            rank = candidate_names.index(root) + 1

            if rank == 1:
                top1_hits += 1

            if rank <= 3:
                top3_hits += 1

            if rank <= 5:
                top5_hits += 1

            reciprocal_rank_sum += 1.0 / rank

    return {
        "cases": total_cases,
        "candidate_coverage": root_found / total_cases,
        "no_candidate_rate": no_candidate_count / total_cases,
        "top1_accuracy": top1_hits / total_cases,
        "top3_accuracy": top3_hits / total_cases,
        "top5_accuracy": top5_hits / total_cases,
        "mrr": reciprocal_rank_sum / total_cases,
        "conditional_top1": (
            top1_hits / root_found if root_found else 0.0
        ),
        "avg_candidates": candidate_count_sum / total_cases,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        path for path in BASE_DIR.iterdir()
        if path.is_dir()
        and (path / "historical_incident_graph.gpickle").exists()
        and (path / "partial_graphs").is_dir()
    )

    print("=" * 65)
    print("SUCCESSOR-BASED ROOT CAUSE ANALYSIS")
    print("=" * 65)
    print("Cases found:", len(case_dirs))
    print("Observation levels:", OBSERVATION_LEVELS)
    print("Maximum successor hops:", MAX_HOPS)

    # Load all complete historical graphs once.
    historical_graphs = []

    for case_dir in case_dirs:
        graph = load_graph(
            case_dir / "historical_incident_graph.gpickle"
        )
        historical_graphs.append((case_dir.name, graph))

    all_rows = []

    for level in OBSERVATION_LEVELS:
        level_rows = []

        print(f"\nProcessing observation level: {level}%")

        for index, case_dir in enumerate(case_dirs, start=1):
            case_name = case_dir.name
            ground_truth_root, _ = parse_case_name(case_name)

            partial_path = (
                case_dir
                / "partial_graphs"
                / f"partial_{level}.gpickle"
            )

            if not partial_path.exists():
                print("[SKIP] Missing:", partial_path)
                continue

            observed_graph = load_graph(partial_path)

            ranked = rank_root_cause_candidates(
                current_case=case_name,
                observed_graph=observed_graph,
                historical_cases=historical_graphs,
            )

            candidate_names = [
                item["candidate"] for item in ranked
            ]

            root_rank = (
                candidate_names.index(ground_truth_root) + 1
                if ground_truth_root in candidate_names
                else None
            )

            row = {
                "case": case_name,
                "observation_level": level,
                "observed_nodes": observed_graph.number_of_nodes(),
                "observed_edges": observed_graph.number_of_edges(),
                "ground_truth_root": ground_truth_root,
                "root_rank": root_rank,
                "candidate_count": len(ranked),
                "ranked_candidates": ranked,
            }

            level_rows.append(row)
            all_rows.append(row)

            if index % 15 == 0 or index == len(case_dirs):
                print(f"  Processed {index}/{len(case_dirs)} cases")

        metrics = calculate_metrics(level_rows)

        print(f"\nResults at {level}% observation:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Save detailed per-case rankings.
    detailed_rows = []

    for row in all_rows:
        detailed_rows.append({
            "case": row["case"],
            "observation_level": row["observation_level"],
            "observed_nodes": row["observed_nodes"],
            "observed_edges": row["observed_edges"],
            "ground_truth_root": row["ground_truth_root"],
            "root_rank": row["root_rank"],
            "candidate_count": row["candidate_count"],
            "ranked_candidates_json": json.dumps(
                row["ranked_candidates"]
            ),
        })

    detailed_df = pd.DataFrame(detailed_rows)

    detailed_path = RESULTS_DIR / "successor_rca_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    # Save summary metrics for each observation level.
    summary_rows = []

    for level in OBSERVATION_LEVELS:
        rows = [
            row for row in all_rows
            if row["observation_level"] == level
        ]

        metrics = calculate_metrics(rows)
        metrics["observation_level"] = level
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    columns = [
        "observation_level",
        "cases",
        "candidate_coverage",
        "no_candidate_rate",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "mrr",
        "conditional_top1",
        "avg_candidates",
    ]

    summary_df = summary_df[columns]

    summary_path = RESULTS_DIR / "successor_rca_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Optional Parquet copies; CSV remains the primary output.
    try:
        detailed_df.to_parquet(
            RESULTS_DIR / "successor_rca_detailed.parquet",
            index=False,
        )
        summary_df.to_parquet(
            RESULTS_DIR / "successor_rca_summary.parquet",
            index=False,
        )
        print("\nParquet files saved.")
    except Exception as error:
        print("\nParquet export skipped:", error)
        print("CSV results were saved successfully.")

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print("Detailed results:", detailed_path.resolve())
    print("Summary results:", summary_path.resolve())
    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
