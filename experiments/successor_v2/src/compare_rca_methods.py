
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import pickle
import re

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DIR = PROJECT_ROOT / "experiments" / "successor_v2" / "graphs"
RESULT_DIR = PROJECT_ROOT / "experiments" / "successor_v2" / "results"

LEVELS = [20, 40, 60, 80, 100]
METHODS = [
    "structural",
    "successor",
    "successor_anomaly",
]


def load_graph(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def parse_case(case_name):
    # Example: re2ob_productcatalogservice_cpu_2
    match = re.match(
        r"re2ob_(.+)_(cpu|delay|disk|loss|mem|socket)_(\d+)$",
        case_name,
    )
    if not match:
        raise ValueError(f"Unexpected case name: {case_name}")

    return match.group(1), match.group(2), int(match.group(3))


def safe_relative_change(before, after):
    """Symmetric relative change, bounded between -1 and 1."""
    before = float(before or 0)
    after = float(after or 0)

    denominator = abs(before) + abs(after)
    if denominator == 0:
        return 0.0

    return (after - before) / denominator


def node_anomaly_score(attrs):
    """
    Evidence score from available before/after graph statistics.
    This is a heuristic, not a trained anomaly detector.
    """
    feature_changes = []

    for before_key, after_key in [
        ("log_count_before", "log_count_after"),
        ("trace_count_before", "trace_count_after"),
        ("median_latency_before_ms", "median_latency_after_ms"),
        ("p95_latency_before_ms", "p95_latency_after_ms"),
    ]:
        before = float(attrs.get(before_key, 0) or 0)
        after = float(attrs.get(after_key, 0) or 0)

        # If both values are zero, this feature supplies no evidence.
        if before == 0 and after == 0:
            continue

        feature_changes.append(
            abs(safe_relative_change(before, after))
        )

    if not feature_changes:
        return 0.0

    return sum(feature_changes) / len(feature_changes)


def structural_score(observed, historical):
    observed_nodes = set(observed.nodes)
    historical_nodes = set(historical.nodes)

    node_score = (
        len(observed_nodes & historical_nodes)
        / max(1, len(observed_nodes))
    )

    observed_edges = set(observed.edges)
    historical_edges = set(historical.edges)

    if not observed_edges:
        return node_score

    edge_score = (
        len(observed_edges & historical_edges)
        / max(1, len(observed_edges))
    )

    return (node_score + edge_score) / 2.0


def successor_distances(observed, historical):
    """
    Return shortest directed successor distances from observed nodes.
    Distances are structural evidence, not proof of causation.
    """
    distances = {}

    for source in observed.nodes:
        if source not in historical:
            continue

        reachable = nx.single_source_shortest_path_length(
            historical, source, cutoff=3
        )

        for candidate, distance in reachable.items():
            if candidate == source:
                continue

            if (
                candidate not in distances
                or distance < distances[candidate]
            ):
                distances[candidate] = distance

    return distances


def candidate_scores(current, historical_cases, current_case, method):
    """
    Score the entire service universe.
    Never uses the true root label to build the candidate list.
    """
    all_services = set(current.nodes)

    for _, graph in historical_cases:
        all_services.update(graph.nodes)

    scores = defaultdict(float)
    support = defaultdict(int)
    anomaly_evidence = defaultdict(list)

    for historical_case, historical in historical_cases:
        if historical_case == current_case:
            continue

        # Avoid using the same root/fault family as a near-duplicate.
        hist_root, hist_fault, _ = parse_case(historical_case)
        current_root, current_fault, _ = parse_case(current_case)

        if hist_root == current_root and hist_fault == current_fault:
            continue

        similarity = structural_score(current, historical)
        if similarity <= 0:
            continue

        distances = successor_distances(current, historical)

        for candidate in all_services:
            # Historical graph evidence is candidate-specific only when
            # a directed path reaches that service.
            distance = distances.get(candidate)

            if method == "structural":
                contribution = similarity

            elif method in ("successor", "successor_anomaly"):
                if distance is None:
                    contribution = 0.0
                else:
                    contribution = similarity / (1 + distance)

            else:
                raise ValueError(f"Unknown method: {method}")

            scores[candidate] += contribution
            if contribution > 0:
                support[candidate] += 1

        if method == "successor_anomaly":
            # Use the observed node attributes as current evidence.
            # Historical structural matches weight this evidence.
            for candidate, attrs in current.nodes(data=True):
                anomaly_evidence[candidate].append(
                    similarity * node_anomaly_score(attrs)
                )

    if method == "successor_anomaly":
        for candidate in all_services:
            values = anomaly_evidence[candidate]
            if values:
                scores[candidate] += sum(values) / len(values)

    ranked = sorted(
        all_services,
        key=lambda candidate: (
            -scores[candidate],
            -support[candidate],
            candidate,
        ),
    )

    return [
        {
            "candidate": candidate,
            "score": round(scores[candidate], 8),
            "historical_support": support[candidate],
        }
        for candidate in ranked
    ]


def evaluate(ranked, true_root):
    names = [item["candidate"] for item in ranked]

    if true_root not in names:
        return {
            "root_rank": None,
            "top1": 0,
            "top3": 0,
            "top5": 0,
            "rr": 0.0,
        }

    rank = names.index(true_root) + 1

    return {
        "root_rank": rank,
        "top1": int(rank <= 1),
        "top3": int(rank <= 3),
        "top5": int(rank <= 5),
        "rr": 1.0 / rank,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    graph_dirs = sorted(
        path for path in GRAPH_DIR.iterdir()
        if path.is_dir()
        and (path / "historical_incident_graph.gpickle").exists()
    )

    historical_cases = [
        (
            path.name,
            load_graph(path / "historical_incident_graph.gpickle"),
        )
        for path in graph_dirs
    ]

    detail_rows = []

    for level in LEVELS:
        for case_dir in graph_dirs:
            partial_path = (
                case_dir / "partial_graphs" / f"partial_{level}.gpickle"
            )
            if not partial_path.exists():
                continue

            current = load_graph(partial_path)
            true_root, fault_type, repetition = parse_case(case_dir.name)

            for method in METHODS:
                ranked = candidate_scores(
                    current,
                    historical_cases,
                    case_dir.name,
                    method,
                )

                metrics = evaluate(ranked, true_root)

                detail_rows.append({
                    "case": case_dir.name,
                    "observation_level": level,
                    "method": method,
                    "true_root": true_root,
                    "fault_type": fault_type,
                    "repetition": repetition,
                    **metrics,
                    "ranked_candidates": json.dumps(ranked),
                })

    detail_path = RESULT_DIR / "rca_methods_v2_detailed.csv"

    if detail_rows:
        with open(detail_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(detail_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_rows = []

    for level in LEVELS:
        for method in METHODS:
            rows = [
                row for row in detail_rows
                if row["observation_level"] == level
                and row["method"] == method
            ]

            if not rows:
                continue

            count = len(rows)

            summary_rows.append({
                "observation_level": level,
                "method": method,
                "cases": count,
                "top1_accuracy": sum(r["top1"] for r in rows) / count,
                "top3_accuracy": sum(r["top3"] for r in rows) / count,
                "top5_accuracy": sum(r["top5"] for r in rows) / count,
                "mrr": sum(r["rr"] for r in rows) / count,
                "mean_root_rank": sum(
                    r["root_rank"] or 11 for r in rows
                ) / count,
            })

    summary_path = RESULT_DIR / "rca_methods_v2_summary.csv"

    if summary_rows:
        with open(summary_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(summary_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n===== V2 RCA METHOD COMPARISON =====")
    for row in summary_rows:
        print(
            f"{row['observation_level']:>3}% | "
            f"{row['method']:<20} | "
            f"Top1={row['top1_accuracy']:.3f} | "
            f"Top3={row['top3_accuracy']:.3f} | "
            f"Top5={row['top5_accuracy']:.3f} | "
            f"MRR={row['mrr']:.3f}"
        )

    print("\nDetailed results:", detail_path)
    print("Summary results:", summary_path)


if __name__ == "__main__":
    main()
