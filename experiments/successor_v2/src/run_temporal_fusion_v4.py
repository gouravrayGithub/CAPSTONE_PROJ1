from pathlib import Path
import csv
import json
import sys

import pandas as pd

# Reuse the existing V3 scoring functions without modifying V3.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_rca_methods_v3 import (
    load_graph,
    parse_case,
    candidate_scores,
    evaluate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_DIR = (
    PROJECT_ROOT / "experiments" / "successor_v2" / "graphs"
)
TEMPORAL_FILE = (
    PROJECT_ROOT / "experiments" / "successor_v2" / "results"
    / "temporal_features" / "service_temporal_features.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "successor_v2" / "results"
    / "temporal_fusion_v4"
)

LEVELS = [20, 40, 60, 80, 100]

# Fixed exploratory weight. Do not tune this using the test results.
TEMPORAL_WEIGHT = 0.30


def rank_strength(ranked):
    """
    Convert an ordered candidate list to [0, 1].
    Rank 1 receives 1.0; the last candidate receives 0.0.
    """
    n = len(ranked)

    if n <= 1:
        return {item["candidate"]: 1.0 for item in ranked}

    return {
        item["candidate"]: (n - position) / (n - 1)
        for position, item in enumerate(ranked, start=1)
    }


def temporal_strength(scores, observed_services):
    """
    Normalize temporal scores using observed services only.
    Hidden services receive no temporal evidence.
    """
    observed = sorted(observed_services)

    if not observed:
        return {}

    ordered = sorted(
        observed,
        key=lambda service: (-scores.get(service, 0.0), service),
    )

    n = len(ordered)

    if n == 1:
        return {ordered[0]: 1.0}

    return {
        service: (n - position) / (n - 1)
        for position, service in enumerate(ordered, start=1)
    }


def calculate_summary(rows):
    output = []

    groups = sorted({
        (row["observation_level"], row["method"], row["root_visibility"])
        for row in rows
    })

    for level, method, visibility in groups:
        subset = [
            row for row in rows
            if row["observation_level"] == level
            and row["method"] == method
            and row["root_visibility"] == visibility
        ]

        if not subset:
            continue

        n = len(subset)

        output.append({
            "observation_level": level,
            "method": method,
            "root_visibility": visibility,
            "cases": n,
            "top1": sum(r["top1"] for r in subset) / n,
            "top3": sum(r["top3"] for r in subset) / n,
            "top5": sum(r["top5"] for r in subset) / n,
            "mrr": sum(r["rr"] for r in subset) / n,
        })

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporal_df = pd.read_csv(TEMPORAL_FILE)

    # Only post-injection windows contribute to the RCA evidence.
    post = temporal_df[temporal_df["period"] == "after"].copy()

    # Average across windows to emphasize sustained anomalies.
    temporal_scores = (
        post.groupby(["case", "service"])["anomaly_score"]
        .mean()
        .to_dict()
    )

    graph_dirs = sorted(
        path for path in GRAPH_DIR.iterdir()
        if path.is_dir()
        and (path / "historical_incident_graph.gpickle").exists()
        and (path / "partial_graphs").is_dir()
    )

    historical_cases = [
        (
            path.name,
            load_graph(path / "historical_incident_graph.gpickle"),
        )
        for path in graph_dirs
    ]

    all_services = set()
    for _, graph in historical_cases:
        all_services.update(graph.nodes())

    detail_rows = []

    for level in LEVELS:
        for case_dir in graph_dirs:
            partial_path = (
                case_dir / "partial_graphs"
                / f"partial_{level}.gpickle"
            )

            if not partial_path.exists():
                continue

            current = load_graph(partial_path)
            case_name = case_dir.name
            true_root, fault_type, repetition = parse_case(case_name)

            observed = set(current.nodes())
            root_visibility = (
                "visible" if true_root in observed else "hidden"
            )

            # Temporal anomaly evidence available for this case only.
            case_temporal = {
                service: float(score)
                for (case, service), score in temporal_scores.items()
                if case == case_name
            }

            # Existing graph-based methods.
            successor_ranked = candidate_scores(
                current, historical_cases, case_name, "successor"
            )

            successor_anomaly_ranked = candidate_scores(
                current, historical_cases, case_name,
                "successor_anomaly"
            )

            # Temporal-only ranking: observed services have scores;
            # unobserved services have no temporal evidence.
            # Temporal-only ranking includes observed services only.
            # Hidden services have no available temporal evidence.
            temporal_ranked = sorted(
                observed,
                key=lambda service: (
                    -case_temporal.get(service, 0.0),
                    service,
                ),
            )

            temporal_ranked = [
                {
                    "candidate": service,
                    "score": case_temporal.get(service, 0.0)
                    if service in observed else None,
                }
                for service in temporal_ranked
            ]

            # Rank-based fusion avoids mixing incompatible raw scales.
            base_strength = rank_strength(successor_ranked)
            temp_strength = temporal_strength(
                case_temporal, observed
            )

            fusion_ranked = []

            for service in all_services:
                graph_score = base_strength.get(service, 0.0)

                if service in observed:
                    temporal_score = temp_strength.get(service, 0.0)
                    # Add temporal evidence without reducing graph evidence.
                    fused_score = (
                        graph_score
                        + TEMPORAL_WEIGHT * temporal_score
                    )
                else:
                    # Preserve the full graph score for hidden services.
                    fused_score = graph_score

                fusion_ranked.append({
                    "candidate": service,
                    "score": fused_score,
                })

            fusion_ranked.sort(
                key=lambda item: (-item["score"], item["candidate"])
            )

            methods = {
                        "successor": successor_ranked,
                        "successor_anomaly": successor_anomaly_ranked,
                        "temporal_only_partial": temporal_ranked,
                        "successor_temporal_fusion": fusion_ranked,
            }

            for method, ranked in methods.items():
                metrics = evaluate(ranked, true_root)

                detail_rows.append({
                    "case": case_name,
                    "observation_level": level,
                    "method": method,
                    "true_root": true_root,
                    "fault_type": fault_type,
                    "repetition": repetition,
                    "root_visibility": root_visibility,
                    "observed_service_count": len(observed),
                    "temporal_weight": (
                        TEMPORAL_WEIGHT
                        if method == "successor_temporal_fusion"
                        else None
                    ),
                    **metrics,
                    "ranked_candidates": json.dumps(ranked),
                })

    detail_file = OUTPUT_DIR / "temporal_fusion_detailed.csv"
    summary_file = OUTPUT_DIR / "temporal_fusion_summary.csv"

    if not detail_rows:
        raise RuntimeError(
            "No results generated. Check graph and temporal input files."
        )

    with detail_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(detail_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = calculate_summary(detail_rows)

    with summary_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n===== TEMPORAL FUSION RESULTS =====")
    print("Cases:", len(graph_dirs))
    print("Temporal weight:", TEMPORAL_WEIGHT)
    print("\nOverall results by observation level:")

    for level in LEVELS:
        for method in [
            "successor",
            "successor_anomaly",
            "temporal_only_partial",
            "successor_temporal_fusion",
        ]:
            subset = [
                row for row in detail_rows
                if row["observation_level"] == level
                and row["method"] == method
            ]

            if not subset:
                continue

            n = len(subset)

            top1 = sum(r["top1"] for r in subset) / n
            top3 = sum(r["top3"] for r in subset) / n
            top5 = sum(r["top5"] for r in subset) / n
            mrr = sum(r["rr"] for r in subset) / n

            print(
                f"{level:>3}% | {method:<27} | "
                f"Top1={top1:.3f} | Top3={top3:.3f} | "
                f"Top5={top5:.3f} | MRR={mrr:.3f}"
            )

    print("\n===== ROOT VISIBILITY BREAKDOWN =====")

    for row in summary_rows:
        if row["method"] in (
            "successor",
            "successor_anomaly",
            "temporal_only_partial",
            "successor_temporal_fusion",
        ):
            print(
                f"{row['observation_level']:>3}% | "
                f"{row['root_visibility']:<7} | "
                f"{row['method']:<27} | "
                f"n={row['cases']:>2} | "
                f"Top1={row['top1']:.3f} | "
                f"Top3={row['top3']:.3f} | "
                f"Top5={row['top5']:.3f} | "
                f"MRR={row['mrr']:.3f}"
            )

    print("\nDetailed results:", detail_file)
    print("Summary results:", summary_file)


if __name__ == "__main__":
    main()
