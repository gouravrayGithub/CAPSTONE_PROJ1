#!/usr/bin/env python3
"""Build a leakage-aware normalized incident bundle from saved V3 artifacts.

Run from experiments/llm_rca_v1, for example:
  python src/prepare_case.py --case re2ob_checkoutservice_cpu_1 --observation-level 100

This script reads saved CSV/Parquet/NetworkX graph files only. It does not rerun V3.
Ground-truth fields (true_root, fault_type, root_rank, top1/top3/top5, rr) are
never included in the generated diagnostic bundle. The source case name is also
replaced with an opaque case ID because the dataset case name encodes the root/fault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = WORKSPACE / "baseline_results"
DATA_DIR = WORKSPACE / "data"
GRAPH_ROOT = PROJECT_ROOT / "experiments" / "successor_v2" / "graphs"


def safe_number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return int(number) if number.is_integer() else number


def load_graph(case: str, level: int) -> dict[str, Any]:
    graph_path = GRAPH_ROOT / case / "partial_graphs" / f"partial_{level}.gpickle"
    if not graph_path.exists():
        raise FileNotFoundError(f"Partial graph not found: {graph_path}")
    # NetworkX graph objects are pickled. Never load graph files from untrusted sources.
    with graph_path.open("rb") as f:
        graph = pickle.load(f)
    nodes = sorted(str(n) for n in graph.nodes())
    edges = sorted({(str(u), str(v)) for u, v in graph.edges()})
    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in edges],
        "metadata": {"source": "saved_partial_graph", "observation_level": level},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Dataset case ID, used locally to select files")
    parser.add_argument("--observation-level", type=int, choices=[20, 40, 60, 80, 100], default=100)
    parser.add_argument("--method", default="successor_anomaly", choices=["successor_anomaly", "successor", "structural", "ancestor_anomaly"])
    parser.add_argument("--output", default="examples/prepared_case.json")
    args = parser.parse_args()

    ranking_path = BASELINE_DIR / "rca_methods_v3_detailed.csv"
    features_path = DATA_DIR / "incident_service_features.parquet"
    temporal_path = DATA_DIR / "service_temporal_features.csv"
    for path in [ranking_path, features_path, temporal_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    rankings = pd.read_csv(ranking_path)
    selected = rankings[
        (rankings["case"] == args.case)
        & (rankings["observation_level"] == args.observation_level)
        & (rankings["method"] == args.method)
    ]
    if selected.empty:
        raise ValueError(f"No saved ranking for case={args.case}, level={args.observation_level}, method={args.method}")
    row = selected.iloc[0]
    # Parse candidate scores but explicitly omit all evaluation/ground-truth columns.
    ranked_raw = json.loads(row["ranked_candidates"])
    initial_ranking = [
        {"service": str(item["candidate"]), "score": safe_number(item.get("score")) or 0.0}
        for item in ranked_raw
        if item.get("candidate")
    ]

    features = pd.read_parquet(features_path)
    case_features = features[features["case"] == args.case].copy()
    if case_features.empty:
        raise ValueError(f"No service features found for {args.case}")

    temporal = pd.read_csv(temporal_path)

    # Exclude the incomplete trailing window at 720 seconds.
    temporal = temporal[temporal["window_start_sec"] < 720].copy()

    case_temporal = temporal[temporal["case"] == args.case].copy()

    graph = load_graph(args.case, args.observation_level)
    graph_nodes = set(graph["nodes"])
    all_services = set(case_features["service"].astype(str))
    all_services.update(str(x["service"]) for x in initial_ranking)
    all_services.update(graph_nodes)

    diagnostic_bundles: dict[str, dict[str, Any]] = {}
    feature_cols = [
        "log_before", "log_after", "trace_before", "trace_after",
        "median_before_ms", "median_after_ms", "p95_before_ms", "p95_after_ms",
        "log_change_pct", "trace_change_pct", "median_change_pct", "p95_change_pct",
        "in_degree", "out_degree",
    ]
    for service in sorted(all_services):
        service_rows = case_features[case_features["service"].astype(str) == service]
        feature_evidence: dict[str, Any] = {}
        if not service_rows.empty:
            feature_row = service_rows.iloc[0]
            feature_evidence = {col: safe_number(feature_row.get(col)) for col in feature_cols if col in case_features.columns}

        service_temporal = case_temporal[case_temporal["service"].astype(str) == service]
        windows: list[dict[str, Any]] = []
        if not service_temporal.empty:
            for _, tr in service_temporal.sort_values("window_start_sec").iterrows():
                windows.append({
                    "window_start_sec": safe_number(tr.get("window_start_sec")),
                    "period": str(tr.get("period", "unknown")),
                    "log_count": safe_number(tr.get("log_count")),
                    "trace_count": safe_number(tr.get("trace_count")),
                    "median_latency_ms": safe_number(tr.get("median_latency_ms")),
                    "p95_latency_ms": safe_number(tr.get("p95_latency_ms")),
                    "anomaly_score": safe_number(tr.get("anomaly_score")),
                })
        scores = [w["anomaly_score"] for w in windows if w["anomaly_score"] is not None]
        diagnostic_bundles[service] = {
            "service": service,
            "observed_in_partial_graph": service in graph_nodes,
            "service_features": feature_evidence,
            "temporal_summary": {
                "window_count": len(windows),
                "mean_anomaly_score": (sum(scores) / len(scores)) if scores else None,
                "max_anomaly_score": max(scores) if scores else None,
                "windows": windows,
            },
            "evidence_note": "Feature values are descriptive evidence, not proof of causality.",
        }

    opaque_id = "incident_" + hashlib.sha256(
        f"{args.case}|{args.observation_level}|{args.method}".encode("utf-8")
    ).hexdigest()[:12]
    bundle = {
        "case_id": opaque_id,
        "observation_level": args.observation_level,
        "ranking_method": args.method,
        "initial_ranking": initial_ranking,
        "graph": graph,
        "diagnostic_bundles": diagnostic_bundles,
        "data_provenance": {
            "ranking": "saved rca_methods_v3_detailed.csv; V3 was not rerun",
            "service_features": "incident_service_features.parquet",
            "temporal_features": "service_temporal_features.csv",
            "graph": f"saved partial graph at {args.observation_level}% observation",
        },
        "safety_note": "Ground-truth labels and evaluation metrics are intentionally excluded from this bundle.",
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared diagnostic bundle: {output_path}")
    print(f"Opaque case ID: {opaque_id}")
    print(f"Initial candidates: {len(initial_ranking)}")
    print(f"Graph nodes: {len(graph['nodes'])}; graph edges: {len(graph['edges'])}")
    print(f"Diagnostic service bundles: {len(diagnostic_bundles)}")
    print("Ground-truth fields excluded; saved V3 ranking reused without rerunning V3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
