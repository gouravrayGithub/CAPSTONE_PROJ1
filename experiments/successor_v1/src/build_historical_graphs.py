
from pathlib import Path
import pickle
import re

import networkx as nx
import numpy as np
import pandas as pd


# Run this script from the LogRCA-Research project root.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("experiments/successor_v1/graphs")

CASE_PATTERN = "re2ob_*"


def read_injection_time(case_dir):
    path = case_dir / "inject_time.txt"

    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text().strip()
    return int(re.findall(r"\d+", text)[0])


def normalize_service(name):
    mapping = {
        "frontendservice": "frontend",
        "currencyservice": "currencyservice",
        "cartservice": "cartservice",
        "recommendationservice": "recommendationservice",
        "adservice": "adservice",
        "shippingservice": "shippingservice",
        "checkoutservice": "checkoutservice",
        "paymentservice": "paymentservice",
        "emailservice": "emailservice",
        "redis": "redis",
        "productcatalogservice": "productcatalogservice",
    }
    return mapping.get(name, name)


def percentile(series, q):
    if len(series) == 0:
        return 0.0
    return float(np.percentile(series, q))


def build_graph(case_dir):
    injection_time = read_injection_time(case_dir)

    logs = pd.read_parquet(case_dir / "logs.parquet")
    traces = pd.read_parquet(case_dir / "traces.parquet")
    traces = traces.drop_duplicates().copy()

    traces["timestamp_sec"] = (
        traces["startTimeMillis"] // 1000
    )
    traces["duration_ms"] = (
        traces["duration"].astype(float) / 1000.0
    )

    traces["service"] = traces["serviceName"].map(
        normalize_service
    )
    logs["service"] = logs["container_name"].map(
        normalize_service
    )

    before_logs = logs[logs["timestamp"] < injection_time]
    after_logs = logs[logs["timestamp"] >= injection_time]

    before_traces = traces[
        traces["timestamp_sec"] < injection_time
    ]
    after_traces = traces[
        traces["timestamp_sec"] >= injection_time
    ]

    graph = nx.DiGraph()

    services = sorted(
        set(logs["service"].dropna())
        | set(traces["service"].dropna())
    )

    # Add service nodes and their before/after measurements.
    for service in services:
        lb = before_logs[before_logs["service"] == service]
        la = after_logs[after_logs["service"] == service]

        tb = before_traces[before_traces["service"] == service]
        ta = after_traces[after_traces["service"] == service]

        latency_before = tb["duration_ms"].dropna()
        latency_after = ta["duration_ms"].dropna()

        graph.add_node(
            service,
            log_count_before=int(len(lb)),
            log_count_after=int(len(la)),
            trace_count_before=int(len(tb)),
            trace_count_after=int(len(ta)),
            median_latency_before_ms=(
                float(latency_before.median())
                if len(latency_before) else 0.0
            ),
            median_latency_after_ms=(
                float(latency_after.median())
                if len(latency_after) else 0.0
            ),
            p95_latency_before_ms=percentile(
                latency_before, 95
            ),
            p95_latency_after_ms=percentile(
                latency_after, 95
            ),
        )

    # Build directed service dependencies from trace spans.
    # Edge direction: parent service -> child service.
    span_lookup = traces.set_index("spanID")["service"]
    edge_counts = {}

    for _, row in traces.iterrows():
        parent_id = row["parentSpanID"]

        if pd.isna(parent_id):
            continue

        if parent_id not in span_lookup.index:
            continue

        parent_service = span_lookup.loc[parent_id]
        child_service = row["service"]

        # Skip ambiguous duplicate span IDs.
        if isinstance(parent_service, pd.Series):
            continue

        if pd.isna(parent_service):
            continue

        if parent_service == child_service:
            continue

        edge = (parent_service, child_service)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    for (source, target), count in edge_counts.items():
        graph.add_edge(
            source,
            target,
            call_count=int(count),
        )

    graph.graph["case"] = case_dir.name
    graph.graph["injection_time"] = injection_time
    graph.graph["duration_unit"] = "microseconds"
    graph.graph["duration_conversion"] = (
        "duration / 1000 -> milliseconds"
    )
    graph.graph["edge_semantics"] = (
        "observed trace parent service -> child service"
    )

    return graph


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        path for path in DATA_DIR.glob(CASE_PATTERN)
        if path.is_dir()
        and (path / "logs.parquet").exists()
        and (path / "traces.parquet").exists()
        and (path / "inject_time.txt").exists()
    )

    print(f"Cases found: {len(case_dirs)}")

    successful = 0
    failed = []

    for index, case_dir in enumerate(case_dirs, start=1):
        try:
            graph = build_graph(case_dir)

            output_case_dir = OUTPUT_DIR / case_dir.name
            output_case_dir.mkdir(parents=True, exist_ok=True)

            output_file = (
                output_case_dir / "historical_incident_graph.gpickle"
            )

            with output_file.open("wb") as file:
                pickle.dump(graph, file)

            successful += 1
            print(
                f"[{index}/{len(case_dirs)}] {case_dir.name}: "
                f"{graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )

        except Exception as error:
            failed.append((case_dir.name, str(error)))
            print(f"[ERROR] {case_dir.name}: {error}")

    print("\n========== BUILD SUMMARY ==========")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    if failed:
        print("\nFailed cases:")
        for case, error in failed:
            print(f"- {case}: {error}")


if __name__ == "__main__":
    main()
