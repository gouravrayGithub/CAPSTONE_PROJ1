from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import re
import pickle


DATA_DIR = Path("data")


def read_injection_time(case_dir):
    text = (case_dir / "inject_time.txt").read_text().strip()
    return int(re.findall(r"\d+", text)[0])


def service_name(name):
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

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    logs = pd.read_parquet(
        case_dir / "logs.parquet"
    )

    traces = pd.read_parquet(
        case_dir / "traces.parquet"
    )

    # Remove exact duplicate trace records
    traces = traces.drop_duplicates()

    # ---------------------------------------------------------
    # NORMALIZE DATA
    # ---------------------------------------------------------

    traces["timestamp_sec"] = (
        traces["startTimeMillis"] // 1000
    )

    traces["duration_ms"] = (
        traces["duration"].astype(float) / 1000.0
    )

    traces["service"] = traces["serviceName"].map(
        service_name
    )

    logs["service"] = logs["container_name"].map(
        service_name
    )

    # ---------------------------------------------------------
    # TIME WINDOWS
    # ---------------------------------------------------------

    before_logs = logs[
        logs["timestamp"] < injection_time
    ]

    after_logs = logs[
        logs["timestamp"] >= injection_time
    ]

    before_traces = traces[
        traces["timestamp_sec"] < injection_time
    ]

    after_traces = traces[
        traces["timestamp_sec"] >= injection_time
    ]

    # ---------------------------------------------------------
    # CREATE GRAPH
    # ---------------------------------------------------------

    G = nx.DiGraph()

    # ---------------------------------------------------------
    # SERVICE NODES
    # ---------------------------------------------------------

    services = sorted(
        set(logs["service"].dropna())
        | set(traces["service"].dropna())
    )

    for service in services:

        log_before = before_logs[
            before_logs["service"] == service
        ]

        log_after = after_logs[
            after_logs["service"] == service
        ]

        trace_before = before_traces[
            before_traces["service"] == service
        ]

        trace_after = after_traces[
            after_traces["service"] == service
        ]

        before_latency = trace_before[
            "duration_ms"
        ].dropna()

        after_latency = trace_after[
            "duration_ms"
        ].dropna()

        G.add_node(
            service,

            log_count_before=int(
                len(log_before)
            ),

            log_count_after=int(
                len(log_after)
            ),

            trace_count_before=int(
                len(trace_before)
            ),

            trace_count_after=int(
                len(trace_after)
            ),

            median_latency_before_ms=(
                float(before_latency.median())
                if len(before_latency)
                else 0.0
            ),

            median_latency_after_ms=(
                float(after_latency.median())
                if len(after_latency)
                else 0.0
            ),

            p95_latency_before_ms=percentile(
                before_latency,
                95
            ),

            p95_latency_after_ms=percentile(
                after_latency,
                95
            ),
        )

    # ---------------------------------------------------------
    # SERVICE DEPENDENCY EDGES
    # ---------------------------------------------------------

    span_lookup = traces.set_index(
        "spanID"
    )["service"]

    edges = {}

    for _, row in traces.iterrows():

        parent_id = row["parentSpanID"]

        if pd.isna(parent_id):
            continue

        child_service = row["service"]

        if parent_id not in span_lookup.index:
            continue

        parent_service = span_lookup.loc[parent_id]

        if pd.isna(parent_service):
            continue

        if parent_service == child_service:
            continue

        key = (
            parent_service,
            child_service
        )

        edges[key] = edges.get(key, 0) + 1

    for (source, target), count in edges.items():

        G.add_edge(
            source,
            target,
            call_count=int(count)
        )

    # ---------------------------------------------------------
    # GRAPH METADATA
    # ---------------------------------------------------------

    G.graph["case"] = case_dir.name
    G.graph["injection_time"] = injection_time
    G.graph["duration_unit"] = "microseconds"
    G.graph["duration_conversion"] = (
        "duration / 1000 -> milliseconds"
    )

    return G


def main():

    cases = sorted(
        DATA_DIR.glob("re2ob_*")
    )

    print("=" * 70)
    print("BUILDING HISTORICAL INCIDENT GRAPH KNOWLEDGE BANK")
    print("=" * 70)

    print()
    print("Cases found:", len(cases))
    print()

    successful = 0
    failed = []

    for i, case_dir in enumerate(cases, start=1):

        print(
            f"[{i}/{len(cases)}] "
            f"{case_dir.name}",
            end=" ... ",
            flush=True
        )

        try:

            required_files = [
                case_dir / "logs.parquet",
                case_dir / "traces.parquet",
                case_dir / "inject_time.txt",
            ]

            if not all(
                path.exists()
                for path in required_files
            ):
                print("SKIPPED: missing files")
                continue

            G = build_graph(case_dir)

            output = (
                case_dir
                / "historical_incident_graph.gpickle"
            )

            with open(output, "wb") as f:
                pickle.dump(G, f)

            print(
                f"OK "
                f"(nodes={G.number_of_nodes()}, "
                f"edges={G.number_of_edges()})"
            )

            successful += 1

        except Exception as e:

            print("FAILED")

            failed.append(
                (case_dir.name, str(e))
            )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        len(failed)
    )

    if failed:

        print()
        print("Failures:")

        for case, error in failed:

            print()
            print(case)
            print(error)


if __name__ == "__main__":
    main()
