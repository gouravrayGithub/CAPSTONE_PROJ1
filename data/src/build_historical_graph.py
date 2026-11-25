from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import re



CASE_DIR = Path("data/re2ob_checkoutservice_cpu_1")


def read_injection_time(case_dir):
    text = (case_dir / "inject_time.txt").read_text().strip()
    return int(re.findall(r"\d+", text)[0])


def service_name(name):
    """
    Normalize trace service names to the names used by the logs.
    """
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


def main():

    print("Building historical incident graph")
    print("Case:", CASE_DIR.name)
    print()

    injection_time = read_injection_time(CASE_DIR)

    print("Injection time:", injection_time)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    logs = pd.read_parquet(
        CASE_DIR / "logs.parquet"
    )

    traces = pd.read_parquet(
        CASE_DIR / "traces.parquet"
    )
     # Remove exact duplicate trace records
    traces = traces.drop_duplicates()
    
    # Convert trace timestamp to seconds
    traces["timestamp_sec"] = (
        traces["startTimeMillis"] // 1000
    )

    # Duration is stored in microseconds.
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

    print()
    print("Logs:")
    print("  before:", len(before_logs))
    print("  after :", len(after_logs))

    print()
    print("Traces:")
    print("  before:", len(before_traces))
    print("  after :", len(after_traces))

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

            # Log evidence
            log_count_before=int(len(log_before)),
            log_count_after=int(len(log_after)),

            # Trace evidence
            trace_count_before=int(len(trace_before)),
            trace_count_after=int(len(trace_after)),

            # Robust latency statistics
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
                before_latency, 95
            ),

            p95_latency_after_ms=percentile(
                after_latency, 95
            ),
        )

    # ---------------------------------------------------------
    # SERVICE DEPENDENCY EDGES
    # ---------------------------------------------------------
    #
    # We use parentSpanID to identify parent-child
    # relationships within traces.
    #
    # IMPORTANT:
    # This is an observed dependency relationship,
    # NOT a proven causal relationship.
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

    G.graph["case"] = CASE_DIR.name
    G.graph["injection_time"] = injection_time
    G.graph["duration_unit"] = "microseconds"
    G.graph["duration_conversion"] = "duration / 1000 -> milliseconds"

    # ---------------------------------------------------------
       # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output = CASE_DIR / "historical_incident_graph.gpickle"

    import pickle

    with open(output, "wb") as f:
        pickle.dump(G, f)

    print()
    print("Graph created:")
    print("  nodes:", G.number_of_nodes())
    print("  edges:", G.number_of_edges())
    print("  output:", output)

    print()
    print("Nodes:")

    for node, data in G.nodes(data=True):
        print(
            node,
            "| logs:",
            data["log_count_before"],
            "->",
            data["log_count_after"],
            "| traces:",
            data["trace_count_before"],
            "->",
            data["trace_count_after"],
            "| p95:",
            round(data["p95_latency_before_ms"], 3),
            "->",
            round(data["p95_latency_after_ms"], 3),
        )

    print()
    print("Edges:")

    for source, target, data in G.edges(data=True):
        print(
            f"{source} -> {target} "
            f"| calls={data['call_count']}"
        )
        


if __name__ == "__main__":
    main()
