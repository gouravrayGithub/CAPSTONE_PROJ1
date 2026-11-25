import pandas as pd
import networkx as nx
from pathlib import Path
import pickle


DATA_DIR = Path("data")


# --------------------------------------------------
# Service-name normalization
# --------------------------------------------------

SERVICE_MAP = {
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


def normalize_service(service):
    return SERVICE_MAP.get(service, service)


# --------------------------------------------------
# Build one current incident graph
# --------------------------------------------------

def build_graph(case_dir):

    case = case_dir.name

    # ----------------------------------------------
    # Read injection time
    # ----------------------------------------------

    inject_file = case_dir / "inject_time.txt"

    if not inject_file.exists():
        print(f"SKIP {case}: inject_time.txt missing")
        return False

    with open(inject_file, "r") as f:
        inject_time = int(f.read().strip())

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    traces_file = case_dir / "traces.parquet"
    events_file = case_dir / "events.parquet"

    if not traces_file.exists():
        print(f"SKIP {case}: traces.parquet missing")
        return False

    if not events_file.exists():
        print(f"SKIP {case}: events.parquet missing")
        return False

    traces = pd.read_parquet(traces_file)
    events = pd.read_parquet(events_file)

    # ----------------------------------------------
    # Remove exact duplicate trace rows
    # ----------------------------------------------

    traces = traces.drop_duplicates().copy()

    # ----------------------------------------------
    # Before / after windows
    # ----------------------------------------------

    before_start = inject_time - 600
    before_end = inject_time

    after_start = inject_time
    after_end = inject_time + 600

    # ==================================================
    # 1. EVENT-LEVEL EVIDENCE
    # ==================================================

    before_events = events[
        (events["timestamp"] >= before_start) &
        (events["timestamp"] < before_end)
    ]

    after_events = events[
        (events["timestamp"] >= after_start) &
        (events["timestamp"] <= after_end)
    ]

    before_counts = (
        before_events["service"]
        .value_counts()
        .rename("event_count_before")
    )

    after_counts = (
        after_events["service"]
        .value_counts()
        .rename("event_count_after")
    )

    event_stats = pd.concat(
        [before_counts, after_counts],
        axis=1
    ).fillna(0)

    event_stats["event_change_pct"] = (
        (
            event_stats["event_count_after"]
            - event_stats["event_count_before"]
        )
        / event_stats["event_count_before"].replace(0, 1)
    ) * 100

    # ==================================================
    # 2. TRACE-LEVEL EVIDENCE
    # ==================================================

    traces["timestamp"] = (
        traces["startTimeMillis"] // 1000
    )

    before_traces = traces[
        (traces["timestamp"] >= before_start) &
        (traces["timestamp"] < before_end)
    ].copy()

    after_traces = traces[
        (traces["timestamp"] >= after_start) &
        (traces["timestamp"] <= after_end)
    ].copy()

    before_traces["duration_raw"] = (
        before_traces["duration"]
    )

    after_traces["duration_raw"] = (
        after_traces["duration"]
    )

    before_duration = (
        before_traces
        .groupby("serviceName")["duration_raw"]
        .agg(
            duration_mean_before="mean",
            duration_median_before="median",
            duration_p95_before=lambda x: x.quantile(0.95)
        )
    )

    after_duration = (
        after_traces
        .groupby("serviceName")["duration_raw"]
        .agg(
            duration_mean_after="mean",
            duration_median_after="median",
            duration_p95_after=lambda x: x.quantile(0.95)
        )
    )

    trace_stats = before_duration.join(
        after_duration,
        how="outer"
    )

    # Normalize trace service names
    trace_stats.index = trace_stats.index.map(
        normalize_service
    )

    # ==================================================
    # 3. MERGE SERVICE EVIDENCE
    # ==================================================

    service_stats = event_stats.copy()

    service_stats = service_stats.join(
        trace_stats,
        how="outer"
    ).fillna(0)

    # Duration changes
    service_stats["duration_mean_change_pct"] = (
        (
            service_stats["duration_mean_after"]
            - service_stats["duration_mean_before"]
        )
        / service_stats["duration_mean_before"].replace(0, 1)
    ) * 100

    service_stats["duration_p95_change_pct"] = (
        (
            service_stats["duration_p95_after"]
            - service_stats["duration_p95_before"]
        )
        / service_stats["duration_p95_before"].replace(0, 1)
    ) * 100

    # ==================================================
    # 4. BUILD GRAPH
    # ==================================================

    G = nx.DiGraph()

    # --------------------------------------------------
    # Add service nodes
    # --------------------------------------------------

    for service in service_stats.index:

        service = normalize_service(service)

        row = service_stats.loc[service]

        G.add_node(
            service,

            event_count_before=float(
                row["event_count_before"]
            ),

            event_count_after=float(
                row["event_count_after"]
            ),

            event_change_pct=float(
                row["event_change_pct"]
            ),

            duration_mean_before=float(
                row["duration_mean_before"]
            ),

            duration_mean_after=float(
                row["duration_mean_after"]
            ),

            duration_p95_before=float(
                row["duration_p95_before"]
            ),

            duration_p95_after=float(
                row["duration_p95_after"]
            ),

            duration_mean_change_pct=float(
                row["duration_mean_change_pct"]
            ),

            duration_p95_change_pct=float(
                row["duration_p95_change_pct"]
            )
        )

    # ==================================================
    # 5. BUILD SERVICE DEPENDENCY EDGES
    # ==================================================

    incident_traces = traces[
        (traces["timestamp"] >= before_start) &
        (traces["timestamp"] <= after_end)
    ].copy()

    # Span → service lookup
    span_service = (
        incident_traces
        .set_index("spanID")["serviceName"]
        .to_dict()
    )

    edge_counts = {}

    for _, row in incident_traces.iterrows():

        parent = row["parentSpanID"]
        child_service = normalize_service(
            row["serviceName"]
        )

        if pd.isna(parent) or parent == "":
            continue

        parent_service = span_service.get(parent)

        if parent_service is None:
            continue

        parent_service = normalize_service(
            parent_service
        )

        # Ignore intra-service calls
        if parent_service == child_service:
            continue

        key = (
            parent_service,
            child_service
        )

        edge_counts[key] = (
            edge_counts.get(key, 0) + 1
        )

    # Add edges
    for (source, target), count in edge_counts.items():

        G.add_edge(
            source,
            target,
            observed_calls=int(count)
        )

    # ==================================================
    # 6. GRAPH METADATA
    # ==================================================

    G.graph["case"] = case
    G.graph["injection_time"] = inject_time
    G.graph["before_start"] = before_start
    G.graph["before_end"] = before_end
    G.graph["after_start"] = after_start
    G.graph["after_end"] = after_end

    G.graph["duration_unit"] = (
        "original RCAEval duration unit"
    )

    # ==================================================
    # 7. SAVE
    # ==================================================

    output = (
        case_dir /
        "current_incident_graph.gpickle"
    )

    with open(output, "wb") as f:
        pickle.dump(G, f)

    return True


# ==================================================
# PROCESS ALL CASES
# ==================================================

case_dirs = sorted(
    p for p in DATA_DIR.glob("re2ob_*")
    if p.is_dir()
)

print("=" * 80)
print("BUILDING CURRENT INCIDENT GRAPHS")
print("=" * 80)

success = 0
failed = 0

for i, case_dir in enumerate(case_dirs, start=1):

    case = case_dir.name

    try:

        ok = build_graph(case_dir)

        if ok:
            success += 1
            print(
                f"[{i}/{len(case_dirs)}] "
                f"OK   {case}"
            )
        else:
            failed += 1

    except Exception as e:

        failed += 1

        print(
            f"[{i}/{len(case_dirs)}] "
            f"ERROR {case}: {e}"
        )


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Cases found : {len(case_dirs)}")
print(f"Graphs built: {success}")
print(f"Failed      : {failed}")

print()
print(
    "Expected: 90 graphs"
)

