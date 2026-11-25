import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

CASE = Path("data/re2ob_currencyservice_cpu_1")
INJECT_TIME = 1705682817

traces = pd.read_parquet(CASE / "traces.parquet")
events = pd.read_parquet(CASE / "events.parquet")

# --------------------------------------------------
# 1. Define before / after windows
# --------------------------------------------------

before_start = INJECT_TIME - 600
before_end = INJECT_TIME

after_start = INJECT_TIME
after_end = INJECT_TIME + 600

# --------------------------------------------------
# 2. Event-level evidence
# --------------------------------------------------

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
    (event_stats["event_count_after"]
     - event_stats["event_count_before"])
    / event_stats["event_count_before"].replace(0, 1)
) * 100

# --------------------------------------------------
# 3. Trace-level evidence
# --------------------------------------------------

traces["timestamp"] = traces["startTimeMillis"] // 1000

before_traces = traces[
    (traces["timestamp"] >= before_start) &
    (traces["timestamp"] < before_end)
].copy()

after_traces = traces[
    (traces["timestamp"] >= after_start) &
    (traces["timestamp"] <= after_end)
].copy()

# Keep duration in the original dataset unit.
before_traces["duration_raw"] = before_traces["duration"]
after_traces["duration_raw"] = after_traces["duration"]

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

# --------------------------------------------------
# 4. Merge evidence
# --------------------------------------------------

service_stats = event_stats.copy()

# Convert trace service names to normal service names
trace_stats.index = trace_stats.index.map(
    lambda x: "frontend" if x == "frontendservice" else x
)

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

print("\n===== SERVICE EVIDENCE =====")
print(service_stats.to_string())

# --------------------------------------------------
# 5. Build observed service dependency graph
# --------------------------------------------------

G = nx.DiGraph()

# Add service nodes
for service in service_stats.index:
    G.add_node(
        service,
        event_count_before=float(
            service_stats.loc[service, "event_count_before"]
        ),
        event_count_after=float(
            service_stats.loc[service, "event_count_after"]
        ),
        event_change_pct=float(
            service_stats.loc[service, "event_change_pct"]
        ),
        duration_mean_before=float(
            service_stats.loc[service, "duration_mean_before"]
        ),
        duration_mean_after=float(
            service_stats.loc[service, "duration_mean_after"]
        ),
        duration_p95_before=float(
            service_stats.loc[service, "duration_p95_before"]
        ),
        duration_p95_after=float(
            service_stats.loc[service, "duration_p95_after"]
        ),
        duration_mean_change_pct=float(
            service_stats.loc[service, "duration_mean_change_pct"]
        ),
        duration_p95_change_pct=float(
            service_stats.loc[service, "duration_p95_change_pct"]
        )
    )

# --------------------------------------------------
# 6. Extract parent-child service relationships
# --------------------------------------------------

# Only use spans from the incident window
incident_traces = traces[
    (traces["timestamp"] >= before_start) &
    (traces["timestamp"] <= after_end)
].copy()

# Create span lookup
span_service = (
    incident_traces
    .set_index("spanID")["serviceName"]
    .to_dict()
)

edge_counts = {}

for _, row in incident_traces.iterrows():

    parent = row["parentSpanID"]
    child_service = row["serviceName"]

    if pd.isna(parent) or parent == "":
        continue

    parent_service = span_service.get(parent)

    if parent_service is None:
        continue

    if parent_service == child_service:
        continue

    if parent_service == "frontendservice":
        parent_service = "frontend"

    key = (parent_service, child_service)

    edge_counts[key] = edge_counts.get(key, 0) + 1

# Add edges
for (source, target), count in edge_counts.items():

    G.add_edge(
        source,
        target,
        observed_calls=count
    )

# --------------------------------------------------
# 7. Print graph
# --------------------------------------------------

print("\n===== CURRENT INCIDENT GRAPH =====")

print("\nNodes:")
for node, attrs in G.nodes(data=True):

    print(
        f"{node}: "
        f"events {attrs['event_count_before']:.0f}"
        f" → {attrs['event_count_after']:.0f}, "
        f"event change {attrs['event_change_pct']:.2f}%, "
        f"mean duration change "
        f"{attrs['duration_mean_change_pct']:.2f}%, "
        f"P95 duration change "
        f"{attrs['duration_p95_change_pct']:.2f}%"
    )

print("\nEdges:")

for source, target, attrs in G.edges(data=True):

    print(
        f"{source} -> {target} "
        f"({attrs['observed_calls']} observed calls)"
    )

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

output = CASE / "current_incident_graph.gpickle"

with open(output, "wb") as f:
    pickle.dump(G, f)

print("\nSaved:")
print(output)