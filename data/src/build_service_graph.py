import pandas as pd
import networkx as nx
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

CASE = Path("data/re2ob_currencyservice_cpu_1")

TRACE_FILE = CASE / "traces.parquet"


# --------------------------------------------------
# 2. Load traces
# --------------------------------------------------

traces = pd.read_parquet(TRACE_FILE)


print("\n===== TRACE DATA =====")

print("Total spans:", len(traces))

print(
    "Unique traces:",
    traces["traceID"].nunique()
)

print(
    "Unique services:",
    traces["serviceName"].nunique()
)


# --------------------------------------------------
# 3. Remove invalid parent spans
# --------------------------------------------------

traces = traces.copy()

traces["parentSpanID"] = (
    traces["parentSpanID"]
    .astype("string")
)


# --------------------------------------------------
# 4. Create span lookup
# --------------------------------------------------

span_lookup = (
    traces
    .set_index("spanID")
    .to_dict("index")
)


# --------------------------------------------------
# 5. Create directed graph
# --------------------------------------------------

G = nx.DiGraph()


# --------------------------------------------------
# 6. Add services as nodes
# --------------------------------------------------

services = (
    traces["serviceName"]
    .dropna()
    .unique()
)

for service in services:

    G.add_node(service)


# --------------------------------------------------
# 7. Create dependency edges
# --------------------------------------------------

for _, span in traces.iterrows():

    parent_id = span["parentSpanID"]

    child_service = span["serviceName"]


    # No parent
    if pd.isna(parent_id):
        continue


    # Parent does not exist
    if parent_id not in span_lookup:
        continue


    parent_span = span_lookup[parent_id]

    parent_service = parent_span["serviceName"]


    # Ignore same-service parent-child relation
    if parent_service == child_service:
        continue


    # Add / update dependency
    if G.has_edge(parent_service, child_service):

        G[parent_service][child_service]["count"] += 1

    else:

        G.add_edge(
            parent_service,
            child_service,
            count=1
        )


# --------------------------------------------------
# 8. Print graph
# --------------------------------------------------

print("\n===== SERVICE DEPENDENCY GRAPH =====")

print("Nodes:", G.number_of_nodes())

print("Edges:", G.number_of_edges())


print("\n===== DEPENDENCIES =====")

edges = sorted(
    G.edges(data=True),
    key=lambda x: x[2]["count"],
    reverse=True
)


for source, target, data in edges:

    print(
        source,
        "->",
        target,
        "| calls:",
        data["count"]
    )


# --------------------------------------------------
# 9. Save graph
# --------------------------------------------------

nx.write_gpickle = getattr(
    nx,
    "write_gpickle",
    None
)

if nx.write_gpickle:

    nx.write_gpickle(
        G,
        CASE / "service_graph.gpickle"
    )

print("\nService graph created.")