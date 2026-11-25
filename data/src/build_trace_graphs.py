import pandas as pd
from pathlib import Path
from collections import defaultdict

CASE = Path("data/re2ob_currencyservice_cpu_1")

traces = pd.read_parquet(
    CASE / "traces.parquet"
)

# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------

traces["spanID"] = traces["spanID"].astype(str)
traces["parentSpanID"] = traces["parentSpanID"].astype(str)
traces["traceID"] = traces["traceID"].astype(str)

print("===== TRACE GRAPH CONSTRUCTION =====")

print("Total spans:", len(traces))
print("Unique traces:", traces["traceID"].nunique())


# --------------------------------------------------
# BUILD ONE TRACE TREE
# --------------------------------------------------

def build_trace_tree(trace_df):

    spans = {}

    for _, row in trace_df.iterrows():

        span_id = row["spanID"]

        spans[span_id] = {
            "service": row["serviceName"],
            "operation": row["operationName"],
            "parent": row["parentSpanID"],
            "start": row["startTimeMillis"],
            "duration": row["duration"],
        }

    children = defaultdict(list)

    roots = []

    for span_id, span in spans.items():

        parent = span["parent"]

        if (
            parent == "nan"
            or parent == ""
            or parent not in spans
        ):
            roots.append(span_id)

        else:
            children[parent].append(span_id)

    return spans, children, roots


# --------------------------------------------------
# SELECT ONE TRACE FOR INSPECTION
# --------------------------------------------------

first_trace_id = traces["traceID"].iloc[0]

trace_df = traces[
    traces["traceID"] == first_trace_id
].copy()

spans, children, roots = build_trace_tree(trace_df)


print("\n===== SAMPLE TRACE =====")

print("Trace ID:", first_trace_id)

print("Spans:", len(spans))

print("Root spans:", len(roots))


# --------------------------------------------------
# PRINT TREE
# --------------------------------------------------

def print_tree(
    span_id,
    spans,
    children,
    level=0
):

    span = spans[span_id]

    indent = "  " * level

    print(
        indent
        + f"├─ {span['service']} | "
        + f"{span['operation']}"
    )

    for child in children.get(span_id, []):

        print_tree(
            child,
            spans,
            children,
            level + 1
        )


for root in roots:

    print_tree(
        root,
        spans,
        children
    )


# --------------------------------------------------
# BUILD SERVICE EDGES FOR THIS TRACE
# --------------------------------------------------

service_edges = []

for span_id, span in spans.items():

    parent_id = span["parent"]

    if parent_id not in spans:
        continue

    parent = spans[parent_id]

    parent_service = parent["service"]
    child_service = span["service"]

    if parent_service == child_service:
        continue

    service_edges.append(
        (
            parent_service,
            child_service
        )
    )


print("\n===== SERVICE EDGES IN SAMPLE TRACE =====")

for parent, child in service_edges:

    print(
        f"{parent} -> {child}"
    )