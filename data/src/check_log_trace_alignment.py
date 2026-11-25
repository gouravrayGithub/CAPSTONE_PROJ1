import pandas as pd
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")

logs_path = CASE / "logs.parquet"
traces_path = CASE / "traces.parquet"

logs = pd.read_parquet(logs_path)
traces = pd.read_parquet(traces_path)

print("===== LOG DATA =====")
print("Logs:", len(logs))
print("Log time range:", logs["timestamp"].min(), "to", logs["timestamp"].max())

print("\n===== TRACE DATA =====")
print("Spans:", len(traces))
print("Trace startTimeMillis:",
      traces["startTimeMillis"].min(),
      "to",
      traces["startTimeMillis"].max())

# Convert trace milliseconds to seconds
traces["trace_seconds"] = traces["startTimeMillis"] // 1000

print("\n===== TIME RANGE COMPARISON =====")
print(
    "Log range:",
    logs["timestamp"].min(),
    "→",
    logs["timestamp"].max()
)

print(
    "Trace range:",
    traces["trace_seconds"].min(),
    "→",
    traces["trace_seconds"].max()
)

# Pick a few interesting log events
keywords = [
    "GetCart",
    "ListRecommendations",
    "ad request",
    "PlaceOrder",
    "GetQuote",
]

print("\n===== SAMPLE LOG EVENTS =====")

for keyword in keywords:

    matches = logs[
        logs["message"]
        .str.contains(keyword, case=False, na=False)
    ]

    print(f"\n--- {keyword} ---")

    if len(matches) == 0:
        print("No matching logs")
        continue

    sample = matches[
        ["timestamp", "container_name", "message"]
    ].head(5)

    print(sample.to_string(index=False))


print("\n===== SAMPLE TRACE OPERATIONS =====")

operations = [
    "GetCart",
    "ListRecommendations",
    "GetAds",
    "PlaceOrder",
    "GetQuote",
]

for operation in operations:

    matches = traces[
        traces["operationName"]
        .astype(str)
        .str.contains(operation, case=False, na=False)
    ]

    print(f"\n--- {operation} ---")

    if len(matches) == 0:
        print("No matching traces")
        continue

    sample = matches[
        [
            "traceID",
            "spanID",
            "serviceName",
            "operationName",
            "parentSpanID",
            "trace_seconds",
        ]
    ].head(5)

    print(sample.to_string(index=False))