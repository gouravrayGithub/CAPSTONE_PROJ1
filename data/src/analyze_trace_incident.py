import pandas as pd
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")
INJECT_TIME = 1705682817

traces = pd.read_parquet(CASE / "traces.parquet")

traces["timestamp"] = (
    traces["startTimeMillis"] // 1000
)

before = traces[
    (traces["timestamp"] >= INJECT_TIME - 600) &
    (traces["timestamp"] < INJECT_TIME)
].copy()

after = traces[
    (traces["timestamp"] >= INJECT_TIME) &
    (traces["timestamp"] <= INJECT_TIME + 600)
].copy()

print("===== TRACE INCIDENT ANALYSIS =====")

print("Before spans:", len(before))
print("After spans:", len(after))

# Duration is assumed to be microseconds in this dataset.
# Convert to milliseconds for readability.
before["duration_ms"] = before["duration"] / 1000
after["duration_ms"] = after["duration"] / 1000

before_stats = before.groupby("serviceName")["duration_ms"].agg(
    count="count",
    mean="mean",
    median="median",
    p95=lambda x: x.quantile(0.95)
)

after_stats = after.groupby("serviceName")["duration_ms"].agg(
    count="count",
    mean="mean",
    median="median",
    p95=lambda x: x.quantile(0.95)
)

comparison = before_stats.join(
    after_stats,
    lsuffix="_before",
    rsuffix="_after"
)

comparison["mean_change_%"] = (
    (comparison["mean_after"] - comparison["mean_before"])
    / comparison["mean_before"].replace(0, pd.NA)
) * 100

comparison["p95_change_%"] = (
    (comparison["p95_after"] - comparison["p95_before"])
    / comparison["p95_before"].replace(0, pd.NA)
) * 100

print("\n===== SERVICE LATENCY COMPARISON =====")
print(comparison.sort_values(
    "p95_change_%",
    ascending=False
).to_string())

print("\n===== STATUS CODE DISTRIBUTION =====")

print("\nBefore:")
print(
    before["statusCode"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAfter:")
print(
    after["statusCode"]
    .value_counts(dropna=False)
    .to_string()
)

comparison.to_parquet(
    CASE / "trace_incident_comparison.parquet"
)

print("\nSaved:")
print(CASE / "trace_incident_comparison.parquet")

