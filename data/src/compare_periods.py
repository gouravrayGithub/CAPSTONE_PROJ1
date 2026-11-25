import pandas as pd
from pathlib import Path


CASE = Path("data/re2ob_currencyservice_cpu_1")


normal = pd.read_parquet(
    CASE / "normal_events.parquet"
)

incident = pd.read_parquet(
    CASE / "incident_events.parquet"
)


# --------------------------------------------------
# Count event types
# --------------------------------------------------

normal_counts = (
    normal
    .groupby(["service", "event_id"])
    .size()
    .reset_index(name="normal_count")
)


incident_counts = (
    incident
    .groupby(["service", "event_id"])
    .size()
    .reset_index(name="incident_count")
)


# --------------------------------------------------
# Combine
# --------------------------------------------------

comparison = pd.merge(
    normal_counts,
    incident_counts,
    on=["service", "event_id"],
    how="outer"
).fillna(0)


# --------------------------------------------------
# Calculate change
# --------------------------------------------------

comparison["change"] = (
    comparison["incident_count"]
    -
    comparison["normal_count"]
)


comparison["ratio"] = (
    comparison["incident_count"] + 1
) / (
    comparison["normal_count"] + 1
)


# --------------------------------------------------
# Sort
# --------------------------------------------------

comparison = comparison.sort_values(
    "change",
    ascending=False
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print("\n===== MOST INCREASED EVENTS =====")

print(
    comparison.head(30).to_string(index=False)
)