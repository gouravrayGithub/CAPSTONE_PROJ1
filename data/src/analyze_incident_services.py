import pandas as pd
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")

INJECT_TIME = 1705682817

events = pd.read_parquet(
    CASE / "events.parquet"
)

print("===== EVENT DATA =====")
print("Total events:", len(events))
print(
    "Time range:",
    events["timestamp"].min(),
    "to",
    events["timestamp"].max()
)

# --------------------------------------------------
# DEFINE WINDOWS
# --------------------------------------------------

# 10 minutes before injection
before_start = INJECT_TIME - 600
before_end = INJECT_TIME

# 10 minutes after injection
after_start = INJECT_TIME
after_end = INJECT_TIME + 600

before = events[
    (events["timestamp"] >= before_start)
    &
    (events["timestamp"] < before_end)
].copy()

after = events[
    (events["timestamp"] >= after_start)
    &
    (events["timestamp"] <= after_end)
].copy()


print("\n===== WINDOWS =====")

print(
    "Before:",
    before_start,
    "→",
    before_end
)

print(
    "After:",
    after_start,
    "→",
    after_end
)

print("\nBefore events:", len(before))
print("After events:", len(after))


# --------------------------------------------------
# SERVICE COUNTS
# --------------------------------------------------

before_service = (
    before["service"]
    .value_counts()
    .rename("before")
)

after_service = (
    after["service"]
    .value_counts()
    .rename("after")
)

service_comparison = pd.concat(
    [
        before_service,
        after_service
    ],
    axis=1
).fillna(0)


service_comparison["change"] = (
    service_comparison["after"]
    -
    service_comparison["before"]
)

service_comparison["percentage_change"] = (
    service_comparison["change"]
    /
    service_comparison["before"].replace(0, 1)
) * 100


service_comparison = service_comparison.sort_values(
    "change",
    ascending=False
)


print("\n===== SERVICE EVENT CHANGES =====")

print(
    service_comparison.to_string()
)


# --------------------------------------------------
# EVENT TYPE CHANGES
# --------------------------------------------------

before_events = (
    before["event_id"]
    .value_counts()
    .rename("before")
)

after_events = (
    after["event_id"]
    .value_counts()
    .rename("after")
)

event_comparison = pd.concat(
    [
        before_events,
        after_events
    ],
    axis=1
).fillna(0)

event_comparison["change"] = (
    event_comparison["after"]
    -
    event_comparison["before"]
)

event_comparison = event_comparison.sort_values(
    "change",
    ascending=False
)


print("\n===== TOP EVENT CHANGES =====")

print(
    event_comparison.head(20).to_string()
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

service_output = CASE / "service_incident_comparison.parquet"

service_comparison.to_parquet(
    service_output
)

print(
    "\nSaved:",
    service_output
)