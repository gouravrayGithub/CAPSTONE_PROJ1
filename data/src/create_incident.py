import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

CASE = Path("data/re2ob_currencyservice_cpu_1")

EVENT_FILE = CASE / "events.parquet"


# --------------------------------------------------
# 2. Load events
# --------------------------------------------------

events = pd.read_parquet(EVENT_FILE)


# --------------------------------------------------
# 3. Read injection time
# --------------------------------------------------

with open(CASE / "inject_time.txt", "r") as f:
    inject_time = int(f.read().strip())


print("Injection time:", inject_time)


# --------------------------------------------------
# 4. Split
# --------------------------------------------------

normal_events = events[
    events["timestamp"] < inject_time
].copy()


incident_events = events[
    events["timestamp"] >= inject_time
].copy()


# --------------------------------------------------
# 5. Save
# --------------------------------------------------

normal_events.to_parquet(
    CASE / "normal_events.parquet",
    index=False
)

incident_events.to_parquet(
    CASE / "incident_events.parquet",
    index=False
)


# --------------------------------------------------
# 6. Statistics
# --------------------------------------------------

print("\n===== NORMAL =====")

print(
    "Events:",
    len(normal_events)
)

print(
    "Unique event types:",
    normal_events["event_id"].nunique()
)


print("\n===== INCIDENT =====")

print(
    "Events:",
    len(incident_events)
)

print(
    "Unique event types:",
    incident_events["event_id"].nunique()
)


# --------------------------------------------------
# 7. Incident services
# --------------------------------------------------

print("\n===== INCIDENT SERVICES =====")

print(
    incident_events["service"]
    .value_counts()
)