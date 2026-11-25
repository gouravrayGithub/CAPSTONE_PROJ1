import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

CASE = Path("data/re2ob_currencyservice_cpu_1")

INPUT_FILE = CASE / "parsed_logs.parquet"
OUTPUT_FILE = CASE / "events.parquet"


# --------------------------------------------------
# 2. Load parsed logs
# --------------------------------------------------

logs = pd.read_parquet(INPUT_FILE)


# --------------------------------------------------
# 3. Sort chronologically
# --------------------------------------------------

logs = logs.sort_values(
    ["timestamp"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Create event instance ID
# --------------------------------------------------

logs["event_instance_id"] = range(
    1,
    len(logs) + 1
)


# --------------------------------------------------
# 5. Select event representation
# --------------------------------------------------

events = logs[
    [
        "event_instance_id",
        "timestamp",
        "container_name",
        "event_id",
        "template",
        "message"
    ]
].copy()


# --------------------------------------------------
# 6. Rename service column
# --------------------------------------------------

events = events.rename(
    columns={
        "container_name": "service"
    }
)


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

events.to_parquet(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 8. Print information
# --------------------------------------------------

print("\n===== EVENT REPRESENTATION =====")

print(
    "Total event instances:",
    len(events)
)

print(
    "Unique event types:",
    events["event_id"].nunique()
)

print(
    "Unique services:",
    events["service"].nunique()
)


print("\n===== EVENT SAMPLE =====")

print(
    events.head(20).to_string(index=False)
)