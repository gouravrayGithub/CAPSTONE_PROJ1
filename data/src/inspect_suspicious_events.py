import pandas as pd
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")
INJECT_TIME = 1705682817

events = pd.read_parquet(CASE / "events.parquet")

before_start = INJECT_TIME - 600
before_end = INJECT_TIME

after_start = INJECT_TIME
after_end = INJECT_TIME + 600

before = events[
    (events["timestamp"] >= before_start) &
    (events["timestamp"] < before_end)
]

after = events[
    (events["timestamp"] >= after_start) &
    (events["timestamp"] <= after_end)
]

before_counts = before["event_id"].value_counts().rename("before")
after_counts = after["event_id"].value_counts().rename("after")

comparison = pd.concat(
    [before_counts, after_counts],
    axis=1
).fillna(0)

comparison["change"] = (
    comparison["after"] - comparison["before"]
)

comparison["relative_change"] = (
    comparison["after"] + 1
) / (
    comparison["before"] + 1
)

# Largest relative increases
suspicious = comparison.sort_values(
    "relative_change",
    ascending=False
)

print("\n===== MOST INCREASED EVENT TYPES =====")
print(suspicious.head(20).to_string())

# Event IDs that did not exist before but appeared after
new_events = comparison[
    (comparison["before"] == 0) &
    (comparison["after"] > 0)
]

print("\n===== NEW EVENT TYPES AFTER INJECTION =====")
print(new_events.sort_values(
    "after",
    ascending=False
).head(20).to_string())

# Show actual templates/messages
interesting_ids = suspicious.head(10).index.tolist()

print("\n===== ACTUAL EVENT TEMPLATES =====")

for event_id in interesting_ids:

    rows = events[
        events["event_id"] == event_id
    ]

    print(f"\n--- EVENT ID {event_id} ---")

    print(
        rows[
            ["service", "template"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

print("\n===== DONE =====")