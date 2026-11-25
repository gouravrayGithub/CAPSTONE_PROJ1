import pandas as pd
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")
INJECT_TIME = 1705682817

events = pd.read_parquet(CASE / "events.parquet")

# Inspect 60 seconds before and after injection
window = events[
    (events["timestamp"] >= INJECT_TIME - 60) &
    (events["timestamp"] <= INJECT_TIME + 60)
].copy()

currency = window[
    window["service"] == "currencyservice"
].copy()

print("===== CURRENCYSERVICE INCIDENT WINDOW =====")
print("Injection:", INJECT_TIME)
print("Events:", len(currency))

print("\n===== EVENT COUNTS =====")
print(
    currency["event_id"]
    .value_counts()
    .to_string()
)

print("\n===== EVENT TIMELINE =====")

cols = [
    "timestamp",
    "event_id",
    "template"
]

print(
    currency[cols]
    .sort_values("timestamp")
    .to_string(index=False)
)

print("\n===== UNIQUE TEMPLATES =====")

print(
    currency[
        ["event_id", "template"]
    ]
    .drop_duplicates()
    .sort_values("event_id")
    .to_string(index=False)
)