import pandas as pd
from pathlib import Path


CASE = Path("data/re2ob_currencyservice_cpu_1")


# Load logs
logs = pd.read_parquet(CASE / "logs.parquet")

print("\n===== LOG DATA =====")
print(logs.head(20))

print("\n===== COLUMNS =====")
print(logs.columns.tolist())

print("\n===== SHAPE =====")
print(logs.shape)

print("\n===== DATA TYPES =====")
print(logs.dtypes)

print("\n===== UNIQUE CONTAINERS =====")
print(logs["container_name"].unique())

print("\n===== FIRST LOG MESSAGE =====")
print(logs.iloc[0]["message"])
print("\n===== FIRST 50 LOGS =====")

for _, row in logs.head(50).iterrows():
    print(
        row["timestamp"],
        "|",
        row["container_name"],
        "|",
        row["message"]
    )