import pandas as pd
from pathlib import Path


CASE = Path("data/re2ob_currencyservice_cpu_1")


# -----------------------------
# 1. Load logs
# -----------------------------

logs = pd.read_parquet(CASE / "logs.parquet")


# -----------------------------
# 2. Basic information
# -----------------------------

print("\n===== COLUMNS =====")
print(logs.columns.tolist())

print("\n===== SHAPE =====")
print(logs.shape)

print("\n===== DATA TYPES =====")
print(logs.dtypes)


# -----------------------------
# 3. Missing values
# -----------------------------

print("\n===== MISSING VALUES =====")
print(logs.isnull().sum())


# -----------------------------
# 4. Number of services
# -----------------------------

print("\n===== NUMBER OF SERVICES =====")
print(logs["container_name"].nunique())


print("\n===== SERVICES =====")
print(logs["container_name"].value_counts())


# -----------------------------
# 5. Timestamp information
# -----------------------------

print("\n===== TIMESTAMP RANGE =====")

print("Minimum timestamp:", logs["timestamp"].min())
print("Maximum timestamp:", logs["timestamp"].max())

print(
    "Total duration:",
    logs["timestamp"].max() - logs["timestamp"].min(),
    "seconds"
)


# -----------------------------
# 6. Duplicate timestamps
# -----------------------------

print("\n===== MOST COMMON TIMESTAMPS =====")

print(
    logs["timestamp"]
    .value_counts()
    .head(10)
)


# -----------------------------
# 7. Sample logs from each service
# -----------------------------

print("\n===== SAMPLE LOGS BY SERVICE =====")

for service in logs["container_name"].unique():

    print("\n--------------------------------")
    print("SERVICE:", service)
    print("--------------------------------")

    service_logs = logs[
        logs["container_name"] == service
    ]

    for _, row in service_logs.head(5).iterrows():

        print(
            row["timestamp"],
            "|",
            row["message"]
        )