import pandas as pd
from pathlib import Path


CASE = Path("data/re2ob_currencyservice_cpu_1")


# --------------------------------------------------
# Load traces
# --------------------------------------------------

traces = pd.read_parquet(
    CASE / "traces.parquet"
)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n===== TRACE COLUMNS =====")

print(
    traces.columns.tolist()
)


print("\n===== TRACE SHAPE =====")

print(
    traces.shape
)


print("\n===== TRACE DATA TYPES =====")

print(
    traces.dtypes
)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n===== MISSING VALUES =====")

print(
    traces.isnull().sum()
)


# --------------------------------------------------
# Sample
# --------------------------------------------------

print("\n===== FIRST 30 TRACE RECORDS =====")

print(
    traces.head(30).to_string(index=False)
)