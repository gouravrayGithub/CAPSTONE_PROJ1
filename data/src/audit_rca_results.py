
from pathlib import Path
import pandas as pd

data_dir = Path(
    "/Users/gourav/Desktop/CAPSTONE PROJ/"
    "LogRCA-Research/data/"
)

def load(name):
    path = data_dir / name
    if not path.exists():
        print(f"MISSING: {path}")
        return None
    return pd.read_csv(path)

# 1. Graph candidate audit
df = load("graph_matching_rca_results.csv")

if df is not None:
    print("\n=== GRAPH CANDIDATE AUDIT ===")

    for level, group in df.groupby("observation_level"):
        n = len(group)
        valid = group[group["num_candidate_services"] > 0]

        print(f"\nObservation: {level}%")
        print("Rows:", n)
        print("Cases:", group["case"].nunique())
        print("Rows without candidates:",
              (group["num_candidate_services"] == 0).sum())
        print("Rows with candidates:", len(valid))
        print("Top-1, all rows:", round(group["top1"].mean(), 4))
        print("Top-1, candidate rows only:",
              round(valid["top1"].mean(), 4)
              if len(valid) else "N/A")
        print("Top-3, candidate rows only:",
              round(valid["top3"].mean(), 4)
              if len(valid) else "N/A")
        print("MRR, candidate rows only:",
              round(valid["mrr"].mean(), 4)
              if len(valid) else "N/A")

# 2. Partial-observation audit
df = load("partial_observation_rca_results.csv")

if df is not None:
    print("\n=== PARTIAL OBSERVATION AUDIT ===")
    metrics = [
        c for c in [
            "current_top1", "current_top3", "current_mrr",
            "historical_top1", "historical_top3", "historical_mrr",
            "n_observed_services"
        ] if c in df.columns
    ]

    print("Rows:", len(df))
    print("Unique cases:", df["case"].nunique())
    print(df.groupby("observation_level")[metrics].mean().round(4))

    print("\nRows per observation level:")
    print(df.groupby("observation_level").size())

# 3. Family holdout metrics
for filename in [
    "historical_family_holdout_rca_results.csv",
    "current_historical_family_holdout_rca_results.csv",
]:
    df = load(filename)
    if df is None:
        continue

    print(f"\n=== {filename} ===")
    print("Rows:", len(df))
    print("Unique cases:", df["case"].nunique())

    metrics = [
        c for c in [
            "top1_correct", "top3_correct", "top5_correct",
            "reciprocal_rank", "top1", "top3", "top5",
            "rank", "current_root_score",
            "historical_root_score", "combined_root_score"
        ] if c in df.columns
    ]

    print(df[metrics].mean(numeric_only=True).round(4))
