
from pathlib import Path
import pandas as pd

data_dir = Path(
    "/Users/gourav/Desktop/CAPSTONE PROJ/"
    "LogRCA-Research/data/"
)

files = [
    "graph_matching_rca_results.csv",
    "graph_current_event_rca_results.csv",
    "graph_historical_event_rca_results.csv",
    "candidate_evidence_rca_results.csv",
    "evidence_graph_matching_rca_results.csv",
    "partial_observation_rca_results.csv",
    "historical_family_holdout_rca_results.csv",
    "current_historical_family_holdout_rca_results.csv",
]

for filename in files:
    path = data_dir / filename

    if not path.exists():
        print(f"\nMISSING: {filename}")
        continue

    df = pd.read_csv(path)
    print(f"\n{'=' * 65}")
    print(filename)
    print("Rows:", len(df))

    # Summarize methods that have observation-level results
    if "observation_level" in df.columns:
        metrics = [
            col for col in
            ["top1", "top3", "top5", "mrr",
             "current_top1", "current_mrr",
             "historical_top1", "historical_mrr"]
            if col in df.columns
        ]

        if metrics:
            summary = df.groupby("observation_level")[metrics].mean()
            print(summary.round(4).to_string())

    # Check candidate counts for suspicious single-candidate results
    candidate_col = next(
        (col for col in
         ["num_candidates", "num_candidate_services"]
         if col in df.columns),
        None
    )

    if candidate_col:
        print("\nCandidate count distribution:")
        print(df[candidate_col].value_counts().sort_index().to_string())

    # Check whether a result is perfect despite a single candidate
    if candidate_col and "top1" in df.columns:
        suspicious = df[
            (df[candidate_col] == 1) & (df["top1"] == 1)
        ]
        print(
            "\nSingle-candidate Top-1 rows:",
            len(suspicious),
            "of", len(df)
        )

print("\nAudit complete.")
