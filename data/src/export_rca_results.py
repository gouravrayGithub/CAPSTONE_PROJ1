
from pathlib import Path
import pandas as pd

data_dir = Path("/Users/gourav/Desktop/CAPSTONE PROJ/LogRCA-Research/data/")

files = [
    "current_only_rca_results.parquet",
    "historical_rca_results.parquet",
    "graph_matching_rca_results.parquet",
    "graph_current_event_rca_results.parquet",
    "graph_historical_current_rca_results.parquet",
    "graph_historical_event_rca_results.parquet",
    "evidence_graph_matching_rca_results.parquet",
    "candidate_evidence_rca_results.parquet",
    "graph_boost_current_results.parquet",
    "partial_observation_rca_results.parquet",
    "historical_family_holdout_rca_results.parquet",
    "current_historical_family_holdout_rca_results.parquet",
]

for filename in files:
    path = data_dir / filename

    if not path.exists():
        print(f"\nMISSING: {filename}")
        continue

    df = pd.read_parquet(path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

    df.to_csv(
        data_dir / filename.replace(".parquet", ".csv"),
        index=False
    )

    print("CSV exported.")
