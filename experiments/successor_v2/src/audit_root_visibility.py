
from pathlib import Path
import csv
import pickle
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DIR = PROJECT_ROOT / "experiments" / "successor_v2" / "graphs"
RESULT_FILE = (
    PROJECT_ROOT
    / "experiments" / "successor_v2" / "results"
    / "rca_methods_v3_detailed.csv"
)

groups = defaultdict(list)

with open(RESULT_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    level = int(row["observation_level"])
    case_name = row["case"]
    method = row["method"]
    true_root = row["true_root"]

    partial_path = (
        GRAPH_DIR / case_name / "partial_graphs"
        / f"partial_{level}.gpickle"
    )

    with open(partial_path, "rb") as f:
        graph = pickle.load(f)

    visibility = (
        "root_visible"
        if true_root in graph.nodes
        else "root_hidden"
    )

    groups[(level, method, visibility)].append(row)

print("\n===== ROOT VISIBILITY AUDIT =====")

for level in [20, 40, 60, 80, 100]:
    print(f"\n--- Observation: {level}% ---")

    for visibility in ["root_visible", "root_hidden"]:
        for method in [
            "structural",
            "successor",
            "successor_anomaly",
            "ancestor_anomaly",
        ]:
            subset = groups.get((level, method, visibility), [])

            if not subset:
                continue

            n = len(subset)
            top1 = sum(int(r["top1"]) for r in subset) / n
            top3 = sum(int(r["top3"]) for r in subset) / n
            mrr = sum(float(r["rr"]) for r in subset) / n

            print(
                f"{visibility:12} | {method:18} | "
                f"n={n:2} | Top1={top1:.3f} | "
                f"Top3={top3:.3f} | MRR={mrr:.3f}"
            )

