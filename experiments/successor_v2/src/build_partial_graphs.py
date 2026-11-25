
from pathlib import Path
import hashlib
import random
import pickle

import networkx as nx


# Project root: LogRCA-Research
PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_DIR = PROJECT_ROOT / "experiments" / "successor_v2" / "graphs"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
RANDOM_SEED = 42


def stable_seed(case_name, observation_level):
    """Create a reproducible seed without using the root-cause label."""
    value = f"{RANDOM_SEED}|{case_name}|{observation_level}"
    digest = hashlib.sha256(value.encode()).hexdigest()
    return int(digest[:8], 16)


def main():
    total_cases = 0
    total_created = 0
    failures = []

    for case_dir in sorted(GRAPH_DIR.iterdir()):
        if not case_dir.is_dir():
            continue

        graph_file = case_dir / "historical_incident_graph.gpickle"

        if not graph_file.exists():
            continue

        try:
            with open(graph_file, 'rb') as f:
                full_graph = pickle.load(f)
            all_nodes = list(full_graph.nodes())

            if not all_nodes:
                raise ValueError("Graph has no nodes")

            partial_dir = case_dir / "partial_graphs"
            partial_dir.mkdir(parents=True, exist_ok=True)

            total_cases += 1

            for level in OBSERVATION_LEVELS:
                rng = random.Random(stable_seed(case_dir.name, level))

                visible_count = round(
                    len(all_nodes) * level / 100
                )
                visible_count = max(
                    1, min(visible_count, len(all_nodes))
                )

                # Crucially, this sampling does not use the true root.
                visible_nodes = rng.sample(all_nodes, visible_count)

                partial_graph = full_graph.subgraph(
                    visible_nodes
                ).copy()

                # Store experiment metadata for later auditing.
                partial_graph.graph["case_name"] = case_dir.name
                partial_graph.graph["observation_level"] = level
                partial_graph.graph["random_seed"] = RANDOM_SEED
                partial_graph.graph["visible_count"] = len(visible_nodes)
                partial_graph.graph["total_node_count"] = len(all_nodes)

                output_file = (
                    partial_dir / f"partial_{level}.gpickle"
                )
                with open(output_file, 'wb') as f:
                    pickle.dump(partial_graph, f)

                total_created += 1

        except Exception as exc:
            failures.append((case_dir.name, str(exc)))

    print("\n===== V2 PARTIAL GRAPH BUILD COMPLETE =====")
    print(f"Cases processed: {total_cases}")
    print(f"Partial graphs created: {total_created}")
    print(f"Failures: {len(failures)}")

    if failures:
        for case_name, error in failures:
            print(f"FAILED: {case_name}: {error}")

    print("\nObservation levels:", OBSERVATION_LEVELS)
    print("Root label used for masking: NO")
    print("Random seed:", RANDOM_SEED)


if __name__ == "__main__":
    main()
