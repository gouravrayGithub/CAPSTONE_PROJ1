
from pathlib import Path
import pickle
from collections import deque

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [100, 80, 60, 40, 20]
MAX_HOPS = 3

OUTPUT_FILE = DATA_DIR / "graph_direction_diagnostic.csv"


# ============================================================
# HELPERS
# ============================================================

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_case(case):
    parts = case.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected case name: {case}")

    return parts[1], parts[2]


def get_neighbors(graph, node, direction):
    if direction == "predecessors":
        return graph.predecessors(node)

    if direction == "successors":
        return graph.successors(node)

    if direction == "undirected":
        return set(graph.predecessors(node)) | set(
            graph.successors(node)
        )

    raise ValueError(f"Unknown direction: {direction}")


def find_candidates(
    current_graph,
    historical_graph,
    max_hops,
    direction,
):
    """
    Return unobserved services reachable by traversing
    up to max_hops in the chosen direction.
    """

    observed = set(current_graph.nodes())
    candidates = set()

    queue = deque(
        (node, 0)
        for node in observed
        if node in historical_graph
    )

    best_distance = {
        node: 0 for node, _ in queue
    }

    while queue:
        node, distance = queue.popleft()

        if distance >= max_hops:
            continue

        for neighbor in get_neighbors(
            historical_graph, node, direction
        ):
            new_distance = distance + 1

            if neighbor not in observed:
                candidates.add(neighbor)

            if (
                neighbor not in best_distance
                or new_distance < best_distance[neighbor]
            ):
                best_distance[neighbor] = new_distance
                queue.append((neighbor, new_distance))

    return candidates


# ============================================================
# LOAD CASES AND COMPLETE HISTORICAL GRAPHS
# ============================================================

case_dirs = sorted(
    p for p in PARTIAL_DIR.iterdir() if p.is_dir()
)

if not case_dirs:
    raise RuntimeError(
        f"No cases found in {PARTIAL_DIR}"
    )

historical_graphs = {}

for case_dir in case_dirs:
    case = case_dir.name
    graph_path = (
        DATA_DIR
        / case
        / "historical_incident_graph.gpickle"
    )

    historical_graphs[case] = load_graph(graph_path)

print("Cases loaded:", len(case_dirs))
print("Historical graphs loaded:", len(historical_graphs))


# ============================================================
# DIAGNOSE CANDIDATE COVERAGE BY EDGE DIRECTION
# ============================================================

directions = [
    "predecessors",
    "successors",
    "undirected",
]

results = []

for level in OBSERVATION_LEVELS:
    print(f"\nObservation level: {level}%")

    for direction in directions:
        root_found_count = 0
        total_candidate_count = 0
        no_candidate_count = 0

        for case_dir in case_dirs:
            case = case_dir.name
            root_service, fault_type = parse_case(case)

            partial_path = (
                case_dir / f"partial_{level}.gpickle"
            )
            current_graph = load_graph(partial_path)

            all_candidates = set()

            # Search the same historical case bank as V2.
            for other_case_dir in case_dirs:
                other_case = other_case_dir.name

                if other_case == case:
                    continue

                other_root, other_fault = parse_case(
                    other_case
                )

                # Preserve the existing family exclusion.
                if (
                    other_root == root_service
                    and other_fault == fault_type
                ):
                    continue

                historical_graph = historical_graphs[
                    other_case
                ]

                candidates = find_candidates(
                    current_graph,
                    historical_graph,
                    MAX_HOPS,
                    direction,
                )

                all_candidates.update(candidates)

            total_candidate_count += len(all_candidates)

            if not all_candidates:
                no_candidate_count += 1

            if root_service in all_candidates:
                root_found_count += 1

        n = len(case_dirs)

        result = {
            "observation_level": level,
            "direction": direction,
            "cases": n,
            "root_candidate_coverage": root_found_count / n,
            "no_candidate_rate": no_candidate_count / n,
            "average_candidate_count": total_candidate_count / n,
        }

        results.append(result)

        print(
            f"{direction:12s} | "
            f"root coverage={root_found_count / n:.3f} | "
            f"no candidates={no_candidate_count / n:.3f} | "
            f"avg candidates={total_candidate_count / n:.2f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

print("\nDiagnostic results saved to:")
print(OUTPUT_FILE)
print("\n", df.to_string(index=False))
