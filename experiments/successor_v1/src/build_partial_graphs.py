
from pathlib import Path
import pickle
import random
import networkx as nx

BASE_DIR = Path("experiments/successor_v1/graphs")

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
RANDOM_SEED = 42

# Initially preserve the old experiment's root-hidden protocol.
# We will explicitly address its evaluation limitation before trusting scores.
HIDE_ROOT = True


def get_root_service(case_name):
    # Example: re2ob_currencyservice_cpu_1
    parts = case_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected case name: {case_name}")
    return parts[1]


def build_partial_graph(graph, observation_level, root_service, seed):
    nodes = list(graph.nodes())

    # Preserve the original protocol: hide the known root service.
    available_nodes = [
        node for node in nodes
        if not (HIDE_ROOT and node == root_service)
    ]

    rng = random.Random(seed)
    rng.shuffle(available_nodes)

    target_count = round(
        len(nodes) * observation_level / 100
    )

    # We cannot observe more nodes than remain available.
    target_count = min(target_count, len(available_nodes))
    selected_nodes = set(available_nodes[:target_count])

    # Induced subgraph keeps only edges between selected services.
    partial = graph.subgraph(selected_nodes).copy()

    partial.graph.update(graph.graph)
    partial.graph["observation_level_requested"] = observation_level
    partial.graph["observed_nodes"] = partial.number_of_nodes()
    partial.graph["original_nodes"] = graph.number_of_nodes()
    partial.graph["hidden_root"] = HIDE_ROOT

    return partial


def main():
    cases = sorted(
        path for path in BASE_DIR.iterdir()
        if path.is_dir()
        and (path / "historical_incident_graph.gpickle").exists()
    )

    print("Cases found:", len(cases))

    created = 0

    for case_dir in cases:
        case_name = case_dir.name
        root_service = get_root_service(case_name)

        graph_path = case_dir / "historical_incident_graph.gpickle"

        with graph_path.open("rb") as file:
            graph = pickle.load(file)

        for level in OBSERVATION_LEVELS:
            partial = build_partial_graph(
                graph=graph,
                observation_level=level,
                root_service=root_service,
                seed=RANDOM_SEED,
            )

            output_dir = case_dir / "partial_graphs"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"partial_{level}.gpickle"

            with output_path.open("wb") as file:
                pickle.dump(partial, file)

            created += 1

        print(f"Created partial graphs: {case_name}")

    print("\n========== SUMMARY ==========")
    print("Cases:", len(cases))
    print("Observation levels:", OBSERVATION_LEVELS)
    print("Partial graphs created:", created)
    print("Output: experiments/successor_v1/graphs/<case>/partial_graphs/")


if __name__ == "__main__":
    main()
