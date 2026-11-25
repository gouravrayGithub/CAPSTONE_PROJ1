from pathlib import Path
import pickle


CASE = "re2ob_checkoutservice_cpu_1"

BASE = (
    Path("data/partial_incident_graphs")
    / CASE
)


graphs = {}

for level in [100, 80, 60, 40, 20]:

    path = (
        BASE /
        f"partial_{level}.gpickle"
    )

    with open(path, "rb") as f:

        graphs[level] = pickle.load(f)


print("=" * 70)
print("NESTED PARTIAL GRAPH CHECK")
print("=" * 70)

previous_nodes = None


for level in [100, 80, 60, 40, 20]:

    G = graphs[level]

    nodes = set(G.nodes())

    print()
    print(
        f"{level}% | "
        f"nodes={len(nodes)} | "
        f"edges={G.number_of_edges()}"
    )

    print(
        "Nodes:",
        sorted(nodes)
    )

    if previous_nodes is not None:

        print(
            "Subset of previous:",
            nodes.issubset(previous_nodes)
        )

    previous_nodes = nodes