from pathlib import Path
import pickle
import random


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

OBSERVATION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2]

RANDOM_SEED = 42

# Stage B:
# The true root service is hidden at every observation level.
HIDE_ROOT = True


# ============================================================
# LOAD HISTORICAL GRAPHS
# ============================================================

graph_files = sorted(
    DATA_DIR.glob(
        "re2ob_*/historical_incident_graph.gpickle"
    )
)

print("Historical graphs found:", len(graph_files))

graphs = {}

for path in graph_files:

    with open(path, "rb") as f:
        G = pickle.load(f)

    case = path.parent.name

    graphs[case] = G


# ============================================================
# PARSE CASE
# ============================================================

def parse_case(case):

    parts = case.split("_")

    root_service = parts[1]
    fault_type = parts[2]
    repetition = parts[3]

    return root_service, fault_type, repetition


# ============================================================
# CREATE NESTED PARTIAL GRAPHS
# ============================================================

def create_nested_graphs(
    G,
    root_service
):

    services = list(G.nodes())

    # --------------------------------------------------------
    # Remove root from the pool.
    # Root must remain hidden.
    # --------------------------------------------------------

    if HIDE_ROOT:

        available_services = [
            service
            for service in services
            if service != root_service
        ]

    else:

        available_services = services.copy()

    # --------------------------------------------------------
    # ONE random ordering per incident.
    #
    # This is the important change.
    #
    # 20% is contained inside 40%
    # 40% is contained inside 60%
    # etc.
    # --------------------------------------------------------

    random.shuffle(available_services)

    # --------------------------------------------------------
    # Create every observation level from the same ordering.
    # --------------------------------------------------------

    partial_graphs = {}

    total_nodes = len(services)

    for observation_level in OBSERVATION_LEVELS:

        target_nodes = round(
            total_nodes * observation_level
        )

        # Cannot observe more nodes than available
        target_nodes = min(
            target_nodes,
            len(available_services)
        )

        selected = set(
            available_services[:target_nodes]
        )

        # ----------------------------------------------------
        # Induced subgraph
        # ----------------------------------------------------

        partial = G.subgraph(
            selected
        ).copy()

        partial.graph[
            "observation_level"
        ] = observation_level

        partial.graph[
            "observed_nodes"
        ] = len(partial.nodes())

        partial.graph[
            "original_nodes"
        ] = len(G.nodes())

        partial.graph[
            "hidden_root"
        ] = root_service

        partial_graphs[
            observation_level
        ] = partial

    return partial_graphs


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    DATA_DIR /
    "partial_incident_graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENERATE
# ============================================================

random.seed(RANDOM_SEED)

count = 0


for case, G in graphs.items():

    root_service, fault_type, repetition = (
        parse_case(case)
    )

    case_dir = (
        OUTPUT_DIR /
        case
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    partial_graphs = create_nested_graphs(
        G,
        root_service
    )

    for observation_level, partial in (
        partial_graphs.items()
    ):

        level_name = int(
            observation_level * 100
        )

        output_file = (
            case_dir /
            f"partial_{level_name}.gpickle"
        )

        with open(
            output_file,
            "wb"
        ) as f:

            pickle.dump(
                partial,
                f
            )

        count += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("NESTED PARTIAL INCIDENT GRAPHS")
print("=" * 60)

print("Cases:", len(graphs))

print(
    "Observation levels:",
    OBSERVATION_LEVELS
)

print(
    "Root hidden:",
    HIDE_ROOT
)

print(
    "Graphs created:",
    count
)

print()
print("Output:")
print(OUTPUT_DIR)