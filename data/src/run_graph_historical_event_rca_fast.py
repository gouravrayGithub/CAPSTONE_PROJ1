from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import re

DATA_DIR = Path("data")
PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
MAX_HOPS = 3
FAMILY_HOLDOUT = True


# ============================================================
# BASIC UTILITIES
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_case_name(case):
    parts = case.replace("re2ob_", "").split("_")
    return parts[0], parts[1]


def normalize_template(template):
    template = str(template).lower()

    template = re.sub(r"<num>", "<value>", template)
    template = re.sub(r"<uuid>", "<id>", template)
    template = re.sub(r"<product_id>", "<id>", template)

    template = re.sub(r"\s+", " ", template).strip()

    return template


# ============================================================
# LOAD DATA
# ============================================================

def load_partial_graphs():

    graphs = []

    for case_dir in sorted(PARTIAL_DIR.glob("re2ob_*")):

        if not case_dir.is_dir():
            continue

        case = case_dir.name

        for level in OBSERVATION_LEVELS:

            path = case_dir / f"partial_{level}.gpickle"

            if path.exists():

                graphs.append({
                    "case": case,
                    "level": level,
                    "graph": load_pickle(path)
                })

    return graphs


def load_historical_graph(case):

    path = (
        DATA_DIR
        / case
        / "historical_incident_graph.gpickle"
    )

    if not path.exists():
        return None

    return load_pickle(path)


def load_events(case):

    path = DATA_DIR / case / "events.parquet"

    if not path.exists():
        return None

    events = pd.read_parquet(path)

    events["service"] = (
        events["service"]
        .astype(str)
    )

    events["template"] = (
        events["template"]
        .map(normalize_template)
    )

    return events


# ============================================================
# GRAPH FUNCTIONS
# ============================================================

def graph_similarity(current_graph, historical_graph):

    current_nodes = set(current_graph.nodes())
    historical_nodes = set(historical_graph.nodes())

    if not current_nodes:
        return 0.0

    node_score = (
        len(current_nodes & historical_nodes)
        / len(current_nodes)
    )

    current_edges = set(current_graph.edges())
    historical_edges = set(historical_graph.edges())

    if current_edges:

        edge_score = (
            len(current_edges & historical_edges)
            / len(current_edges)
        )

        return 0.5 * node_score + 0.5 * edge_score

    return node_score


def find_upstream_candidates(
    current_graph,
    historical_graph,
    max_hops=3
):

    current_nodes = set(current_graph.nodes())

    candidates = {}

    queue = [
        (node, 0)
        for node in current_nodes
        if node in historical_graph
    ]

    visited = set()

    while queue:

        node, distance = queue.pop(0)

        state = (node, distance)

        if state in visited:
            continue

        visited.add(state)

        if distance >= max_hops:
            continue

        if node not in historical_graph:
            continue

        for predecessor in historical_graph.predecessors(node):

            new_distance = distance + 1

            if predecessor not in current_nodes:

                if (
                    predecessor not in candidates
                    or
                    new_distance
                    < candidates[predecessor]
                ):
                    candidates[predecessor] = new_distance

            queue.append(
                (predecessor, new_distance)
            )

    return candidates


def get_downstream_services(
    historical_graph,
    candidate,
    max_hops=3
):

    descendants = {}

    if candidate not in historical_graph:
        return descendants

    queue = [(candidate, 0)]
    visited = set()

    while queue:

        node, distance = queue.pop(0)

        if (node, distance) in visited:
            continue

        visited.add((node, distance))

        if distance >= max_hops:
            continue

        for child in historical_graph.successors(node):

            new_distance = distance + 1

            if child != candidate:

                if (
                    child not in descendants
                    or
                    new_distance
                    < descendants[child]
                ):
                    descendants[child] = new_distance

            queue.append(
                (child, new_distance)
            )

    return descendants


# ============================================================
# EVENT PROFILES
# ============================================================

def build_event_profile(events):

    if events is None or events.empty:
        return {}

    counts = (
        events
        .groupby(
            ["service", "template"]
        )
        .size()
    )

    return counts.to_dict()


def filter_profile_by_services(
    profile,
    allowed_services
):

    return {
        key: count
        for key, count in profile.items()
        if key[0] in allowed_services
    }


def event_similarity(
    current_profile,
    historical_profile
):

    if not current_profile or not historical_profile:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for key, count in current_profile.items():

        current_weight = np.log1p(count)

        denominator += current_weight

        historical_count = (
            historical_profile.get(key, 0)
        )

        if historical_count > 0:

            numerator += min(
                current_weight,
                np.log1p(historical_count)
            )

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# PRECOMPUTE CANDIDATE EVENT PROFILES
# ============================================================

def build_candidate_profiles(
    historical_graph,
    historical_profile
):

    profiles = {}

    for candidate in historical_graph.nodes():

        downstream = get_downstream_services(
            historical_graph,
            candidate,
            MAX_HOPS
        )

        allowed_services = set(
            downstream.keys()
        )

        allowed_services.add(candidate)

        profile = filter_profile_by_services(
            historical_profile,
            allowed_services
        )

        profiles[candidate] = profile

    return profiles


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FAST GRAPH + HISTORICAL EVENT RCA")
    print("=" * 70)

    partial_graphs = load_partial_graphs()

    print(
        f"Partial graphs: {len(partial_graphs)}"
    )

    cases = sorted({
        item["case"]
        for item in partial_graphs
    })

    print(
        f"Cases: {len(cases)}"
    )

    # --------------------------------------------------------
    # Load everything once
    # --------------------------------------------------------

    print()
    print("Loading historical graphs and events...")

    historical_graphs = {}
    event_profiles = {}

    for i, case in enumerate(cases, 1):

        historical_graphs[case] = (
            load_historical_graph(case)
        )

        events = load_events(case)

        event_profiles[case] = (
            build_event_profile(events)
        )

        if i % 15 == 0 or i == len(cases):

            print(
                f"  Loaded {i}/{len(cases)} cases"
            )

    print("Precomputing candidate event profiles...")

    candidate_profiles = {}

    for i, case in enumerate(cases, 1):

        graph = historical_graphs[case]

        if graph is None:
            continue

        candidate_profiles[case] = (
            build_candidate_profiles(
                graph,
                event_profiles[case]
            )
        )

        if i % 15 == 0 or i == len(cases):

            print(
                f"  Profiles {i}/{len(cases)}"
            )

    print()
    print("Starting RCA evaluation...")

    results = []

    total = len(partial_graphs)

    for idx, item in enumerate(
        partial_graphs,
        1
    ):

        case = item["case"]
        level = item["level"]
        current_graph = item["graph"]

        true_root, true_fault = (
            parse_case_name(case)
        )

        current_profile = (
            filter_profile_by_services(
                event_profiles[case],
                set(current_graph.nodes())
            )
        )

        candidate_support = {}

        for historical_case in cases:

            if historical_case == case:
                continue

            hist_root, hist_fault = (
                parse_case_name(
                    historical_case
                )
            )

            if FAMILY_HOLDOUT:

                if (
                    hist_root == true_root
                    and hist_fault == true_fault
                ):
                    continue

            historical_graph = (
                historical_graphs[
                    historical_case
                ]
            )

            if historical_graph is None:
                continue

            graph_score = graph_similarity(
                current_graph,
                historical_graph
            )

            candidates = (
                find_upstream_candidates(
                    current_graph,
                    historical_graph,
                    MAX_HOPS
                )
            )

            if not candidates:
                continue

            profiles = (
                candidate_profiles[
                    historical_case
                ]
            )

            for candidate, distance in (
                candidates.items()
            ):

                hist_profile = (
                    profiles.get(
                        candidate,
                        {}
                    )
                )

                hist_event_score = (
                    event_similarity(
                        current_profile,
                        hist_profile
                    )
                )

                distance_score = (
                    1.0 / (1.0 + distance)
                )

                score = (
                    graph_score
                    * distance_score
                    * (1.0 + hist_event_score)
                )

                candidate_support.setdefault(
                    candidate,
                    0.0
                )

                candidate_support[candidate] += score

        ranked = sorted(
            candidate_support.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        ranking = [
            candidate
            for candidate, _ in ranked
        ]

        if true_root in ranking:

            rank = (
                ranking.index(true_root)
                + 1
            )

            top1 = int(rank <= 1)
            top3 = int(rank <= 3)
            top5 = int(rank <= 5)
            mrr = 1.0 / rank

        else:

            rank = None
            top1 = 0
            top3 = 0
            top5 = 0
            mrr = 0.0

        results.append({

            "case": case,
            "observation_level": level,
            "true_root": true_root,
            "fault_type": true_fault,

            "top1": top1,
            "top3": top3,
            "top5": top5,
            "mrr": mrr,
            "rank": rank,

            "num_candidates": len(ranked),

            "num_observed_nodes":
                current_graph.number_of_nodes(),

            "num_observed_edges":
                current_graph.number_of_edges()
        })

        # Progress
        if (
            idx % 10 == 0
            or idx == total
        ):

            print(
                f"  Processed "
                f"{idx}/{total}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    output = (
        DATA_DIR
        / "graph_historical_event_rca_results.parquet"
    )

    df.to_parquet(
        output,
        index=False
    )

    summary = (
        df
        .groupby("observation_level")
        .agg(
            cases=("case", "count"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            mrr=("mrr", "mean"),
            avg_candidates=(
                "num_candidates",
                "mean"
            ),
            avg_nodes=(
                "num_observed_nodes",
                "mean"
            ),
            avg_edges=(
                "num_observed_edges",
                "mean"
            )
        )
        .reset_index()
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
