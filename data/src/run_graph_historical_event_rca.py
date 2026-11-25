from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd
import networkx as nx


DATA_DIR = Path("data")
PARTIAL_DIR = DATA_DIR / "partial_incident_graphs"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]
MAX_HOPS = 3

# Do not compare a case against the same root+fault family.
# This prevents the experiment from simply memorizing repetitions.
FAMILY_HOLDOUT = True


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def parse_case_name(case):
    """
    Example:
    re2ob_checkoutservice_cpu_1

    Returns:
        root_service = checkoutservice
        fault_type   = cpu
    """
    parts = case.replace("re2ob_", "").split("_")

    root_service = parts[0]
    fault_type = parts[1]

    return root_service, fault_type


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_partial_graphs():

    graphs = []

    for case_dir in sorted(PARTIAL_DIR.glob("re2ob_*")):

        if not case_dir.is_dir():
            continue

        case = case_dir.name

        for level in OBSERVATION_LEVELS:

            path = case_dir / f"partial_{level}.gpickle"

            if not path.exists():
                continue

            G = load_pickle(path)

            graphs.append({
                "case": case,
                "level": level,
                "graph": G
            })

    return graphs


def load_historical_graph(case):

    path = DATA_DIR / case / "historical_incident_graph.gpickle"

    if not path.exists():
        return None

    return load_pickle(path)


def load_events(case):

    path = DATA_DIR / case / "events.parquet"

    if not path.exists():
        return None

    events = pd.read_parquet(path)

    events["timestamp"] = pd.to_numeric(
        events["timestamp"],
        errors="coerce"
    )

    events = events.dropna(subset=["timestamp"]).copy()

    events["service"] = events["service"].astype(str)
    events["template"] = events["template"].astype(str)

    return events


# ------------------------------------------------------------
# Graph matching
# ------------------------------------------------------------

def graph_similarity(current_graph, historical_graph):

    current_nodes = set(current_graph.nodes())
    historical_nodes = set(historical_graph.nodes())

    if not current_nodes:
        return 0.0

    node_overlap = (
        len(current_nodes & historical_nodes)
        / len(current_nodes)
    )

    current_edges = set(current_graph.edges())
    historical_edges = set(historical_graph.edges())

    if current_edges:
        edge_overlap = (
            len(current_edges & historical_edges)
            / len(current_edges)
        )

        return 0.5 * node_overlap + 0.5 * edge_overlap

    return node_overlap


# ------------------------------------------------------------
# Historical upstream candidates
# ------------------------------------------------------------

def find_upstream_candidates(
    current_graph,
    historical_graph,
    max_hops=3
):

    current_nodes = set(current_graph.nodes())

    candidates = {}

    queue = []

    for node in current_nodes:

        if node in historical_graph:
            queue.append((node, 0))

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
                    or new_distance < candidates[predecessor]
                ):
                    candidates[predecessor] = new_distance

            queue.append(
                (predecessor, new_distance)
            )

    return candidates


# ------------------------------------------------------------
# Historical downstream services
# ------------------------------------------------------------

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
                    or new_distance < descendants[child]
                ):
                    descendants[child] = new_distance

            queue.append(
                (child, new_distance)
            )

    return descendants


# ------------------------------------------------------------
# Event similarity
# ------------------------------------------------------------

def normalize_template(template):

    template = str(template).lower()

    # Replace generic placeholders consistently.
    template = re.sub(r"<num>", "<value>", template)
    template = re.sub(r"<uuid>", "<id>", template)
    template = re.sub(r"<product_id>", "<id>", template)

    # Normalize whitespace.
    template = re.sub(r"\s+", " ", template).strip()

    return template


def build_event_profile(events):

    """
    Event profile:

        service + normalized template -> frequency

    We use frequency-normalized evidence rather than
    individual event instances.
    """

    if events is None or events.empty:
        return {}

    profile = {}

    for _, row in events.iterrows():

        service = str(row["service"])
        template = normalize_template(row["template"])

        key = (service, template)

        profile[key] = profile.get(key, 0) + 1

    return profile


def candidate_historical_event_profile(
    historical_events,
    historical_graph,
    candidate,
    max_hops=3
):

    if historical_events is None:
        return {}

    services = get_downstream_services(
        historical_graph,
        candidate,
        max_hops
    )

    # Include candidate itself.
    services[candidate] = 0

    allowed_services = set(services.keys())

    filtered = historical_events[
        historical_events["service"].isin(
            allowed_services
        )
    ]

    return build_event_profile(filtered)


def event_similarity(
    current_profile,
    historical_profile
):

    if not current_profile or not historical_profile:
        return 0.0

    current_total = sum(current_profile.values())

    if current_total == 0:
        return 0.0

    matched = 0.0

    for key, count in current_profile.items():

        historical_count = historical_profile.get(
            key,
            0
        )

        if historical_count > 0:

            # Log-scaled frequency agreement.
            matched += min(
                np.log1p(count),
                np.log1p(historical_count)
            )

    denominator = sum(
        np.log1p(v)
        for v in current_profile.values()
    )

    if denominator == 0:
        return 0.0

    return matched / denominator


# ------------------------------------------------------------
# Current observed event profile
# ------------------------------------------------------------

def get_observed_events(
    events,
    current_graph
):

    if events is None:
        return None

    observed_services = set(
        current_graph.nodes()
    )

    return events[
        events["service"].isin(
            observed_services
        )
    ].copy()


# ------------------------------------------------------------
# Candidate scoring
# ------------------------------------------------------------

def rank_candidates(
    current_graph,
    historical_graph,
    current_events,
    historical_events
):

    graph_score = graph_similarity(
        current_graph,
        historical_graph
    )

    candidates = find_upstream_candidates(
        current_graph,
        historical_graph,
        MAX_HOPS
    )

    if not candidates:
        return []

    current_observed_events = get_observed_events(
        current_events,
        current_graph
    )

    current_profile = build_event_profile(
        current_observed_events
    )

    results = {}

    for candidate, distance in candidates.items():

        # Historical downstream event profile
        hist_profile = candidate_historical_event_profile(
            historical_events,
            historical_graph,
            candidate,
            MAX_HOPS
        )

        historical_event_score = event_similarity(
            current_profile,
            hist_profile
        )

        distance_score = 1.0 / (1.0 + distance)

        # Candidate-specific score.
        #
        # Graph score is global to the matched historical
        # graph, while event similarity is specific to
        # this candidate.
        score = (
            graph_score
            * distance_score
            * (1.0 + historical_event_score)
        )

        results[candidate] = {
            "score": score,
            "graph_score": graph_score,
            "historical_event_score":
                historical_event_score,
            "distance": distance
        }

    ranked = sorted(
        results.items(),
        key=lambda x: (
            -x[1]["score"],
            x[0]
        )
    )

    return ranked


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_rank(
    ranked,
    true_root
):

    ranking = [
        candidate
        for candidate, _ in ranked
    ]

    if true_root in ranking:

        rank = ranking.index(true_root) + 1

        top1 = int(rank <= 1)
        top3 = int(rank <= 3)
        top5 = int(rank <= 5)
        reciprocal = 1.0 / rank

    else:

        rank = None
        top1 = 0
        top3 = 0
        top5 = 0
        reciprocal = 0.0

    return (
        top1,
        top3,
        top5,
        reciprocal,
        rank
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    partial_graphs = load_partial_graphs()

    print()
    print("=" * 70)
    print("GRAPH + HISTORICAL EVENT RCA")
    print("=" * 70)
    print(f"Partial graphs: {len(partial_graphs)}")

    # Cache events and historical graphs.
    cases = sorted(
        {
            item["case"]
            for item in partial_graphs
        }
    )

    historical_graphs = {}
    events_cache = {}

    for case in cases:

        historical_graphs[case] = (
            load_historical_graph(case)
        )

        events_cache[case] = load_events(case)

    valid_events = sum(
        e is not None
        for e in events_cache.values()
    )

    print(f"Cases: {len(cases)}")
    print(f"Cases with events: {valid_events}")

    results = []

    for item in partial_graphs:

        case = item["case"]
        level = item["level"]
        current_graph = item["graph"]

        true_root, true_fault = parse_case_name(case)

        current_events = events_cache[case]

        eligible_cases = []

        for historical_case in cases:

            if historical_case == case:
                continue

            hist_root, hist_fault = parse_case_name(
                historical_case
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

            historical_events = (
                events_cache[
                    historical_case
                ]
            )

            if historical_graph is None:
                continue

            if historical_events is None:
                continue

            eligible_cases.append(
                (
                    historical_case,
                    historical_graph,
                    historical_events
                )
            )

        # Collect candidate support across historical incidents.
        candidate_support = {}

        for (
            historical_case,
            historical_graph,
            historical_events
        ) in eligible_cases:

            ranked = rank_candidates(
                current_graph,
                historical_graph,
                current_events,
                historical_events
            )

            for candidate, info in ranked:

                if candidate not in candidate_support:

                    candidate_support[candidate] = {
                        "score": 0.0,
                        "graph_score": 0.0,
                        "event_score": 0.0,
                        "count": 0
                    }

                candidate_support[candidate]["score"] += (
                    info["score"]
                )

                candidate_support[candidate]["graph_score"] += (
                    info["graph_score"]
                )

                candidate_support[candidate]["event_score"] += (
                    info["historical_event_score"]
                )

                candidate_support[candidate]["count"] += 1

        ranked_final = sorted(
            candidate_support.items(),
            key=lambda x: (
                -x[1]["score"],
                x[0]
            )
        )

        top1, top3, top5, rr, rank = evaluate_rank(
            ranked_final,
            true_root
        )

        results.append({

            "case": case,
            "observation_level": level,

            "true_root": true_root,
            "fault_type": true_fault,

            "top1": top1,
            "top3": top3,
            "top5": top5,
            "mrr": rr,
            "rank": rank,

            "num_candidates": len(
                ranked_final
            ),

            "num_observed_nodes": current_graph.number_of_nodes(),

            "num_observed_edges": current_graph.number_of_edges()
        })

    df = pd.DataFrame(results)

    output = DATA_DIR / (
        "graph_historical_event_rca_results.parquet"
    )

    df.to_parquet(
        output,
        index=False
    )

    summary = (
        df.groupby("observation_level")
        .agg(
            cases=("case", "count"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            mrr=("mrr", "mean"),
            avg_candidates=("num_candidates", "mean"),
            avg_nodes=("num_observed_nodes", "mean"),
            avg_edges=("num_observed_edges", "mean")
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
