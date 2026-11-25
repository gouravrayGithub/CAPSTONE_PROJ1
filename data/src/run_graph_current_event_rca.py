from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

FEATURE_FILE = DATA_DIR / "incident_service_features.parquet"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]

MAX_HOPS = 3

# Prevent using the same root-service + fault-type family.
# Example:
# re2ob_checkoutservice_cpu_1
# cannot use:
# re2ob_checkoutservice_cpu_2
# re2ob_checkoutservice_cpu_3
FAMILY_HOLDOUT = True


# ============================================================
# HELPERS
# ============================================================

def parse_case_name(case):
    """
    Example:
        re2ob_checkoutservice_cpu_1

    Returns:
        root_service = checkoutservice
        fault_type   = cpu
    """

    parts = case.split("_")

    root_service = parts[1]
    fault_type = parts[2]

    return root_service, fault_type


def signed_log_change(before, after):
    """
    Stable transformation of frequency changes.

    Used only to obtain anomaly magnitude.
    """

    if before == 0:

        if after == 0:
            return 0.0

        return np.log1p(after)

    change = (after - before) / before

    return np.sign(change) * np.log1p(abs(change))


def load_graph(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# FIND UPSTREAM CANDIDATES
# ============================================================

def find_upstream_candidates(
    current_graph,
    historical_graph,
    max_hops=3
):
    """
    Find historical upstream services that are not
    currently observed.

    candidate -> ... -> observed current node
    """

    current_nodes = set(
        current_graph.nodes()
    )

    candidates = {}

    queue = []

    for node in current_nodes:

        if node in historical_graph:

            queue.append(
                (node, 0)
            )

    visited = set()

    while queue:

        node, distance = queue.pop(0)

        state = (
            node,
            distance
        )

        if state in visited:
            continue

        visited.add(state)

        if distance >= max_hops:
            continue

        if node not in historical_graph:
            continue

        for predecessor in (
            historical_graph.predecessors(node)
        ):

            new_distance = distance + 1

            if predecessor not in current_nodes:

                if (
                    predecessor not in candidates
                    or
                    new_distance
                    < candidates[predecessor]
                ):

                    candidates[predecessor] = (
                        new_distance
                    )

            queue.append(
                (
                    predecessor,
                    new_distance
                )
            )

    return candidates


# ============================================================
# CURRENT EVENT-LEVEL EVIDENCE
# ============================================================

def build_current_event_evidence(case_dir):
    """
    Build event-level anomaly evidence.

    Each event is represented using:

        service
        event_id

    Evidence is based on before/after frequency.
    """

    events_file = (
        case_dir / "events.parquet"
    )

    if not events_file.exists():

        return pd.DataFrame()

    events = pd.read_parquet(
        events_file
    )

    if events.empty:

        return pd.DataFrame()

    injection_file = (
        case_dir / "inject_time.txt"
    )

    injection_time = int(
        injection_file.read_text().strip()
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    events["timestamp"] = pd.to_numeric(
        events["timestamp"],
        errors="coerce"
    )

    events = events.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Before / after
    # --------------------------------------------------------

    before = events[
        events["timestamp"] < injection_time
    ]

    after = events[
        events["timestamp"] >= injection_time
    ]

    # --------------------------------------------------------
    # Event counts
    # --------------------------------------------------------

    group_cols = [
        "service",
        "event_id"
    ]

    before_counts = (
        before
        .groupby(group_cols)
        .size()
        .rename("count_before")
    )

    after_counts = (
        after
        .groupby(group_cols)
        .size()
        .rename("count_after")
    )

    evidence = pd.concat(
        [
            before_counts,
            after_counts
        ],
        axis=1
    ).fillna(0)

    evidence = evidence.reset_index()

    # --------------------------------------------------------
    # Changes
    # --------------------------------------------------------

    evidence["frequency_change"] = (
        evidence["count_after"]
        -
        evidence["count_before"]
    )

    evidence["relative_change"] = np.where(
        evidence["count_before"] > 0,

        (
            evidence["count_after"]
            -
            evidence["count_before"]
        )
        /
        evidence["count_before"],

        np.where(
            evidence["count_after"] > 0,
            1.0,
            0.0
        )
    )

    # --------------------------------------------------------
    # Anomaly magnitude
    # --------------------------------------------------------

    evidence["anomaly_score"] = evidence.apply(
        lambda row: abs(
            signed_log_change(
                row["count_before"],
                row["count_after"]
            )
        ),
        axis=1
    )

    # Keep only changed events
    evidence = evidence[
        evidence["anomaly_score"] > 0
    ].copy()

    return evidence


# ============================================================
# CANDIDATE-SPECIFIC CURRENT EVENT EVIDENCE
# ============================================================

def candidate_event_evidence(
    candidate,
    historical_graph,
    current_graph,
    current_events,
    max_hops=3
):
    """
    Calculate current event evidence for one candidate.

    Candidate
        ↓
    historical downstream path
        ↓
    observed current service
        ↓
    anomalous current event
    """

    if current_events.empty:

        return 0.0

    current_nodes = set(
        current_graph.nodes()
    )

    if candidate not in historical_graph:

        return 0.0

    # --------------------------------------------------------
    # Find downstream observed services
    # --------------------------------------------------------

    queue = [
        (candidate, 0)
    ]

    visited = set()

    downstream = {}

    while queue:

        node, distance = queue.pop(0)

        if node in visited:
            continue

        visited.add(node)

        if distance >= max_hops:
            continue

        for child in (
            historical_graph.successors(node)
        ):

            new_distance = distance + 1

            if child in current_nodes:

                if (
                    child not in downstream
                    or
                    new_distance
                    < downstream[child]
                ):

                    downstream[child] = (
                        new_distance
                    )

            queue.append(
                (
                    child,
                    new_distance
                )
            )

    if not downstream:

        return 0.0

    # --------------------------------------------------------
    # Match current events
    # --------------------------------------------------------

    score = 0.0
    weight_sum = 0.0

    for _, row in current_events.iterrows():

        service = row["service"]

        if service not in downstream:

            continue

        distance = downstream[service]

        distance_weight = (
            1.0 /
            (1.0 + distance)
        )

        anomaly = float(
            row["anomaly_score"]
        )

        score += (
            distance_weight
            * anomaly
        )

        weight_sum += (
            distance_weight
        )

    if weight_sum == 0:

        return 0.0

    return (
        score /
        weight_sum
    )


# ============================================================
# GRAPH STRUCTURAL SCORE
# ============================================================

def graph_match_score(
    current_graph,
    historical_graph
):

    current_nodes = set(
        current_graph.nodes()
    )

    historical_nodes = set(
        historical_graph.nodes()
    )

    if not current_nodes:

        return 0.0

    # --------------------------------------------------------
    # Node overlap
    # --------------------------------------------------------

    node_overlap = (
        len(
            current_nodes
            &
            historical_nodes
        )
        /
        len(current_nodes)
    )

    # --------------------------------------------------------
    # Edge overlap
    # --------------------------------------------------------

    current_edges = set(
        current_graph.edges()
    )

    historical_edges = set(
        historical_graph.edges()
    )

    if current_edges:

        edge_overlap = (
            len(
                current_edges
                &
                historical_edges
            )
            /
            len(current_edges)
        )

        structural_score = (
            node_overlap
            +
            edge_overlap
        ) / 2.0

    else:

        structural_score = (
            node_overlap
        )

    return structural_score


# ============================================================
# LOAD PARTIAL GRAPHS
# ============================================================

def load_partial_graphs():

    graphs = {}

    partial_root = (
        DATA_DIR /
        "partial_incident_graphs"
    )

    # Example:
    #
    # data/partial_incident_graphs/
    #     re2ob_checkoutservice_cpu_1/
    #         partial_20.gpickle

    pattern = (
        "re2ob_*"
        "/*partial_*.gpickle"
    )

    paths = list(
        partial_root.glob(pattern)
    )

    for path in paths:

        graph = load_graph(path)

        # ----------------------------------------------------
        # IMPORTANT:
        # Get case from directory name.
        # Do not depend on graph metadata.
        # ----------------------------------------------------

        case = path.parent.name

        # ----------------------------------------------------
        # Get observation level from filename.
        # ----------------------------------------------------

        match = re.match(
            r"partial_(\d+)\.gpickle$",
            path.name
        )

        if not match:

            continue

        level = int(
            match.group(1)
        )

        graphs[
            (case, level)
        ] = graph

    return graphs


# ============================================================
# LOAD HISTORICAL GRAPHS
# ============================================================

def load_historical_graphs(
    cases
):

    historical_graphs = {}

    for case in cases:

        path = (
            DATA_DIR
            / case
            / "historical_incident_graph.gpickle"
        )

        if not path.exists():

            continue

        historical_graphs[case] = (
            load_graph(path)
        )

    return historical_graphs


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GRAPH + CURRENT EVENT RCA")
    print("=" * 70)

    # ========================================================
    # LOAD SERVICE FEATURES
    # ========================================================

    features = pd.read_parquet(
        FEATURE_FILE
    )

    cases = sorted(
        features["case"].unique()
    )

    print(
        f"Cases: {len(cases)}"
    )

    # ========================================================
    # LOAD PARTIAL GRAPHS
    # ========================================================

    partial_graphs = (
        load_partial_graphs()
    )

    print(
        f"Partial graphs loaded: "
        f"{len(partial_graphs)}"
    )

    print(
        "Sample partial graph keys:"
    )

    for key in list(
        partial_graphs.keys()
    )[:5]:

        print(
            " ",
            key
        )

    # ========================================================
    # CHECK EXPECTED CASES
    # ========================================================

    print(
        "Sample feature cases:"
    )

    for case in cases[:5]:

        print(
            " ",
            case
        )

    # ========================================================
    # LOAD CURRENT EVENT EVIDENCE
    # ========================================================

    print()
    print(
        "Building current event evidence..."
    )

    event_cache = {}

    for case in cases:

        case_dir = (
            DATA_DIR / case
        )

        event_cache[case] = (
            build_current_event_evidence(
                case_dir
            )
        )

    # ========================================================
    # LOAD HISTORICAL GRAPHS
    # ========================================================

    historical_graphs = (
        load_historical_graphs(
            cases
        )
    )

    print(
        f"Historical graphs loaded: "
        f"{len(historical_graphs)}"
    )

    print(
        "Sample historical cases:"
    )

    for case in list(
        historical_graphs.keys()
    )[:5]:

        print(
            " ",
            case
        )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if len(partial_graphs) == 0:

        raise RuntimeError(
            "No partial graphs were loaded."
        )

    if len(historical_graphs) == 0:

        raise RuntimeError(
            "No historical graphs were loaded."
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    for observation_level in (
        OBSERVATION_LEVELS
    ):

        print()
        print(
            f"Observation level: "
            f"{observation_level}%"
        )

        for case in cases:

            key = (
                case,
                observation_level
            )

            if key not in partial_graphs:

                continue

            current_graph = (
                partial_graphs[key]
            )

            true_root, fault_type = (
                parse_case_name(case)
            )

            current_events = (
                event_cache[case]
            )

            candidates = {}

            # =================================================
            # HISTORICAL INCIDENTS
            # =================================================

            for (
                historical_case,
                historical_graph
            ) in historical_graphs.items():

                # Never compare case with itself
                if historical_case == case:

                    continue

                hist_root, hist_fault = (
                    parse_case_name(
                        historical_case
                    )
                )

                # ------------------------------------------------
                # FAMILY HOLDOUT
                # ------------------------------------------------

                if FAMILY_HOLDOUT:

                    if (
                        hist_root == true_root
                        and
                        hist_fault == fault_type
                    ):

                        continue

                # ------------------------------------------------
                # GRAPH MATCH
                # ------------------------------------------------

                structural = (
                    graph_match_score(
                        current_graph,
                        historical_graph
                    )
                )

                if structural <= 0:

                    continue

                # ------------------------------------------------
                # UPSTREAM CANDIDATES
                # ------------------------------------------------

                upstream = (
                    find_upstream_candidates(
                        current_graph,
                        historical_graph,
                        MAX_HOPS
                    )
                )

                for (
                    candidate,
                    distance
                ) in upstream.items():

                    # ==========================================
                    # GRAPH EVIDENCE
                    # ==========================================

                    graph_score = (
                        structural
                        *
                        (
                            1.0 /
                            (1.0 + distance)
                        )
                    )

                    # ==========================================
                    # CURRENT EVENT EVIDENCE
                    # ==========================================

                    event_score = (
                        candidate_event_evidence(
                            candidate,
                            historical_graph,
                            current_graph,
                            current_events,
                            MAX_HOPS
                        )
                    )

                    # ==========================================
                    # COMBINED SCORE
                    # ==========================================

                    candidate_score = (
                        graph_score
                        *
                        (1.0 + event_score)
                    )

                    candidates.setdefault(
                        candidate,
                        []
                    ).append(
                        candidate_score
                    )

            # =================================================
            # AGGREGATE HISTORICAL SUPPORT
            # =================================================

            candidate_scores = {}

            for (
                candidate,
                scores
            ) in candidates.items():

                candidate_scores[candidate] = (
                    np.mean(scores)
                )

            # =================================================
            # RANKING
            # =================================================

            ranked = sorted(
                candidate_scores.items(),
                key=lambda x: (
                    -x[1],
                    x[0]
                )
            )

            ranking = [
                service
                for service, score
                in ranked
            ]

            # =================================================
            # METRICS
            # =================================================

            if true_root in ranking:

                rank = (
                    ranking.index(
                        true_root
                    )
                    + 1
                )

                top1 = int(
                    rank == 1
                )

                top3 = int(
                    rank <= 3
                )

                top5 = int(
                    rank <= 5
                )

                reciprocal_rank = (
                    1.0 / rank
                )

            else:

                rank = np.nan

                top1 = 0
                top3 = 0
                top5 = 0

                reciprocal_rank = 0.0

            results.append(
                {
                    "case": case,

                    "observation_level":
                        observation_level,

                    "true_root":
                        true_root,

                    "predicted_root":
                        ranking[0]
                        if ranking
                        else None,

                    "rank":
                        rank,

                    "top1":
                        top1,

                    "top3":
                        top3,

                    "top5":
                        top5,

                    "mrr":
                        reciprocal_rank,

                    "num_candidates":
                        len(ranking),
                }
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise RuntimeError(
            "No RCA results were generated. "
            "Check case names and graph keys."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby(
            "observation_level"
        )
        .agg(
            cases=(
                "case",
                "count"
            ),

            top1=(
                "top1",
                "mean"
            ),

            top3=(
                "top3",
                "mean"
            ),

            top5=(
                "top5",
                "mean"
            ),

            mrr=(
                "mrr",
                "mean"
            ),

            avg_candidates=(
                "num_candidates",
                "mean"
            ),
        )
        .reset_index()
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = (
        DATA_DIR
        /
        "graph_current_event_rca_results.parquet"
    )

    results_df.to_parquet(
        output,
        index=False
    )

    print()
    print("Saved:")
    print(output)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()