from pathlib import Path
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

FEATURE_FILE = DATA_DIR / "incident_service_features.parquet"

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]

# Graph contribution.
# We will test several values.
LAMBDA_VALUES = [0.25, 0.5, 0.75, 1.0]

# Randomness for reproducibility
RANDOM_SEED = 42

# Hold out same root + same fault type historical family
# This is not actually needed for this experiment because
# we are using the CURRENT graph only.
FAMILY_HOLDOUT = True


# ============================================================
# CASE PARSER
# ============================================================

def parse_case_name(case):
    """
    Example:

        re2ob_currencyservice_cpu_1

    returns:

        root_service = currencyservice
        fault_type   = cpu
    """

    parts = case.split("_")

    root_service = parts[1]
    fault_type = parts[2]

    return root_service, fault_type


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# CURRENT ANOMALY TRANSFORMATION
# ============================================================

def signed_log_change(before, after):

    if before == 0:

        if after == 0:
            return 0.0

        return np.log1p(after)

    change = (after - before) / before

    return (
        np.sign(change)
        * np.log1p(abs(change))
    )


def calculate_anomaly_score(row):

    signals = []

    # --------------------------------------------------------
    # Log count change
    # --------------------------------------------------------

    signals.append(
        abs(
            signed_log_change(
                row["log_before"],
                row["log_after"]
            )
        )
    )

    # --------------------------------------------------------
    # Trace count change
    # --------------------------------------------------------

    signals.append(
        abs(
            signed_log_change(
                row["trace_before"],
                row["trace_after"]
            )
        )
    )

    # --------------------------------------------------------
    # Median latency change
    # --------------------------------------------------------

    signals.append(
        abs(
            signed_log_change(
                row["median_before_ms"],
                row["median_after_ms"]
            )
        )
    )

    # --------------------------------------------------------
    # P95 latency change
    # --------------------------------------------------------

    signals.append(
        abs(
            signed_log_change(
                row["p95_before_ms"],
                row["p95_after_ms"]
            )
        )
    )

    return float(np.mean(signals))


# ============================================================
# NORMALIZE ANOMALY SCORES
# ============================================================

def normalize_scores(scores):

    if not scores:
        return {}

    values = np.array(
        list(scores.values()),
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    # All services have same score
    if maximum == minimum:

        return {
            service: 0.0
            for service in scores
        }

    return {
        service:
        (score - minimum)
        / (maximum - minimum)

        for service, score
        in scores.items()
    }


# ============================================================
# CURRENT-ONLY SCORE
# ============================================================

def current_only_scores(
    feature_rows,
    observed_services
):

    scores = {}

    for _, row in feature_rows.iterrows():

        service = row["service"]

        if service not in observed_services:
            continue

        scores[service] = calculate_anomaly_score(row)

    return normalize_scores(scores)


# ============================================================
# GRAPH PROPAGATION SCORE
# ============================================================

def graph_guided_scores(
    graph,
    current_scores,
    observed_services,
    lambda_value
):

    final_scores = {}

    for candidate in observed_services:

        # ----------------------------------------------------
        # Current anomaly of candidate itself
        # ----------------------------------------------------

        own_anomaly = current_scores.get(
            candidate,
            0.0
        )

        # ----------------------------------------------------
        # Find observed downstream services
        #
        # candidate -> ... -> downstream
        # ----------------------------------------------------

        downstream_evidence = []

        try:

            descendants = nx.descendants(
                graph,
                candidate
            )

        except nx.NetworkXError:

            descendants = set()

        for service in descendants:

            if service not in observed_services:
                continue

            if service not in current_scores:
                continue

            try:

                distance = nx.shortest_path_length(
                    graph,
                    candidate,
                    service
                )

            except nx.NetworkXNoPath:

                continue

            # ------------------------------------------------
            # Distance decay
            #
            # 1 hop  = 1/2
            # 2 hops = 1/3
            # 3 hops = 1/4
            # ------------------------------------------------

            weight = 1.0 / (1.0 + distance)

            evidence = (
                weight
                * current_scores[service]
            )

            downstream_evidence.append(
                evidence
            )

        # ----------------------------------------------------
        # Normalize downstream evidence
        #
        # Mean prevents high-degree services from automatically
        # winning just because they have many descendants.
        # ----------------------------------------------------

        if downstream_evidence:

            graph_evidence = np.mean(
                downstream_evidence
            )

        else:

            graph_evidence = 0.0

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        final_score = (
            own_anomaly
            +
            lambda_value
            * graph_evidence
        )

        final_scores[candidate] = final_score

    return final_scores


# ============================================================
# RANKING
# ============================================================

def rank_scores(scores):

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    return [
        service
        for service, score
        in ranked
    ]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(ranking, true_root):

    if true_root not in ranking:

        return {
            "top1": 0,
            "top3": 0,
            "top5": 0,
            "mrr": 0.0,
            "rank": None
        }

    rank = (
        ranking.index(true_root)
        + 1
    )

    return {
        "top1": int(rank == 1),
        "top3": int(rank <= 3),
        "top5": int(rank <= 5),
        "mrr": 1.0 / rank,
        "rank": rank
    }


# ============================================================
# BUILD ROOT-VISIBLE OBSERVATION
# ============================================================

def create_observed_services(
    all_services,
    true_root,
    observation_level,
    rng
):

    all_services = list(all_services)

    # --------------------------------------------------------
    # Root MUST remain visible.
    # --------------------------------------------------------

    remaining = [
        service
        for service in all_services
        if service != true_root
    ]

    # --------------------------------------------------------
    # Number of services to observe
    # --------------------------------------------------------

    target_count = int(
        round(
            len(all_services)
            * observation_level
            / 100.0
        )
    )

    target_count = max(
        target_count,
        1
    )

    target_count = min(
        target_count,
        len(all_services)
    )

    # Root always visible.
    number_to_select = target_count - 1

    if number_to_select > 0:

        selected = rng.sample(
            remaining,
            min(
                number_to_select,
                len(remaining)
            )
        )

    else:

        selected = []

    selected.append(true_root)

    return set(selected)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 80)
    print("CURRENT-ONLY vs GRAPH-GUIDED CURRENT RCA")
    print("=" * 80)

    print()
    print("Experiment:")
    print("  Root remains observable.")
    print("  Other services are partially hidden.")
    print("  Both methods receive exactly the same observed services.")
    print("  Graph is used only to propagate downstream anomaly evidence.")
    print()

    # --------------------------------------------------------
    # Load feature table
    # --------------------------------------------------------

    features = pd.read_parquet(
        FEATURE_FILE
    )

    cases = sorted(
        features["case"].unique()
    )

    print(
        f"Cases loaded: {len(cases)}"
    )

    # --------------------------------------------------------
    # Load current graphs
    # --------------------------------------------------------

    graphs = {}

    for case in cases:

        graph_path = (
            DATA_DIR
            / case
            / "current_incident_graph.gpickle"
        )

        if not graph_path.exists():

            print(
                f"WARNING: graph missing: {case}"
            )

            continue

        graphs[case] = load_graph(
            graph_path
        )

    print(
        f"Graphs loaded: {len(graphs)}"
    )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results = []

    # ========================================================
    # Observation levels
    # ========================================================

    for observation_level in OBSERVATION_LEVELS:

        print()
        print(
            "=" * 80
        )

        print(
            f"OBSERVATION LEVEL: "
            f"{observation_level}%"
        )

        print(
            "=" * 80
        )

        for case in cases:

            if case not in graphs:
                continue

            graph = graphs[case]

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            true_root, fault_type = (
                parse_case_name(case)
            )

            # ------------------------------------------------
            # Services available in the COMPLETE graph
            # ------------------------------------------------

            all_services = set(
                graph.nodes()
            )

            if true_root not in all_services:

                print(
                    f"WARNING: root "
                    f"{true_root} not in graph "
                    f"for {case}"
                )

                continue

            # ------------------------------------------------
            # Create partial observation
            #
            # IMPORTANT:
            # root is always visible.
            # ------------------------------------------------

            observed_services = (
                create_observed_services(
                    all_services,
                    true_root,
                    observation_level,
                    rng
                )
            )

            # ------------------------------------------------
            # Feature rows for this case
            # ------------------------------------------------

            case_features = features[
                features["case"] == case
            ].copy()

            # ------------------------------------------------
            # Current-only anomaly scores
            # ------------------------------------------------

            current_scores = (
                current_only_scores(
                    case_features,
                    observed_services
                )
            )

            # ------------------------------------------------
            # Current-only ranking
            # ------------------------------------------------

            current_ranking = rank_scores(
                current_scores
            )

            current_metrics = (
                calculate_metrics(
                    current_ranking,
                    true_root
                )
            )

            # ------------------------------------------------
            # Graph-guided ranking
            # ------------------------------------------------

            graph_results = {}

            for lambda_value in LAMBDA_VALUES:

                graph_scores = (
                    graph_guided_scores(
                        graph,
                        current_scores,
                        observed_services,
                        lambda_value
                    )
                )

                graph_ranking = rank_scores(
                    graph_scores
                )

                graph_metrics = (
                    calculate_metrics(
                        graph_ranking,
                        true_root
                    )
                )

                all_results.append({

                    "case": case,

                    "root": true_root,

                    "fault": fault_type,

                    "observation_level":
                        observation_level,

                    "lambda":
                        lambda_value,

                    "observed_services":
                        len(observed_services),

                    "current_top1":
                        current_metrics["top1"],

                    "current_top3":
                        current_metrics["top3"],

                    "current_top5":
                        current_metrics["top5"],

                    "current_mrr":
                        current_metrics["mrr"],

                    "graph_top1":
                        graph_metrics["top1"],

                    "graph_top3":
                        graph_metrics["top3"],

                    "graph_top5":
                        graph_metrics["top5"],

                    "graph_mrr":
                        graph_metrics["mrr"],

                    "current_rank":
                        current_metrics["rank"],

                    "graph_rank":
                        graph_metrics["rank"]
                })

    # ========================================================
    # Convert results
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    output_file = (
        DATA_DIR
        / "graph_boost_current_results.parquet"
    )

    results.to_parquet(
        output_file,
        index=False
    )

    print()
    print(
        f"Raw results saved to:"
    )
    print(
        output_file
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results
        .groupby(
            [
                "observation_level",
                "lambda"
            ]
        )
        .agg({

            "current_top1":
                "mean",

            "current_top3":
                "mean",

            "current_top5":
                "mean",

            "current_mrr":
                "mean",

            "graph_top1":
                "mean",

            "graph_top3":
                "mean",

            "graph_top5":
                "mean",

            "graph_mrr":
                "mean"
        })
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert to percentage
    # --------------------------------------------------------

    percentage_columns = [

        "current_top1",
        "current_top3",
        "current_top5",
        "graph_top1",
        "graph_top3",
        "graph_top5"
    ]

    for column in percentage_columns:

        summary[column] *= 100

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    summary["top1_improvement_pp"] = (
        summary["graph_top1"]
        -
        summary["current_top1"]
    )

    summary["mrr_improvement"] = (
        summary["graph_mrr"]
        -
        summary["current_mrr"]
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best lambda by Top-1
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("BEST GRAPH CONTRIBUTION")
    print("=" * 100)

    for level in OBSERVATION_LEVELS:

        subset = summary[
            summary["observation_level"]
            == level
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["graph_top1"].idxmax()
        ]

        print()

        print(
            f"{level}% observation"
        )

        print(
            f"  Lambda: "
            f"{best['lambda']}"
        )

        print(
            f"  Current Top-1: "
            f"{best['current_top1']:.2f}%"
        )

        print(
            f"  Graph+Current Top-1: "
            f"{best['graph_top1']:.2f}%"
        )

        print(
            f"  Improvement: "
            f"{best['top1_improvement_pp']:+.2f} pp"
        )

        print(
            f"  Current MRR: "
            f"{best['current_mrr']:.4f}"
        )

        print(
            f"  Graph+Current MRR: "
            f"{best['graph_mrr']:.4f}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()