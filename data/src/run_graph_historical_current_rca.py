import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

OBSERVATION_LEVELS = [20, 40, 60, 80, 100]

# Graph contribution
LAMBDA_GRAPH = [0.25, 0.5, 0.75, 1.0]

# Historical contribution
LAMBDA_HIST = [0.25, 0.5, 0.75, 1.0]

SEED = 42

# ============================================================
# HELPERS
# ============================================================

def parse_case(case):
    parts = case.replace("re2ob_", "").split("_")

    root = parts[0]
    fault = parts[1]

    return root, fault


def normalize_service(service):
    if service == "frontendservice":
        return "frontend"
    return service


def signed_log(x):
    return np.sign(x) * np.log1p(abs(x))


def anomaly_score(row):
    values = [
        row.get("log_change_pct", 0),
        row.get("trace_change_pct", 0),
        row.get("median_change_pct", 0),
        row.get("p95_change_pct", 0),
    ]

    values = [
        abs(signed_log(float(x)))
        for x in values
        if pd.notna(x)
    ]

    if not values:
        return 0.0

    return float(np.mean(values))


def normalize_scores(scores):

    if not scores:
        return {}

    vals = np.array(list(scores.values()), dtype=float)

    mn = vals.min()
    mx = vals.max()

    if mx - mn < 1e-12:
        return {k: 0.0 for k in scores}

    return {
        k: float((v - mn) / (mx - mn))
        for k, v in scores.items()
    }


def get_downstream_nodes(G, source, observed=None, max_hops=3):

    if source not in G:
        return set()

    if observed is not None:
        observed = set(observed)

    result = set()

    queue = [(source, 0)]
    visited = {source}

    while queue:

        node, depth = queue.pop(0)

        if depth >= max_hops:
            continue

        for child in G.successors(node):

            if child in visited:
                continue

            visited.add(child)

            if observed is None or child in observed:
                result.add(child)

            queue.append((child, depth + 1))

    return result


def jaccard(a, b):

    a = set(a)
    b = set(b)

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def shortest_distance(G, source, target):

    try:
        return nx.shortest_path_length(
            G,
            source=source,
            target=target
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ============================================================
# LOAD CASES
# ============================================================

cases = sorted([
    p.name
    for p in DATA_DIR.glob("re2ob_*")
    if p.is_dir()
])

print("=" * 90)
print("CURRENT + GRAPH + HISTORICAL GRAPH RCA")
print("=" * 90)

print()
print("Experiment:")
print("  Root remains observable.")
print("  Other services are partially hidden.")
print("  Current anomaly evidence is used.")
print("  Current dependency graph propagates anomaly evidence.")
print("  Historical graphs provide structural evidence.")
print("  Same root+fault historical cases are excluded.")
print()

print(f"Cases found: {len(cases)}")


# ============================================================
# LOAD DATA
# ============================================================

feature_file = DATA_DIR / "incident_service_features.parquet"

features = pd.read_parquet(feature_file)

features["service"] = features["service"].map(
    normalize_service
)

# Make lookup:
# case -> service -> feature row
feature_lookup = {}

for case, group in features.groupby("case"):

    feature_lookup[case] = {
        row["service"]: row
        for _, row in group.iterrows()
    }


# ============================================================
# LOAD CURRENT + HISTORICAL GRAPHS
# ============================================================

current_graphs = {}
historical_graphs = {}

for case in cases:

    current_path = (
        DATA_DIR /
        case /
        "current_incident_graph.gpickle"
    )

    historical_path = (
        DATA_DIR /
        case /
        "historical_incident_graph.gpickle"
    )

    if current_path.exists():

        with open(current_path, "rb") as f:
            current_graphs[case] = pickle.load(f)

    if historical_path.exists():

        with open(historical_path, "rb") as f:
            historical_graphs[case] = pickle.load(f)


print(f"Current graphs loaded: {len(current_graphs)}")
print(f"Historical graphs loaded: {len(historical_graphs)}")


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN EXPERIMENT
# ============================================================

for observation_level in OBSERVATION_LEVELS:

    print()
    print("=" * 90)
    print(f"OBSERVATION LEVEL: {observation_level}%")
    print("=" * 90)

    for lambda_graph in LAMBDA_GRAPH:

        for lambda_hist in LAMBDA_HIST:

            current_ranks = []
            graph_ranks = []
            combined_ranks = []

            current_top1 = 0
            graph_top1 = 0
            combined_top1 = 0

            current_top3 = 0
            graph_top3 = 0
            combined_top3 = 0

            current_top5 = 0
            graph_top5 = 0
            combined_top5 = 0

            current_rr = []
            graph_rr = []
            combined_rr = []

            valid_cases = 0

            for case_index, case in enumerate(cases):

                if case not in current_graphs:
                    continue

                if case not in historical_graphs:
                    continue

                if case not in feature_lookup:
                    continue

                root, fault = parse_case(case)

                G = current_graphs[case]

                # ------------------------------------------------
                # Feature lookup
                # ------------------------------------------------

                case_features = feature_lookup[case]

                # Services available in current feature table
                services = set(case_features.keys())

                # Root must be present
                if root not in services:
                    continue

                # ------------------------------------------------
                # Observation sampling
                # ------------------------------------------------

                rng = np.random.default_rng(
                    SEED + case_index
                )

                other_services = sorted(
                    services - {root}
                )

                target_count = max(
                    1,
                    round(
                        len(services) *
                        observation_level / 100
                    )
                )

                # Root ALWAYS observable
                n_other = max(
                    0,
                    target_count - 1
                )

                n_other = min(
                    n_other,
                    len(other_services)
                )

                if n_other > 0:

                    selected = rng.choice(
                        other_services,
                        size=n_other,
                        replace=False
                    )

                    observed = set(selected)
                    observed.add(root)

                else:

                    observed = {root}

                # ------------------------------------------------
                # Current anomaly score
                # ------------------------------------------------

                raw_current = {}

                for service in observed:

                    row = case_features.get(service)

                    if row is None:
                        continue

                    raw_current[service] = anomaly_score(row)

                current_scores = normalize_scores(
                    raw_current
                )

                # ------------------------------------------------
                # Induced current graph
                #
                # Important:
                # hidden services are NOT allowed to act as
                # intermediate paths.
                # ------------------------------------------------

                G_obs = G.subgraph(
                    observed
                ).copy()

                # ------------------------------------------------
                # GRAPH SCORE
                # ------------------------------------------------

                graph_raw = {}

                for candidate in observed:

                    propagation = 0.0

                    for service, anomaly in current_scores.items():

                        if service == candidate:
                            continue

                        distance = shortest_distance(
                            G_obs,
                            candidate,
                            service
                        )

                        if distance is None:
                            continue

                        weight = 1.0 / (
                            1.0 + distance
                        )

                        propagation += (
                            anomaly *
                            weight
                        )

                    graph_raw[candidate] = propagation

                graph_scores = normalize_scores(
                    graph_raw
                )

                # ------------------------------------------------
                # HISTORICAL STRUCTURAL SCORE
                # ------------------------------------------------

                historical_scores = {}

                # Historical cases from the SAME
                # root+fault family are excluded.
                valid_history = []

                for hist_case, H in historical_graphs.items():

                    if hist_case == case:
                        continue

                    hist_root, hist_fault = parse_case(
                        hist_case
                    )

                    if (
                        hist_root == root
                        and
                        hist_fault == fault
                    ):
                        continue

                    valid_history.append(
                        (hist_case, H)
                    )

                for candidate in observed:

                    current_downstream = (
                        get_downstream_nodes(
                            G_obs,
                            candidate,
                            observed=observed,
                            max_hops=3
                        )
                    )

                    similarities = []

                    for hist_case, H in valid_history:

                        if candidate not in H:
                            continue

                        historical_downstream = (
                            get_downstream_nodes(
                                H,
                                candidate,
                                observed=None,
                                max_hops=3
                            )
                        )

                        # Compare only services that are
                        # actually observed in current incident.
                        historical_downstream &= observed

                        similarity = jaccard(
                            current_downstream,
                            historical_downstream
                        )

                        similarities.append(
                            similarity
                        )

                    if similarities:

                        # Use strongest historical
                        # structural support rather than
                        # blindly averaging all incidents.
                        historical_scores[candidate] = (
                            max(similarities)
                        )

                    else:

                        historical_scores[candidate] = 0.0

                historical_scores = normalize_scores(
                    historical_scores
                )

                # ------------------------------------------------
                # COMBINED SCORE
                # ------------------------------------------------

                combined_raw = {}

                for candidate in observed:

                    current = current_scores.get(
                        candidate,
                        0.0
                    )

                    graph = graph_scores.get(
                        candidate,
                        0.0
                    )

                    historical = historical_scores.get(
                        candidate,
                        0.0
                    )

                    combined_raw[candidate] = (
                        current
                        +
                        lambda_graph * graph
                        +
                        lambda_hist * historical
                    )

                # ------------------------------------------------
                # RANK
                # ------------------------------------------------

                def rank(scores):

                    return [
                        x[0]
                        for x in sorted(
                            scores.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    ]

                current_rank = rank(
                    current_scores
                )

                graph_rank = rank(
                    {
                        s:
                        current_scores.get(s, 0.0)
                        +
                        lambda_graph *
                        graph_scores.get(s, 0.0)
                        for s in observed
                    }
                )

                combined_rank = rank(
                    combined_raw
                )

                if root not in current_rank:
                    continue

                valid_cases += 1

                # ------------------------------------------------
                # Rank metrics
                # ------------------------------------------------

                cr = current_rank.index(root) + 1
                gr = graph_rank.index(root) + 1
                hr = combined_rank.index(root) + 1

                current_rr.append(
                    1.0 / cr
                )

                graph_rr.append(
                    1.0 / gr
                )

                combined_rr.append(
                    1.0 / hr
                )

                if cr <= 1:
                    current_top1 += 1

                if cr <= 3:
                    current_top3 += 1

                if cr <= 5:
                    current_top5 += 1

                if gr <= 1:
                    graph_top1 += 1

                if gr <= 3:
                    graph_top3 += 1

                if gr <= 5:
                    graph_top5 += 1

                if hr <= 1:
                    combined_top1 += 1

                if hr <= 3:
                    combined_top3 += 1

                if hr <= 5:
                    combined_top5 += 1

            # ----------------------------------------------------
            # SAVE METRICS
            # ----------------------------------------------------

            if valid_cases == 0:
                continue

            results.append({

                "observation_level":
                    observation_level,

                "lambda_graph":
                    lambda_graph,

                "lambda_hist":
                    lambda_hist,

                "valid_cases":
                    valid_cases,

                "current_top1":
                    100 * current_top1 / valid_cases,

                "current_top3":
                    100 * current_top3 / valid_cases,

                "current_top5":
                    100 * current_top5 / valid_cases,

                "current_mrr":
                    np.mean(current_rr),

                "graph_top1":
                    100 * graph_top1 / valid_cases,

                "graph_top3":
                    100 * graph_top3 / valid_cases,

                "graph_top5":
                    100 * graph_top5 / valid_cases,

                "graph_mrr":
                    np.mean(graph_rr),

                "combined_top1":
                    100 * combined_top1 / valid_cases,

                "combined_top3":
                    100 * combined_top3 / valid_cases,

                "combined_top5":
                    100 * combined_top5 / valid_cases,

                "combined_mrr":
                    np.mean(combined_rr),

                "combined_top1_improvement_pp":
                    (
                        100 * combined_top1 / valid_cases
                        -
                        100 * current_top1 / valid_cases
                    ),

                "combined_mrr_improvement":
                    (
                        np.mean(combined_rr)
                        -
                        np.mean(current_rr)
                    )
            })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

output = (
    DATA_DIR /
    "graph_historical_current_rca_results.parquet"
)

results_df.to_parquet(
    output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 110)
print("FINAL SUMMARY")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST COMBINED RESULT PER OBSERVATION LEVEL
# ============================================================

print()
print("=" * 110)
print("BEST GRAPH + HISTORICAL + CURRENT")
print("=" * 110)

for level in OBSERVATION_LEVELS:

    subset = results_df[
        results_df["observation_level"]
        == level
    ]

    if subset.empty:
        continue

    best = subset.loc[
        subset["combined_top1"].idxmax()
    ]

    print()
    print(f"{level}% observation")

    print(
        f"  λ_graph: "
        f"{best['lambda_graph']}"
    )

    print(
        f"  λ_hist: "
        f"{best['lambda_hist']}"
    )

    print(
        f"  Current Top-1: "
        f"{best['current_top1']:.2f}%"
    )

    print(
        f"  Graph Top-1: "
        f"{best['graph_top1']:.2f}%"
    )

    print(
        f"  Graph+Historical+Current Top-1: "
        f"{best['combined_top1']:.2f}%"
    )

    print(
        f"  Improvement: "
        f"{best['combined_top1_improvement_pp']:+.2f} pp"
    )

    print(
        f"  Current MRR: "
        f"{best['current_mrr']:.4f}"
    )

    print(
        f"  Combined MRR: "
        f"{best['combined_mrr']:.4f}"
    )

print()
print(f"Results saved to: {output}")
