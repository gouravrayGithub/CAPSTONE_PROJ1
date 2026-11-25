from pathlib import Path
import pickle
import pandas as pd
import networkx as nx


DATA_DIR = Path("data")


def parse_case_name(case_name):
    """
    Example:
    re2ob_checkoutservice_cpu_1

    Returns:
        root_service = checkoutservice
        fault_type   = cpu
        repetition   = 1
    """

    parts = case_name.replace("re2ob_", "").split("_")

    repetition = int(parts[-1])
    fault_type = parts[-2]

    root_service = "_".join(parts[:-2])

    return root_service, fault_type, repetition


def graph_statistics(G, root_service):

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "graph_density": nx.density(G),
    }

    # ---------------------------------------------------------
    # ROOT SERVICE FEATURES
    # ---------------------------------------------------------

    if root_service in G.nodes:

        node = G.nodes[root_service]

        stats["root_log_before"] = node.get(
            "log_count_before", 0
        )

        stats["root_log_after"] = node.get(
            "log_count_after", 0
        )

        stats["root_trace_before"] = node.get(
            "trace_count_before", 0
        )

        stats["root_trace_after"] = node.get(
            "trace_count_after", 0
        )

        stats["root_median_before_ms"] = node.get(
            "median_latency_before_ms", 0
        )

        stats["root_median_after_ms"] = node.get(
            "median_latency_after_ms", 0
        )

        stats["root_p95_before_ms"] = node.get(
            "p95_latency_before_ms", 0
        )

        stats["root_p95_after_ms"] = node.get(
            "p95_latency_after_ms", 0
        )

        # -----------------------------------------------------
        # RELATIVE CHANGES
        # -----------------------------------------------------

        if stats["root_log_before"] > 0:
            stats["root_log_change_pct"] = (
                (
                    stats["root_log_after"]
                    - stats["root_log_before"]
                )
                / stats["root_log_before"]
                * 100
            )
        else:
            stats["root_log_change_pct"] = 0

        if stats["root_trace_before"] > 0:
            stats["root_trace_change_pct"] = (
                (
                    stats["root_trace_after"]
                    - stats["root_trace_before"]
                )
                / stats["root_trace_before"]
                * 100
            )
        else:
            stats["root_trace_change_pct"] = 0

        if stats["root_median_before_ms"] > 0:
            stats["root_median_change_pct"] = (
                (
                    stats["root_median_after_ms"]
                    - stats["root_median_before_ms"]
                )
                / stats["root_median_before_ms"]
                * 100
            )
        else:
            stats["root_median_change_pct"] = 0

        if stats["root_p95_before_ms"] > 0:
            stats["root_p95_change_pct"] = (
                (
                    stats["root_p95_after_ms"]
                    - stats["root_p95_before_ms"]
                )
                / stats["root_p95_before_ms"]
                * 100
            )
        else:
            stats["root_p95_change_pct"] = 0

        # -----------------------------------------------------
        # GRAPH POSITION
        # -----------------------------------------------------

        stats["root_in_degree"] = G.in_degree(
            root_service
        )

        stats["root_out_degree"] = G.out_degree(
            root_service
        )

    else:

        for key in [
            "root_log_before",
            "root_log_after",
            "root_trace_before",
            "root_trace_after",
            "root_median_before_ms",
            "root_median_after_ms",
            "root_p95_before_ms",
            "root_p95_after_ms",
            "root_log_change_pct",
            "root_trace_change_pct",
            "root_median_change_pct",
            "root_p95_change_pct",
            "root_in_degree",
            "root_out_degree",
        ]:
            stats[key] = 0

    return stats


def main():

    rows = []

    graph_files = sorted(
        DATA_DIR.glob(
            "re2ob_*/historical_incident_graph.gpickle"
        )
    )

    print("=" * 70)
    print("BUILDING HISTORICAL INCIDENT INDEX")
    print("=" * 70)

    print(
        "Graphs found:",
        len(graph_files)
    )

    for graph_file in graph_files:

        case_dir = graph_file.parent
        case_name = case_dir.name

        root_service, fault_type, repetition = (
            parse_case_name(case_name)
        )

        with open(graph_file, "rb") as f:
            G = pickle.load(f)

        stats = graph_statistics(
            G,
            root_service
        )

        row = {
            "case": case_name,
            "root_service": root_service,
            "fault_type": fault_type,
            "repetition": repetition,
        }

        row.update(stats)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort for easier inspection
    df = df.sort_values(
        [
            "root_service",
            "fault_type",
            "repetition",
        ]
    ).reset_index(drop=True)

    output = (
        DATA_DIR
        / "historical_incident_index.parquet"
    )

    df.to_parquet(
        output,
        index=False
    )

    print()
    print("Index created:")
    print(output)

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Root services:")
    print(
        df["root_service"].value_counts()
    )

    print()
    print("Fault types:")
    print(
        df["fault_type"].value_counts()
    )

    print()
    print("First 10 rows:")
    print(
        df.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
