from pathlib import Path
import pandas as pd
import numpy as np
import pickle


DATA_DIR = Path("data")


def percentile(series, q):

    series = series.dropna()

    if len(series) == 0:
        return 0.0

    return float(np.percentile(series, q))


def process_case(case_dir):

    graph_file = (
        case_dir
        / "historical_incident_graph.gpickle"
    )

    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    injection_time = G.graph["injection_time"]

    logs = pd.read_parquet(
        case_dir / "logs.parquet"
    )

    traces = pd.read_parquet(
        case_dir / "traces.parquet"
    )

    traces = traces.drop_duplicates()

    # ---------------------------------------------------------
    # NORMALIZE
    # ---------------------------------------------------------

    service_map = {
        "frontendservice": "frontend",
        "currencyservice": "currencyservice",
        "cartservice": "cartservice",
        "recommendationservice": "recommendationservice",
        "adservice": "adservice",
        "shippingservice": "shippingservice",
        "checkoutservice": "checkoutservice",
        "paymentservice": "paymentservice",
        "emailservice": "emailservice",
        "redis": "redis",
        "productcatalogservice": "productcatalogservice",
    }

    logs["service"] = logs[
        "container_name"
    ].map(service_map)

    traces["service"] = traces[
        "serviceName"
    ].map(service_map)

    traces["timestamp_sec"] = (
        traces["startTimeMillis"] // 1000
    )

    traces["duration_ms"] = (
        traces["duration"].astype(float)
        / 1000.0
    )

    # ---------------------------------------------------------
    # SPLIT
    # ---------------------------------------------------------

    before_logs = logs[
        logs["timestamp"] < injection_time
    ]

    after_logs = logs[
        logs["timestamp"] >= injection_time
    ]

    before_traces = traces[
        traces["timestamp_sec"] < injection_time
    ]

    after_traces = traces[
        traces["timestamp_sec"] >= injection_time
    ]

    # ---------------------------------------------------------
    # SERVICES
    # ---------------------------------------------------------

    services = sorted(
        set(logs["service"].dropna())
        |
        set(traces["service"].dropna())
    )

    rows = []

    for service in services:

        lb = before_logs[
            before_logs["service"] == service
        ]

        la = after_logs[
            after_logs["service"] == service
        ]

        tb = before_traces[
            before_traces["service"] == service
        ]

        ta = after_traces[
            after_traces["service"] == service
        ]

        # -----------------------------------------------------
        # LATENCY
        # -----------------------------------------------------

        median_before = (
            float(tb["duration_ms"].median())
            if len(tb)
            else 0.0
        )

        median_after = (
            float(ta["duration_ms"].median())
            if len(ta)
            else 0.0
        )

        p95_before = percentile(
            tb["duration_ms"],
            95
        )

        p95_after = percentile(
            ta["duration_ms"],
            95
        )

        # -----------------------------------------------------
        # RELATIVE CHANGES
        # -----------------------------------------------------

        log_before = len(lb)
        log_after = len(la)

        trace_before = len(tb)
        trace_after = len(ta)

        log_change = (
            ((log_after - log_before) / log_before)
            * 100
            if log_before > 0
            else 0
        )

        trace_change = (
            ((trace_after - trace_before) / trace_before)
            * 100
            if trace_before > 0
            else 0
        )

        median_change = (
            ((median_after - median_before)
             / median_before)
            * 100
            if median_before > 0
            else 0
        )

        p95_change = (
            ((p95_after - p95_before)
             / p95_before)
            * 100
            if p95_before > 0
            else 0
        )

        # -----------------------------------------------------
        # GRAPH POSITION
        # -----------------------------------------------------

        in_degree = G.in_degree(service)

        out_degree = G.out_degree(service)

        rows.append({

            "case": case_dir.name,

            "service": service,

            "log_before": log_before,
            "log_after": log_after,

            "trace_before": trace_before,
            "trace_after": trace_after,

            "median_before_ms": median_before,
            "median_after_ms": median_after,

            "p95_before_ms": p95_before,
            "p95_after_ms": p95_after,

            "log_change_pct": log_change,
            "trace_change_pct": trace_change,

            "median_change_pct": median_change,
            "p95_change_pct": p95_change,

            "in_degree": in_degree,
            "out_degree": out_degree,
        })

    return rows


def main():

    all_rows = []

    cases = sorted(
        DATA_DIR.glob("re2ob_*")
    )

    print(
        "Processing",
        len(cases),
        "cases..."
    )

    for i, case_dir in enumerate(
        cases,
        start=1
    ):

        print(
            f"[{i}/{len(cases)}] "
            f"{case_dir.name}"
        )

        try:

            rows = process_case(
                case_dir
            )

            all_rows.extend(rows)

        except Exception as e:

            print(
                "ERROR:",
                e
            )

    df = pd.DataFrame(
        all_rows
    )

    output = (
        DATA_DIR
        / "incident_service_features.parquet"
    )

    df.to_parquet(
        output,
        index=False
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "Output:",
        output
    )

    print()
    print(
        df.head(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()