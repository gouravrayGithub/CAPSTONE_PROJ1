
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "successor_v2" / "results"
    / "temporal_features"
)

WINDOW_SECONDS = 30


def read_injection_time(case_dir):
    text = (case_dir / "inject_time.txt").read_text().strip()
    return int(re.findall(r"\d+", text)[0])


def normalize_service(name):
    mapping = {
        "frontendservice": "frontend",
    }
    return mapping.get(name, name)


def build_case(case_dir):
    injection_time = read_injection_time(case_dir)

    logs = pd.read_parquet(case_dir / "logs.parquet")
    traces = pd.read_parquet(case_dir / "traces.parquet")

    logs["service"] = logs["container_name"].map(normalize_service)
    logs["relative_time"] = logs["timestamp"] - injection_time
    logs["window_start_sec"] = (
        np.floor(logs["relative_time"] / WINDOW_SECONDS)
        * WINDOW_SECONDS
    ).astype(int)

    traces["service"] = traces["serviceName"].map(normalize_service)
    traces["timestamp_sec"] = traces["startTimeMillis"] // 1000
    traces["relative_time"] = traces["timestamp_sec"] - injection_time
    traces["window_start_sec"] = (
        np.floor(traces["relative_time"] / WINDOW_SECONDS)
        * WINDOW_SECONDS
    ).astype(int)

    traces["duration_ms"] = traces["duration"].astype(float) / 1000.0

    log_counts = (
        logs.groupby(["window_start_sec", "service"])
        .size()
        .rename("log_count")
    )

    trace_groups = traces.groupby(["window_start_sec", "service"])

    trace_counts = trace_groups.size().rename("trace_count")
    median_latency = trace_groups["duration_ms"].median().rename(
        "median_latency_ms"
    )
    p95_latency = trace_groups["duration_ms"].quantile(0.95).rename(
        "p95_latency_ms"
    )

    features = pd.concat(
        [log_counts, trace_counts, median_latency, p95_latency],
        axis=1,
    ).fillna(0).reset_index()

    features["case"] = case_dir.name
    features["injection_time"] = injection_time
    features["period"] = np.where(
        features["window_start_sec"] < 0,
        "before",
        "after",
    )

    # Estimate baseline separately for each service and metric.
    metrics = [
        "log_count",
        "trace_count",
        "median_latency_ms",
        "p95_latency_ms",
    ]

    features["anomaly_score"] = 0.0

    for service in features["service"].unique():
        service_mask = features["service"] == service
        baseline_mask = service_mask & (features["period"] == "before")

        total_score = pd.Series(0.0, index=features.index)
        contributing_metrics = 0

        for metric in metrics:
            baseline = features.loc[baseline_mask, metric]

            if len(baseline) < 3:
                continue

            mean = baseline.mean()
            std = baseline.std(ddof=1)

            if not np.isfinite(std) or std < 1e-9:
                continue

            z = (
                features.loc[service_mask, metric] - mean
            ) / std

            # Absolute deviation detects both unexpected increases
            # and unexpected decreases.
            total_score.loc[service_mask] += z.abs().clip(upper=10)
            contributing_metrics += 1

        if contributing_metrics:
            features.loc[service_mask, "anomaly_score"] = (
                total_score.loc[service_mask] / contributing_metrics
            )

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        path for path in DATA_DIR.glob("re2ob_*")
        if path.is_dir()
        and (path / "logs.parquet").exists()
        and (path / "traces.parquet").exists()
        and (path / "inject_time.txt").exists()
    )

    all_features = []
    failures = []

    for index, case_dir in enumerate(case_dirs, start=1):
        try:
            frame = build_case(case_dir)
            all_features.append(frame)
            print(f"[{index}/{len(case_dirs)}] {case_dir.name}")
        except Exception as exc:
            failures.append((case_dir.name, str(exc)))
            print(f"[ERROR] {case_dir.name}: {exc}")

    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        output_file = OUTPUT_DIR / "service_temporal_features.csv"
        result.to_csv(output_file, index=False)

        print("\n===== TEMPORAL FEATURE BUILD =====")
        print("Cases found:", len(case_dirs))
        print("Cases processed:", len(all_features))
        print("Failures:", len(failures))
        print("Rows:", len(result))
        print("Output:", output_file)

    if failures:
        for case, error in failures:
            print(f"FAILED: {case}: {error}")


if __name__ == "__main__":
    main()
