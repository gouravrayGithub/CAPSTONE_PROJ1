
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RESULT_DIR = (
    ROOT / "experiments/llm_rca_v1/full_evaluation/results"
)
BASELINE_FILE = (
    ROOT
    / "experiments/llm_rca_v1/baseline_results/rca_methods_v3_detailed.csv"
)
OUTPUT_FILE = (
    ROOT / "experiments/llm_rca_v1/full_evaluation/accuracy_report.json"
)


def parse_candidates(value):
    """Read a serialized candidate ranking."""
    if isinstance(value, list):
        data = value
    else:
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            data = ast.literal_eval(value)

    if isinstance(data, dict):
        data = data.get("ranked_candidates", data.get("ranking", []))

    if not isinstance(data, list):
        raise ValueError("Candidate ranking is not a list")

    candidates = []
    for item in data:
        if isinstance(item, str):
            candidates.append(item)
        elif isinstance(item, dict):
            for key in ("service", "service_name", "candidate", "name", "node"):
                if item.get(key):
                    candidates.append(str(item[key]))
                    break

    return candidates


def metrics(ranked, truth):
    ranked = [str(x).strip() for x in ranked]
    truth = str(truth).strip()

    # Use the first occurrence if a service appears more than once.
    rank = next(
        (i for i, service in enumerate(ranked, start=1)
         if service == truth),
        None,
    )

    return {
        "top1": int(rank == 1),
        "top3": int(rank is not None and rank <= 3),
        "top5": int(rank is not None and rank <= 5),
        "rr": 1.0 / rank if rank is not None else 0.0,
        "root_rank": rank,
    }


def summarize(records, label):
    if not records:
        print(f"\n{label}: NO MATCHED CASES")
        return None

    n = len(records)
    result = {
        "method": label,
        "cases": n,
        "top1_accuracy": sum(r["top1"] for r in records) / n,
        "top3_accuracy": sum(r["top3"] for r in records) / n,
        "top5_accuracy": sum(r["top5"] for r in records) / n,
        "mrr": sum(r["rr"] for r in records) / n,
    }

    print(f"\n{label} (n={n})")
    for key in ("top1_accuracy", "top3_accuracy",
                "top5_accuracy", "mrr"):
        print(f"  {key}: {result[key]:.4f} ({result[key] * 100:.2f}%)")

    return result


def main():
    # Load baseline ground truth and baseline rankings.
    with BASELINE_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    baseline_by_case = {}
    for row in rows:
        try:
            observation = float(row["observation_level"])
        except (ValueError, KeyError):
            continue

        if observation != 100 or row.get("method") != "successor_anomaly":
            continue

        case = row["case"].strip()
        if case not in baseline_by_case:
            baseline_by_case[case] = row

    llm_records = []
    baseline_records = []
    skipped = []

    for path in sorted(RESULT_DIR.glob("*_hybrid.json")):
        case = path.name.removesuffix("_hybrid.json")

        if case not in baseline_by_case:
            skipped.append((case, "No matching 100% successor_anomaly baseline"))
            continue

        row = baseline_by_case[case]
        truth = row.get("true_root", "").strip()

        if not truth:
            skipped.append((case, "Missing true_root"))
            continue

        try:
            data = json.loads(path.read_text())
            llm_ranking = [
                item["service"]
                for item in data.get("final_ranking", [])
                if isinstance(item, dict) and item.get("service")
            ]

            baseline_ranking = parse_candidates(row["ranked_candidates"])

            if not llm_ranking:
                raise ValueError("LLM ranking is empty")
            if not baseline_ranking:
                raise ValueError("Baseline ranking is empty")

            llm_records.append({
                "case": case,
                **metrics(llm_ranking, truth),
            })
            baseline_records.append({
                "case": case,
                **metrics(baseline_ranking, truth),
            })

        except Exception as exc:
            skipped.append((case, str(exc)))

    # Evaluate both methods on exactly the same cases.
    llm_summary = summarize(llm_records, "LLM hybrid")
    baseline_summary = summarize(baseline_records, "successor_anomaly baseline")

    print(f"\nSaved LLM rankings: {len(list(RESULT_DIR.glob('*_hybrid.json')))}")
    print(f"Matched and evaluated cases: {len(llm_records)}")
    print(f"Skipped cases: {len(skipped)}")

    if skipped:
        print("\nFirst skipped cases:")
        for case, reason in skipped[:10]:
            print(f"  {case}: {reason}")

    report = {
        "observation_level": 100,
        "llm": llm_summary,
        "baseline": baseline_summary,
        "llm_case_results": llm_records,
        "baseline_case_results": baseline_records,
        "skipped": [
            {"case": case, "reason": reason}
            for case, reason in skipped
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
