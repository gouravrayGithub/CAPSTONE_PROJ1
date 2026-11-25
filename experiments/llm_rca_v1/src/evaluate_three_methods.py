
import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "experiments/llm_rca_v1/full_evaluation"

HYBRID_DIR = EVAL_DIR / "results"
LLM_ONLY_DIR = EVAL_DIR / "llm_only_results"
BASELINE_FILE = (
    ROOT / "experiments/llm_rca_v1/baseline_results/"
    "rca_methods_v3_detailed.csv"
)

REPORT_JSON = EVAL_DIR / "three_method_accuracy.json"
REPORT_CSV = EVAL_DIR / "three_method_case_results.csv"


def parse_candidates(value):
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
        raise ValueError("Baseline ranking is not a list")

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


def get_llm_ranking(path):
    data = json.loads(path.read_text())
    ranking = data.get("final_ranking", [])
    return [
        str(item["service"]).strip()
        for item in ranking
        if isinstance(item, dict) and item.get("service")
    ]


def calculate_metrics(ranking, true_root):
    true_root = str(true_root).strip()
    ranking = [str(x).strip() for x in ranking]

    rank = next(
        (i for i, service in enumerate(ranking, 1)
         if service == true_root),
        None,
    )

    return {
        "root_rank": rank,
        "top1": int(rank == 1),
        "top3": int(rank is not None and rank <= 3),
        "top5": int(rank is not None and rank <= 5),
        "rr": 1.0 / rank if rank is not None else 0.0,
    }


def summarize(name, records):
    n = len(records)
    if n == 0:
        return {"method": name, "cases": 0}

    result = {
        "method": name,
        "cases": n,
        "top1_accuracy": sum(r["top1"] for r in records) / n,
        "top3_accuracy": sum(r["top3"] for r in records) / n,
        "top5_accuracy": sum(r["top5"] for r in records) / n,
        "mrr": sum(r["rr"] for r in records) / n,
    }

    print(f"\n{name} (n={n})")
    for metric in ("top1_accuracy", "top3_accuracy",
                   "top5_accuracy", "mrr"):
        value = result[metric]
        print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")

    return result


def main():
    with BASELINE_FILE.open(newline="") as f:
        baseline_rows = list(csv.DictReader(f))

    baseline_by_case = {}
    for row in baseline_rows:
        try:
            observation = float(row["observation_level"])
        except (ValueError, KeyError):
            continue

        if (
            observation == 100
            and row.get("method") == "successor_anomaly"
        ):
            baseline_by_case.setdefault(row["case"].strip(), row)

    hybrid_files = {
        p.name.removesuffix("_hybrid.json"): p
        for p in HYBRID_DIR.glob("*_hybrid.json")
    }
    llm_only_files = {
        p.name.removesuffix("_llm_only.json"): p
        for p in LLM_ONLY_DIR.glob("*_llm_only.json")
    }

    common_cases = sorted(
        set(hybrid_files)
        & set(llm_only_files)
        & set(baseline_by_case)
    )

    print("Saved hybrid rankings:", len(hybrid_files))
    print("Saved LLM-only rankings:", len(llm_only_files))
    print("Cases with all three methods:", len(common_cases))

    all_records = []
    records = {
        "successor_anomaly": [],
        "LLM hybrid": [],
        "LLM-only": [],
    }
    skipped = []

    for case in common_cases:
        baseline_row = baseline_by_case[case]
        true_root = baseline_row.get("true_root", "").strip()

        try:
            baseline_ranking = parse_candidates(
                baseline_row["ranked_candidates"]
            )
            hybrid_ranking = get_llm_ranking(hybrid_files[case])
            llm_only_ranking = get_llm_ranking(llm_only_files[case])

            rankings = {
                "successor_anomaly": baseline_ranking,
                "LLM hybrid": hybrid_ranking,
                "LLM-only": llm_only_ranking,
            }

            if not true_root or any(not r for r in rankings.values()):
                raise ValueError("Missing true root or empty ranking")

            for method, ranking in rankings.items():
                scores = calculate_metrics(ranking, true_root)
                records[method].append(scores)
                all_records.append({
                    "case": case,
                    "method": method,
                    "true_root": true_root,
                    **scores,
                })

        except Exception as exc:
            skipped.append({"case": case, "reason": str(exc)})

    # If any case failed parsing for any method, exclude it from ALL methods.
    skipped_cases = {item["case"] for item in skipped}
    if skipped_cases:
        all_records = [
            r for r in all_records if r["case"] not in skipped_cases
        ]
        for method in records:
            records[method] = [
                r for case, r in zip(
                    [c for c in common_cases if c not in skipped_cases],
                    records[method]
                )
            ]

    summaries = [
        summarize(method, records[method])
        for method in ("successor_anomaly", "LLM hybrid", "LLM-only")
    ]

    print("\nCommon evaluation cases:", len(records["LLM-only"]))
    print("Cases excluded due to parsing/data issues:", len(skipped))

    REPORT_JSON.write_text(json.dumps({
        "observation_level": 100,
        "summaries": summaries,
        "case_results": all_records,
        "skipped": skipped,
        "note": (
            "All methods use the same successfully matched incident cases. "
            "LLM-only refers to the investigator mode; initial candidate "
            "selection may still depend on the baseline."
        ),
    }, indent=2))

    with REPORT_CSV.open("w", newline="") as f:
        fields = [
            "case", "method", "true_root",
            "root_rank", "top1", "top3", "top5", "rr",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    print("\nJSON report:", REPORT_JSON)
    print("Case-level CSV:", REPORT_CSV)


if __name__ == "__main__":
    main()
