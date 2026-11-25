from pathlib import Path
import csv
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
LLM_DIR = ROOT / "experiments" / "llm_rca_v1"
GRAPH_DIR = ROOT / "experiments" / "successor_v1" / "graphs"
RUN_DIR = LLM_DIR / "full_evaluation"
PREP_DIR = RUN_DIR / "prepared"
RESULT_DIR = RUN_DIR / "results"
LOG_DIR = RUN_DIR / "logs"

for folder in (PREP_DIR, RESULT_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

SUMMARY = RUN_DIR / "batch_summary.csv"

cases = sorted(
    p.name for p in GRAPH_DIR.glob("re2ob_*")
    if p.is_dir()
    and (p / "partial_graphs" / "partial_100.gpickle").exists()
)

if not cases:
    raise SystemExit(f"No incident cases found under {GRAPH_DIR}")

print(f"Discovered {len(cases)} cases with a 100% graph.", flush=True)
print("Mode: hybrid | Observation: 100% | Provider: local", flush=True)
print("Results:", RESULT_DIR, flush=True)

fields = ["case", "status", "seconds", "message"]
rows = []

for i, case in enumerate(cases, start=1):
    prepared = PREP_DIR / f"{case}.json"
    result = RESULT_DIR / f"{case}_hybrid.json"
    log = LOG_DIR / f"{case}.log"
    started = time.time()
    status = "failed"
    message = ""

    print(f"\n[{i}/{len(cases)}] {case}", flush=True)

    # Resume safely when a previous run already produced valid JSON.
    if result.exists():
        try:
            previous = json.loads(result.read_text())
            if isinstance(previous, dict) and previous.get("final_ranking"):
                status = "already_completed"
                message = "Valid result exists; skipped"
                elapsed = round(time.time() - started, 2)
                print(message, flush=True)
                rows.append({
                    "case": case, "status": status,
                    "seconds": elapsed, "message": message
                })
                continue
        except (json.JSONDecodeError, OSError):
            pass

    try:
        prep_cmd = [
            "python", "src/prepare_case.py",
            "--case", case,
            "--observation-level", "100",
            "--method", "successor_anomaly",
            "--output", str(prepared.relative_to(LLM_DIR)),
        ]
        prep = subprocess.run(
            prep_cmd, cwd=LLM_DIR, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=120
        )

        if prep.returncode != 0 or not prepared.exists():
            message = "Preparation failed:\n" + prep.stdout[-2500:]
            log.write_text(message)
            raise RuntimeError(message)

        run_cmd = [
            "python", "src/run_investigation.py",
            "--input", str(prepared.relative_to(LLM_DIR)),
            "--output", str(result.relative_to(LLM_DIR)),
            "--provider", "local",
            "--mode", "hybrid",
            "--top-k", "3",
            "--threshold", "0.75",
            "--batch-size", "2",
            "--max-investigations", "10",
        ]

        run = subprocess.run(
            run_cmd, cwd=LLM_DIR, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=900
        )
        log.write_text(
            "PREPARATION\n" + prep.stdout +
            "\nINVESTIGATION\n" + run.stdout
        )

        if run.returncode != 0:
            message = run.stdout[-2500:]
        elif not result.exists():
            message = "Investigation returned success but no output file"
        else:
            try:
                output = json.loads(result.read_text())
                if not isinstance(output, dict) or not output.get("final_ranking"):
                    message = "Output JSON has no final_ranking"
                else:
                    status = "success"
                    message = "Completed"
            except (json.JSONDecodeError, OSError) as exc:
                message = f"Invalid result JSON: {exc}"

    except subprocess.TimeoutExpired:
        message = "Timed out; rerun later to retry"
        log.write_text(message)
    except Exception as exc:
        if not message:
            message = f"{type(exc).__name__}: {exc}"
        if not log.exists():
            log.write_text(message)

    elapsed = round(time.time() - started, 2)
    rows.append({
        "case": case, "status": status,
        "seconds": elapsed, "message": message
    })

    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Status: {status} | {elapsed}s", flush=True)
    if status == "failed":
        print(message[-700:], flush=True)

# Write a final summary even if every case was skipped.
with SUMMARY.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

counts = {}
for row in rows:
    counts[row["status"]] = counts.get(row["status"], 0) + 1

print("\n===== BATCH SUMMARY =====")
print(f"Discovered: {len(cases)}")
for status, count in sorted(counts.items()):
    print(f"{status}: {count}")
print(f"Summary CSV: {SUMMARY}")
print(f"Individual logs: {LOG_DIR}")
print("This is a run summary, not yet an RCA accuracy evaluation.")
