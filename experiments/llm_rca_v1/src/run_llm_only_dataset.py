import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LLM_DIR = ROOT / "experiments" / "llm_rca_v1"

PREP_DIR = LLM_DIR / "full_evaluation" / "prepared"
RESULT_DIR = LLM_DIR / "full_evaluation" / "llm_only_results"
LOG_DIR = LLM_DIR / "full_evaluation" / "llm_only_logs"
SUMMARY = LLM_DIR / "full_evaluation" / "llm_only_summary.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

prepared_files = sorted(PREP_DIR.glob("*.json"))
rows = []

if not prepared_files:
    raise SystemExit(f"No prepared bundles found in {PREP_DIR}")

print(f"Prepared incident bundles found: {len(prepared_files)}")

for i, prepared in enumerate(prepared_files, start=1):
    case = prepared.stem
    output = RESULT_DIR / f"{case}_llm_only.json"
    log_file = LOG_DIR / f"{case}.log"

    if output.exists():
        try:
            existing = json.loads(output.read_text())
            if existing.get("final_ranking"):
                print(f"[{i}/{len(prepared_files)}] {case}: already completed")
                rows.append({
                    "case": case,
                    "status": "already_completed",
                    "seconds": 0,
                    "message": "",
                })
                continue
        except Exception:
            pass

    command = [
        sys.executable,
        "src/run_investigation.py",
        "--input", str(prepared.relative_to(LLM_DIR)),
        "--output", str(output.relative_to(LLM_DIR)),
        "--provider", "local",
        "--mode", "llm-only",
        "--top-k", "3",
        "--threshold", "0.75",
        "--batch-size", "2",
        "--max-investigations", "10",
    ]

    print(f"\n[{i}/{len(prepared_files)}] {case}")
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=LLM_DIR,
            capture_output=True,
            text=True,
            timeout=1200,
        )

        elapsed = round(time.time() - start, 2)
        log_file.write_text(
            "STDOUT:\n" + result.stdout +
            "\nSTDERR:\n" + result.stderr
        )

        if result.returncode == 0 and output.exists():
            try:
                data = json.loads(output.read_text())
                if data.get("final_ranking"):
                    status = "success"
                    message = ""
                else:
                    status = "failed"
                    message = "Output has no final_ranking"
            except Exception as exc:
                status = "failed"
                message = f"Invalid output JSON: {exc}"
        else:
            status = "failed"
            message = (result.stderr or result.stdout)[-1500:]

    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start, 2)
        status = "failed"
        message = "Investigation timed out after 1200 seconds"
        log_file.write_text(message)

    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        status = "failed"
        message = str(exc)
        log_file.write_text(message)

    rows.append({
        "case": case,
        "status": status,
        "seconds": elapsed,
        "message": message,
    })

    print(f"Status: {status} | {elapsed}s")

    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case", "status", "seconds", "message"]
        )
        writer.writeheader()
        writer.writerows(rows)

print("\n===== LLM-ONLY SUMMARY =====")
for status in ("already_completed", "success", "failed"):
    count = sum(r["status"] == status for r in rows)
    print(f"{status}: {count}")

print(f"Summary: {SUMMARY}")
print(f"Results: {RESULT_DIR}")
print(f"Logs: {LOG_DIR}")