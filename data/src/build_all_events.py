from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path("data")


def normalize_message(msg):
    msg = str(msg)

    # UUID
    msg = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "<UUID>",
        msg
    )

    # Email
    msg = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "<EMAIL>",
        msg
    )

    # Card-like numbers
    msg = re.sub(
        r"\b(?:\d[ -]*?){13,19}\b",
        "<CARD>",
        msg
    )

    # Product IDs: 10-character alphanumeric strings
    msg = re.sub(
        r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{10}\b",
        "<PRODUCT_ID>",
        msg
    )

    # Numbers
    msg = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", msg)

    # Normalize whitespace
    msg = re.sub(r"\s+", " ", msg).strip()

    return msg


def process_case(case_dir):

    logs_file = case_dir / "logs.parquet"

    if not logs_file.exists():
        return False, "logs.parquet missing"

    logs = pd.read_parquet(logs_file)

    if logs.empty:
        return False, "empty logs"

    logs["timestamp"] = pd.to_numeric(logs["timestamp"], errors="coerce")
    logs = logs.dropna(subset=["timestamp"])

    logs["service"] = logs["container_name"].astype(str)
    logs["template"] = logs["message"].map(normalize_message)

    # Event type = normalized log template
    logs["event_id"] = pd.factorize(logs["template"])[0]

    # One row = one event occurrence
    events = logs[
        [
            "timestamp",
            "service",
            "event_id",
            "template",
            "message",
        ]
    ].copy()

    events["event_instance_id"] = range(len(events))

    events = events[
        [
            "event_instance_id",
            "timestamp",
            "service",
            "event_id",
            "template",
            "message",
        ]
    ]

    output = case_dir / "events.parquet"
    events.to_parquet(output, index=False)

    return True, len(events)


def main():

    cases = sorted(DATA_DIR.glob("re2ob_*"))

    success = 0
    failed = 0

    print("=" * 70)
    print("BUILDING EVENTS FOR ALL RE2-OB CASES")
    print("=" * 70)

    for case_dir in cases:

        if not case_dir.is_dir():
            continue

        try:
            ok, result = process_case(case_dir)

            if ok:
                success += 1
                print(f"[OK]   {case_dir.name:45s} {result:,} events")
            else:
                failed += 1
                print(f"[FAIL] {case_dir.name:45s} {result}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {case_dir.name:45s} {e}")

    print()
    print("=" * 70)
    print(f"Successful: {success}")
    print(f"Failed:     {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
