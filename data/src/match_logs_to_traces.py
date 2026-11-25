import pandas as pd
import re
from pathlib import Path

CASE = Path("data/re2ob_currencyservice_cpu_1")

logs = pd.read_parquet(CASE / "logs.parquet")
traces = pd.read_parquet(CASE / "traces.parquet")

# --------------------------------------------------
# SERVICE NAME NORMALIZATION
# --------------------------------------------------

SERVICE_MAP = {
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
}

traces["log_service"] = traces["serviceName"].map(SERVICE_MAP)

# Remove traces whose service cannot be mapped
traces = traces.dropna(subset=["log_service"]).copy()

# Convert trace start time to seconds
traces["trace_seconds"] = traces["startTimeMillis"] // 1000


# --------------------------------------------------
# OPERATION NORMALIZATION
# --------------------------------------------------

def normalize_operation(operation):
    """
    Convert trace operation into a simpler operation name.
    Example:
    hipstershop.CartService/GetCart
    -> getcart
    """

    operation = str(operation).lower()

    # Keep only final part after /
    operation = operation.split("/")[-1]

    # Remove common separators
    operation = re.sub(r"[^a-z0-9]", "", operation)

    return operation


traces["normalized_operation"] = traces["operationName"].apply(
    normalize_operation
)


# --------------------------------------------------
# LOG MESSAGE → POSSIBLE OPERATION
# --------------------------------------------------

def extract_log_operation(message):

    message = str(message).lower()

    patterns = {
        "getcart": "getcart",
        "listrecommendations": "listrecommendations",
        "getads": "getads",
        "placeorder": "placeorder",
        "getquote": "getquote",
        "convert": "convert",
        "getproduct": "getproduct",
        "listproducts": "listproducts",
        "charge": "charge",
        "sendorderconfirmation": "sendorderconfirmation",
    }

    for keyword, operation in patterns.items():

        if keyword in message:
            return operation

    return None


logs["normalized_operation"] = logs["message"].apply(
    extract_log_operation
)


# --------------------------------------------------
# FIND CANDIDATE SPANS
# --------------------------------------------------

TIME_WINDOW = 2

results = []

sample_logs = logs[
    logs["normalized_operation"].notna()
].head(100)


for _, log in sample_logs.iterrows():

    service = log["container_name"]
    timestamp = log["timestamp"]
    operation = log["normalized_operation"]

    candidates = traces[
        (traces["log_service"] == service)
        &
        (
            abs(
                traces["trace_seconds"] - timestamp
            ) <= TIME_WINDOW
        )
        &
        (
            traces["normalized_operation"] == operation
        )
    ]

    for _, span in candidates.head(5).iterrows():

        results.append({
            "log_timestamp": timestamp,
            "log_service": service,
            "log_message": log["message"],

            "trace_id": span["traceID"],
            "span_id": span["spanID"],
            "trace_service": span["serviceName"],
            "operation": span["operationName"],
            "trace_timestamp": span["trace_seconds"],

            "time_difference":
                abs(
                    span["trace_seconds"] - timestamp
                ),

            "parent_span":
                span["parentSpanID"],
        })


matches = pd.DataFrame(results)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n===== LOG → TRACE MATCHING =====")

print(
    "Logs considered:",
    len(sample_logs)
)

print(
    "Candidate matches:",
    len(matches)
)

if len(matches) > 0:

    print(
        matches[
            [
                "log_timestamp",
                "log_service",
                "log_message",
                "operation",
                "trace_timestamp",
                "time_difference",
                "parent_span",
            ]
        ].head(30).to_string(index=False)
    )

else:

    print("No candidate matches found.")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output = CASE / "log_trace_candidates.parquet"

matches.to_parquet(
    output,
    index=False
)

print("\nSaved:", output)