import re
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

CASE = Path("data/re2ob_currencyservice_cpu_1")

INPUT_FILE = CASE / "logs.parquet"
OUTPUT_FILE = CASE / "parsed_logs.parquet"


# --------------------------------------------------
# 2. Load logs
# --------------------------------------------------

logs = pd.read_parquet(INPUT_FILE)


# --------------------------------------------------
# 3. Template extraction function
# --------------------------------------------------

def extract_template(message):

    message = str(message)

    # -----------------------------------------
    # UUIDs
    # -----------------------------------------

    message = re.sub(
        r'\b[0-9a-fA-F]{8}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}\b',
        '<UUID>',
        message
    )

    # -----------------------------------------
    # Email
    # -----------------------------------------

    message = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '<EMAIL>',
        message
    )

    # -----------------------------------------
    # Credit card
    # -----------------------------------------

    message = re.sub(
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',
        '<CARD>',
        message
    )

    # -----------------------------------------
    # Product IDs
    # -----------------------------------------

    message = re.sub(
        r'\b[A-Z0-9]{10}\b',
        '<PRODUCT_ID>',
        message
    )

    # -----------------------------------------
    # Numeric values
    # -----------------------------------------

    message = re.sub(
        r'\b\d+(?:\.\d+)?\b',
        '<NUM>',
        message
    )

    # -----------------------------------------
    # Extra spaces
    # -----------------------------------------

    message = re.sub(
        r'\s+',
        ' ',
        message
    ).strip()

    return message
    """
    Convert a raw log message into a normalized
    event template by replacing dynamic values.
    """

    message = str(message)

    # UUID
    message = re.sub(
        r'\b[0-9a-fA-F]{8}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}\b',
        '<UUID>',
        message
    )

    # Email addresses
    message = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '<EMAIL>',
        message
    )

    # Credit card numbers
    message = re.sub(
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',
        '<CARD>',
        message
    )

    # Product IDs such as 9SIQT8TOJO
    message = re.sub(
        r'\b[A-Z0-9]{10}\b',
        '<PRODUCT_ID>',
        message
    )

    # Transaction IDs
    message = re.sub(
        r'\b[0-9a-fA-F]{8}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}\b',
        '<TRANSACTION_ID>',
        message
    )

    # Numbers
    message = re.sub(
        r'\b\d+(?:\.\d+)?\b',
        '<NUM>',
        message
    )

    # Remove extra spaces
    message = re.sub(
        r'\s+',
        ' ',
        message
    ).strip()

    return message


# --------------------------------------------------
# 4. Apply template extraction
# --------------------------------------------------

logs["template"] = logs["message"].apply(
    extract_template
)


# --------------------------------------------------
# 5. Create event ID
# --------------------------------------------------

template_to_id = {}

next_id = 1

event_ids = []

for template in logs["template"]:

    if template not in template_to_id:

        template_to_id[template] = next_id
        next_id += 1

    event_ids.append(
        template_to_id[template]
    )


logs["event_id"] = event_ids


# --------------------------------------------------
# 6. Save parsed logs
# --------------------------------------------------

logs.to_parquet(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 7. Print statistics
# --------------------------------------------------

print("\n===== PARSING COMPLETE =====")

print("Total logs:", len(logs))

print(
    "Unique templates:",
    logs["template"].nunique()
)

print(
    "Unique event IDs:",
    logs["event_id"].nunique()
)


# --------------------------------------------------
# 8. Display examples
# --------------------------------------------------

print("\n===== RAW → TEMPLATE =====")

examples = logs[
    ["message", "template", "event_id"]
].drop_duplicates(
    subset=["template"]
).head(30)


for _, row in examples.iterrows():

    print("\nRAW:")
    print(row["message"])

    print("TEMPLATE:")
    print(row["template"])

    print("EVENT ID:")
    print(row["event_id"])