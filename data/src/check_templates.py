import pandas as pd
from pathlib import Path


CASE = Path("data/re2ob_currencyservice_cpu_1")

logs = pd.read_parquet(
    CASE / "parsed_logs.parquet"
)


print("\n===== TOTAL TEMPLATES =====")

print(
    logs["template"].nunique()
)


print("\n===== MOST COMMON TEMPLATES =====")

template_counts = (
    logs["template"]
    .value_counts()
    .head(30)
)


for template, count in template_counts.items():

    print(
        count,
        "|",
        template
    )