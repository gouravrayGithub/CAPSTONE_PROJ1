
#!/usr/bin/env python3
"""Run graph-guided root-cause investigation using mock, local Ollama, or OpenAI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from graph_guided_investigation import graph_guided_investigation
from investigator import (
    LocalOllamaInvestigator,
    MockInvestigator,
    OpenAIInvestigator,
)


def load_case(input_path: Path) -> dict[str, Any]:
    """Load and validate a prepared RCA case."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    try:
        case = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {input_path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(case, dict):
        raise ValueError("Input JSON must contain an object.")

    required = ("initial_ranking", "graph", "diagnostic_bundles")
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"Input JSON is missing required keys: {missing}")

    if not isinstance(case["initial_ranking"], list) or not case["initial_ranking"]:
        raise ValueError("'initial_ranking' must be a non-empty list.")

    if not isinstance(case["graph"], dict):
        raise ValueError("'graph' must be a JSON object.")

    if not isinstance(case["diagnostic_bundles"], dict):
        raise ValueError("'diagnostic_bundles' must be a JSON object.")

    return case


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run graph-guided root-cause investigation."
    )

    parser.add_argument(
        "--input",
        default="examples/sample_case.json",
        help="Path to the prepared case JSON.",
    )
    parser.add_argument(
        "--output",
        default="results/sample_investigation.json",
        help="Path for the output JSON.",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "local", "openai"),
        default="mock",
        help="Investigator backend: mock, local Ollama, or OpenAI.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "llm-only"),
        default="hybrid",
        help=(
            "hybrid passes the baseline ranking to the investigator; "
            "llm-only hides that ranking from the investigator."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-investigations", type=int, default=30)

    return parser


def create_investigator(provider: str):
    """Create the requested investigator backend."""
    if provider == "local":
        return LocalOllamaInvestigator()

    if provider == "openai":
        return OpenAIInvestigator()

    return MockInvestigator()


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.mode == "llm-only" and args.provider == "mock":
        parser.error(
            "--mode llm-only requires --provider local or --provider openai."
        )

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.max_investigations < 1:
        parser.error("--max-investigations must be at least 1.")

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        case = load_case(input_path)
        investigator = create_investigator(args.provider)

        def selected_investigator(
            service: str,
            bundle: dict[str, Any],
            ranking: list[dict[str, Any]],
        ) -> dict[str, Any]:
            
            if args.mode == "llm-only":
                return investigator(service, bundle, None)

            return investigator(service, bundle, ranking)

        result = graph_guided_investigation(
            initial_ranking=case["initial_ranking"],
            graph=case["graph"],
            diagnostic_bundles=case["diagnostic_bundles"],
            investigator=selected_investigator,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_investigations=args.max_investigations,
        )

        # Retain only the opaque case identifier; do not add ground-truth labels.
        result["case_id"] = case.get("case_id", input_path.stem)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nInvestigation complete")
    print(f"Provider: {args.provider}")
    print(f"Mode: {args.mode}")
    print(f"Investigated services: {result['counts']['investigated_services']}")
    print(f"Output: {output_path.resolve()}")
    print("\nTop candidates:")

    for item in result["final_ranking"][:5]:
        print(
            f"  {item['service']:<32} "
            f"confidence={item['confidence']:.3f} "
            f"causal={item['causal_confidence']:.3f}"
        )

    if args.provider == "mock":
        print(
            "\nNOTE: Mock results are smoke tests, not LLM results "
            "or scientific evaluation results."
        )
    elif args.provider == "local":
        print(
            "\nNOTE: Ollama ran locally. Its confidence values are "
            "subjective model estimates, not calibrated probabilities."
        )

    if args.mode == "llm-only":
        print(
            "NOTE: The baseline still selects the initial candidates; "
            "only its ranking information is hidden from the investigator."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
