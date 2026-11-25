# LLM RCA V1 — graph-guided investigation scaffold

This workspace implements an **experimental scaffold inspired by** the Graph-Guided Investigation pseudocode supplied in the conversation. It is not claimed to reproduce GALA exactly. It starts from an initial service ranking, investigates Top-K services, expands to predecessor/successor neighbors if confidence is below a threshold, samples neighbors in batches, and records global investigation memory.

## Important distinctions
- `compare_rca_methods_v3.py` is copied only as a baseline reference. Do not execute it from this workspace yet: its paths may point to the original experiment directories.
- The graph-guided loop is a sequential prototype for deterministic debugging. The supplied paper algorithm describes parallel branches.
- The included sample case is synthetic and only validates the software pipeline. Never report its results as research findings.
- LLM-reported confidence is subjective and not a calibrated probability or proof of causality.
- The real RE2-OB adapter is not connected yet. We must inspect the actual V3 CSV and graph schemas before preparing per-case inputs.
- Ground-truth labels must never be included in LLM prompts or diagnostic bundles.

## Folder layout
- `src/compare_rca_methods_v3.py`: copied V3 baseline code.
- `src/graph_guided_investigation.py`: graph-neighbor investigation loop.
- `src/investigator.py`: mock smoke-test investigator and optional OpenAI investigator.
- `src/run_investigation.py`: command-line runner.
- `examples/sample_case.json`: synthetic smoke-test input only.
- `baseline_results/`: copied V3 results when matching files exist.
- `data/`: copied derived features when available.
- `results/`: new investigation outputs only.

## Step 1 — install the workspace
From the `LogRCA-Research` project root, run the bundled `install_workspace.py` script. It stops rather than overwriting an existing workspace.

## Step 2 — smoke-test without an API call
```bash
cd experiments/llm_rca_v1
python src/run_investigation.py --provider mock
```
This creates `results/sample_investigation.json`. It is only a software smoke test.

## Step 3 — connect the LLM after preparing real evidence
Install the optional dependency:
```bash
pip install -r requirements-llm.txt
```
Set `OPENAI_API_KEY` in your shell or local environment. Never paste the key into chat or commit it. Then run:
```bash
python src/run_investigation.py --provider openai --input path/to/normalized_case.json
```
The expected JSON format is shown in `examples/sample_case.json`.

## Next integration task
Build a leakage-safe adapter that produces normalized JSON for each RE2-OB case from logs, traces, temporal features, graph edges, and V3 candidate rankings. Root labels are used only in the evaluator, never in the prompt. Before implementing the adapter, inspect the local V3 result filenames and columns.
