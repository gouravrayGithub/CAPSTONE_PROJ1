"""Graph-guided iterative RCA investigation inspired by the supplied GALA algorithm.

This is an experimental implementation scaffold, not a reproduction of GALA.
It uses a pluggable Investigator and intentionally does not accept ground-truth labels.
"""
from __future__ import annotations

import random
from typing import Any, Callable

Investigator = Callable[[str, dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def _neighbors(service: str, edges: list[list[str]]) -> set[str]:
    """Return predecessors UNION successors for directed [source, target] edges."""
    result: set[str] = set()
    for edge in edges:
        if len(edge) != 2:
            continue
        source, target = edge
        if source == service:
            result.add(target)
        if target == service:
            result.add(source)
    result.discard(service)
    return result


def graph_guided_investigation(
    *,
    initial_ranking: list[dict[str, Any]],
    graph: dict[str, Any],
    diagnostic_bundles: dict[str, dict[str, Any]],
    investigator: Investigator,
    top_k: int = 3,
    threshold: float = 0.75,
    batch_size: int = 2,
    random_seed: int = 42,
    max_investigations: int = 30,
) -> dict[str, Any]:
    """Investigate Top-K ranked services, expanding to graph neighbors if uncertain.

    Required ranking item: {"service": str, "score": number}. Scores are retained as
    context; they are not interpreted as calibrated probabilities.

    Each investigator result should contain confidence, evidence_confidence,
    causal_confidence, supporting_facts, contradicting_facts, and conclusion.
    """
    if top_k < 1 or batch_size < 1 or max_investigations < 1:
        raise ValueError("top_k, batch_size and max_investigations must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    edges = graph.get("edges", [])
    graph_nodes = set(graph.get("nodes", []))
    for edge in edges:
        if len(edge) == 2:
            graph_nodes.update(edge)

    ranked = sorted(initial_ranking, key=lambda x: float(x.get("score", 0)), reverse=True)
    ranked = [r for r in ranked if r.get("service")]
    if not ranked:
        raise ValueError("initial_ranking must contain at least one service")

    rng = random.Random(random_seed)
    memory: list[dict[str, Any]] = []
    investigated: set[str] = set()
    branch_summaries: list[dict[str, Any]] = []

    def investigate(service: str) -> dict[str, Any]:
        if service in investigated:
            for record in reversed(memory):
                if record["service"] == service:
                    return record
        if len(investigated) >= max_investigations:
            return {"service": service, "confidence": 0.0, "skipped": True,
                    "reason": "max_investigations reached"}
        bundle = diagnostic_bundles.get(service, {})
        result = investigator(service, bundle, ranked)
        investigated.add(service)
        record = {
            "service": service,
            "confidence": _confidence(result.get("confidence", 0.0)),
            "evidence_confidence": _confidence(result.get("evidence_confidence", 0.0)),
            "causal_confidence": _confidence(result.get("causal_confidence", 0.0)),
            "supporting_facts": result.get("supporting_facts", []),
            "contradicting_facts": result.get("contradicting_facts", []),
            "conclusion": result.get("conclusion", ""),
            "reasoning": result.get("reasoning", ""),
            "source": result.get("source", "unknown"),
        }
        memory.append(record)
        return record

    # The paper describes these branches as parallel. This first local prototype runs
    # them sequentially for reproducibility and easier debugging.
    for initial in ranked[:top_k]:
        root_candidate = str(initial["service"])
        local = investigate(root_candidate)
        branch = {"initial_candidate": root_candidate, "investigated_services": [root_candidate]}
        if batch_records and max(r["confidence"] for r in batch_records) >= threshold:
            branch["stop_reason"] = "neighbor batch met confidence threshold"
            stopped = True
            break

        remaining = _neighbors(root_candidate, edges)
        if graph_nodes:
            remaining &= graph_nodes
        remaining.discard(root_candidate)
        remaining -= investigated
        stopped = False
        while remaining and len(investigated) < max_investigations:
            batch = rng.sample(sorted(remaining), k=min(batch_size, len(remaining)))
            remaining.difference_update(batch)
            batch_records = []
            for neighbor in batch:
                record = investigate(neighbor)
                branch["investigated_services"].append(neighbor)
                batch_records.append(record)
                if len(investigated) >= max_investigations:
                    break
            if batch_records and max(r["confidence"] for r in batch_records) >= threshold:
                branch["stop_reason"] = "neighbor batch met confidence threshold"
                stopped = True
                break
        if not stopped:
            branch["stop_reason"] = (
                "max investigations reached" if len(investigated) >= max_investigations
                else "no remaining uninvestigated neighbors"
            )
        branch_summaries.append(branch)

    # Candidate ranking is a transparent aggregation of investigator confidence,
    # not a calibrated causal probability. Multiple branch mentions are deduplicated.
    best_by_service: dict[str, dict[str, Any]] = {}
    for record in memory:
        previous = best_by_service.get(record["service"])
        if previous is None or record["confidence"] > previous["confidence"]:
            best_by_service[record["service"]] = record
    final_ranking = sorted(
        best_by_service.values(),
        key=lambda x: (x["confidence"], x["causal_confidence"], x["evidence_confidence"]),
        reverse=True,
    )
    return {
        "algorithm": "graph_guided_investigation_scaffold_v1",
        "settings": {
            "top_k": top_k, "threshold": threshold, "batch_size": batch_size,
            "random_seed": random_seed, "max_investigations": max_investigations,
            "branch_execution": "sequential prototype; paper algorithm describes parallel branches",
        },
        "final_ranking": final_ranking,
        "global_memory": memory,
        "branches": branch_summaries,
        "counts": {"investigated_services": len(investigated), "memory_records": len(memory)},
        "warning": "LLM-reported confidence is a heuristic, not a calibrated probability or proof of causality.",
    }


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
