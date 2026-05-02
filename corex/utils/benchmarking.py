"""Benchmark helpers."""

from __future__ import annotations

import tracemalloc
import time
from typing import Any

import numpy as np

from corex.utils.explanation import ExplanationResult


def benchmark_explainer(
    explainer_name: str,
    result: ExplanationResult | None = None,
    reference_result: ExplanationResult | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    tracemalloc.start()
    comparison = None
    if result is not None and reference_result is not None:
        from corex.utils.validation import compare_explainers

        comparison = compare_explainers(reference_result, result)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime_ms = (time.perf_counter() - started) * 1000.0
    exactness = None
    if result is not None:
        validation = result.validate(check_efficiency=True, check_properties=True)
        exactness = {
            "exact_or_approximate": result.exact_or_approximate,
            "epsilon": result.epsilon,
            "valid": validation["valid"],
            "mean_abs_allocation": float(np.mean(np.abs(result.feature_values))),
            "null_players": validation["property_checks"].get("null_players", []),
            "symmetric_pairs": validation["property_checks"].get("symmetric_pairs", []),
            "stability_metric": result.solution_concepts.get("nucleolus").solver_diagnostics.get("max_excess")
            if result.solution_concepts.get("nucleolus") is not None
            else None,
        }
    return {
        "explainer_name": explainer_name,
        "comparison_target": "COREX",
        "status": "local_corex_benchmark_only",
        "runtime_ms": runtime_ms,
        "peak_memory_bytes": peak_memory,
        "exactness": exactness,
        "comparison": comparison,
        "reference_library": None,
    }


def benchmark_against_shap(
    explainer_name: str,
    result: ExplanationResult | None = None,
    reference_result: ExplanationResult | None = None,
) -> dict[str, Any]:
    benchmark = benchmark_explainer(explainer_name, result=result, reference_result=reference_result)
    benchmark["comparison_target"] = "SHAP"
    benchmark["reference_library"] = "shap"
    benchmark["deprecated_alias"] = True
    return benchmark
