"""Local fairness-risk metrics for proxy auditing."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from corex.games.game_theory import Nucleolus
from corex.games.value_functions import AblationValueFunction
from corex.utils.background import resolve_baseline


def _default_feature_names(width: int) -> list[str]:
    return [f"x{i}" for i in range(width)]


def resolve_proxy_feature_indices(
    proxy_features: Iterable[str | int],
    feature_names: Iterable[str] | None,
    width: int,
) -> list[int]:
    """Resolve proxy features expressed as names or indices."""

    names = list(feature_names) if feature_names is not None else _default_feature_names(width)
    if len(names) != width:
        raise ValueError(f"feature_names width {len(names)} does not match instance width {width}.")
    name_to_index = {name: index for index, name in enumerate(names)}
    resolved: list[int] = []
    for feature in proxy_features:
        if isinstance(feature, int):
            index = int(feature)
        else:
            if feature not in name_to_index:
                raise KeyError(f"Unknown proxy feature: {feature}")
            index = name_to_index[str(feature)]
        if index < 0 or index >= width:
            raise IndexError(f"Proxy feature index {index} is out of bounds for {width} features.")
        if index not in resolved:
            resolved.append(index)
    return resolved


def mask_proxy_features(
    instance: Iterable[float],
    proxy_features: Iterable[str | int],
    *,
    feature_names: Iterable[str] | None = None,
    baseline: Iterable[float] | None = None,
) -> list[float]:
    """Return x^{-Q} by replacing proxy features Q with baseline values."""

    values = [float(value) for value in instance]
    resolved_baseline = resolve_baseline(baseline, width=len(values)) or [0.0] * len(values)
    masked = list(values)
    for index in resolve_proxy_feature_indices(proxy_features, feature_names, len(values)):
        masked[index] = float(resolved_baseline[index])
    return masked


def predictive_nucleolus_allocation(
    predictor: Any,
    instance: Iterable[float],
    *,
    feature_names: Iterable[str] | None = None,
    baseline: Iterable[float] | None = None,
    prediction_method: str | None = None,
    target_class: int | None = None,
    normalize: bool = True,
    nucleolus_solver: Nucleolus | None = None,
) -> dict[str, Any]:
    """Compute the Nucleolus allocation for the predictive ablation game."""

    values = [float(value) for value in instance]
    names = list(feature_names) if feature_names is not None else _default_feature_names(len(values))
    resolved_baseline = resolve_baseline(baseline, width=len(values)) or [0.0] * len(values)
    value_function = AblationValueFunction(
        predictor=predictor,
        instance=values,
        baseline=resolved_baseline,
        feature_names=names,
        prediction_method=prediction_method,
        target_class=target_class,
        normalize=normalize,
        metadata={"metric": "predictive_nucleolus_allocation"},
    )
    game = value_function.make_game(metadata={"metric": "predictive_nucleolus_allocation"})
    solver = nucleolus_solver or Nucleolus()
    solution = solver.solve(game)
    return {
        "players": names,
        "allocation": np.asarray(solution.allocations, dtype=float),
        "solution": solution,
        "game": game,
    }


def compute_local_fairness_risk(
    predictor: Any,
    instance: Iterable[float],
    proxy_features: Iterable[str | int],
    *,
    feature_names: Iterable[str] | None = None,
    baseline: Iterable[float] | None = None,
    prediction_method: str | None = None,
    target_class: int | None = None,
    normalize: bool = True,
    nucleolus_solver: Nucleolus | None = None,
) -> dict[str, Any]:
    r"""Compute LFR = ||nu_pred(x) - nu_pred(x^{-Q})||_2.

    The proxy set Q is removed by replacing those features with the baseline
    used by the predictive ablation game. The returned dictionary is fully
    serializable and includes the two Nucleolus vectors needed to audit the
    calculation.
    """

    values = [float(value) for value in instance]
    names = list(feature_names) if feature_names is not None else _default_feature_names(len(values))
    if len(names) != len(values):
        raise ValueError(f"feature_names width {len(names)} does not match instance width {len(values)}.")
    resolved_baseline = resolve_baseline(baseline, width=len(values)) or [0.0] * len(values)
    proxy_indices = resolve_proxy_feature_indices(proxy_features, names, len(values))
    masked_instance = mask_proxy_features(
        values,
        proxy_indices,
        feature_names=names,
        baseline=resolved_baseline,
    )
    solver = nucleolus_solver or Nucleolus()
    original = predictive_nucleolus_allocation(
        predictor,
        values,
        feature_names=names,
        baseline=resolved_baseline,
        prediction_method=prediction_method,
        target_class=target_class,
        normalize=normalize,
        nucleolus_solver=solver,
    )
    masked = predictive_nucleolus_allocation(
        predictor,
        masked_instance,
        feature_names=names,
        baseline=resolved_baseline,
        prediction_method=prediction_method,
        target_class=target_class,
        normalize=normalize,
        nucleolus_solver=solver,
    )
    original_allocation = np.asarray(original["allocation"], dtype=float)
    masked_allocation = np.asarray(masked["allocation"], dtype=float)
    delta = original_allocation - masked_allocation
    return {
        "lfr": float(np.linalg.norm(delta, ord=2)),
        "proxy_features": [names[index] for index in proxy_indices],
        "proxy_feature_indices": [int(index) for index in proxy_indices],
        "masked_instance": [float(value) for value in masked_instance],
        "nucleolus_original": original_allocation.astype(float).tolist(),
        "nucleolus_without_proxy_features": masked_allocation.astype(float).tolist(),
        "delta": delta.astype(float).tolist(),
        "norm": "l2",
        "definition": "||nu_pred(x) - nu_pred(x_without_Q)||_2",
        "normalize_ablation_game": bool(normalize),
        "original_solver_diagnostics": dict(original["solution"].solver_diagnostics),
        "masked_solver_diagnostics": dict(masked["solution"].solver_diagnostics),
    }


__all__ = [
    "compute_local_fairness_risk",
    "mask_proxy_features",
    "predictive_nucleolus_allocation",
    "resolve_proxy_feature_indices",
]
