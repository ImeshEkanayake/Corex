"""Linear explainer with analytic coalition game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from corex.explainers.base import BaseExplainer
from corex.games.game_theory import CooperativeGame


def _flatten_linear_params(values: Any, name: str, target_class: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        if target_class is None:
            raise ValueError(f"{name} is multiclass; provide target_class to select which linear score to explain.")
        return array[int(target_class)]
    raise ValueError(f"Unsupported {name} shape: {array.shape}")


@dataclass(init=False)
class LinearExplainer(BaseExplainer):
    coefficients: list[float] | None = None
    intercept: float | None = None
    target_class: int | None = None
    output: str = "linear_score"

    def __init__(
        self,
        model,
        background=None,
        feature_names=None,
        coefficients=None,
        intercept: float | None = None,
        target_class: int | None = None,
        output: str = "linear_score",
        gt_budget: int | None = None,
        baseline=None,
        callable_mode: str | None = None,
        scorer=None,
    ) -> None:
        self.predictor = model
        self.baseline = background if background is not None else baseline
        self.default_feature_names = feature_names
        self.scorer = scorer
        self.value_function_factory = None
        self.primary_solution = "banzhaf"
        self.cache_mode = "auto"
        self.sample_budget = gt_budget
        self.prediction_method = None
        self.target_class = target_class
        self.callable_mode = callable_mode
        self.coefficients = coefficients
        self.intercept = intercept
        self.output = output
        self.__post_init__()

    def _resolve_coefficients(self) -> np.ndarray:
        if self.coefficients is not None:
            return np.asarray(self.coefficients, dtype=float)
        model_coefficients = getattr(self.raw_predictor, "coef_", None)
        if model_coefficients is None:
            raise ValueError("LinearExplainer requires explicit coefficients or a predictor with coef_.")
        return _flatten_linear_params(model_coefficients, "coef_", target_class=self.target_class)

    def _resolve_intercept(self) -> float:
        if self.intercept is not None:
            return float(self.intercept)
        model_intercept = getattr(self.raw_predictor, "intercept_", 0.0)
        return float(_flatten_linear_params(model_intercept, "intercept_", target_class=self.target_class)[0])

    def _build_game(self, instance: list[float], feature_names: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs):
        if self.output != "linear_score":
            raise ValueError("LinearExplainer currently supports output='linear_score' only.")

        coefficients = self._resolve_coefficients()
        resolved_intercept = self._resolve_intercept()
        feature_names = feature_names or [f"feature_{index}" for index in range(len(instance))]
        baseline = np.asarray(self.baseline or [0.0] * len(instance), dtype=float)
        instance_array = np.asarray(instance, dtype=float)
        if coefficients.shape[0] != len(feature_names):
            raise ValueError("Coefficient vector length must match the explained feature count.")

        contribution_map = {
            name: float(coefficients[index] * (instance_array[index] - baseline[index]))
            for index, name in enumerate(feature_names)
        }

        def value_function(coalition: frozenset[str]) -> float:
            return float(sum(contribution_map[name] for name in coalition))

        return CooperativeGame(
            players=feature_names,
            value_function=value_function,
            value_function_name="LinearScoreValueFunction",
            game_metadata={
                **(metadata or {}),
                "analytic": True,
                "intercept": resolved_intercept,
                "output": self.output,
                "target_class": self.target_class,
            },
        )
