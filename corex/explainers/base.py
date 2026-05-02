"""GT-first explainer base class."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from corex.games.game_theory import GameTheoreticScorer
from corex.games.value_functions import PredictionValueFunction
from corex.utils.background import resolve_baseline
from corex.utils.explanation import ExplanationResult
from corex.utils.predictors import resolve_scalar_predictor


ValueFunctionFactory = Callable[..., Any]


@dataclass
class BaseExplainer:
    predictor: Any
    baseline: list[float] | None = None
    scorer: GameTheoreticScorer | None = None
    value_function_factory: ValueFunctionFactory | None = None
    primary_solution: str = "banzhaf"
    cache_mode: str = "auto"
    sample_budget: int | None = None
    prediction_method: str | None = None
    target_class: int | None = None
    callable_mode: str | None = None
    default_feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.raw_predictor = self.predictor
        self.baseline = resolve_baseline(self.baseline)
        self.scorer = self.scorer or GameTheoreticScorer()
        self.predictor = resolve_scalar_predictor(
            self.predictor,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
            callable_mode=self.callable_mode,
        )

    def _make_value_function_factory(self) -> ValueFunctionFactory:
        return self.value_function_factory or PredictionValueFunction

    def _build_value_function(
        self,
        instance: list[float],
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        factory = self._make_value_function_factory()
        return factory(
            predictor=self.predictor,
            instance=instance,
            baseline=self.baseline,
            feature_names=feature_names,
            metadata=metadata,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
            **kwargs,
        )

    def _build_game(
        self,
        instance: list[float],
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        value_function = self._build_value_function(instance, feature_names=feature_names, metadata=metadata, **kwargs)
        return value_function.make_game(cache_mode=self.cache_mode, sample_budget=self.sample_budget)

    def _score_game(self, game):
        return self.scorer.score(game)

    def _make_result(self, game, solutions, primary_solution: str | None = None, metadata: dict[str, Any] | None = None):
        return ExplanationResult.from_game(
            game,
            solutions,
            primary_solution=primary_solution or self.primary_solution,
            metadata=metadata,
        )

    def explain(
        self,
        instance: list[float],
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        feature_names = feature_names or self.default_feature_names
        game = self._build_game(instance, feature_names=feature_names, metadata=metadata, **kwargs)
        solutions = self._score_game(game)
        return self._make_result(game, solutions, metadata=metadata)

    def _normalize_batch_instances(self, instances: list[list[float]] | np.ndarray) -> list[list[float]]:
        array = np.asarray(instances, dtype=float)
        if array.ndim == 1:
            return [array.astype(float).tolist()]
        if array.ndim == 2:
            return array.astype(float).tolist()
        raise ValueError("instances must be a 1D row, a 2D batch, or a list of rows.")

    def explain_batch(
        self,
        instances: list[list[float]] | np.ndarray,
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> list[ExplanationResult]:
        if n_jobs not in {None, 1}:
            raise ValueError("explain_batch() is sequential. Use parallel_explain_batch(..., n_jobs=...) for explicit threaded execution.")
        normalized_instances = self._normalize_batch_instances(instances)
        if metadata is None or isinstance(metadata, dict):
            metadata_items = [dict(metadata or {}) for _ in normalized_instances]
        else:
            if len(metadata) != len(normalized_instances):
                raise ValueError("Batch metadata must match the number of instances.")
            metadata_items = [dict(item or {}) for item in metadata]
        return [
            self.explain(instance, feature_names=feature_names, metadata=metadata_item, **kwargs)
            for instance, metadata_item in zip(normalized_instances, metadata_items)
        ]

    def parallel_explain_batch(
        self,
        instances: list[list[float]] | np.ndarray,
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        n_jobs: int = 2,
        assume_thread_safe: bool = False,
        **kwargs: Any,
    ) -> list[ExplanationResult]:
        normalized_instances = self._normalize_batch_instances(instances)
        if metadata is None or isinstance(metadata, dict):
            metadata_items = [dict(metadata or {}) for _ in normalized_instances]
        else:
            if len(metadata) != len(normalized_instances):
                raise ValueError("Batch metadata must match the number of instances.")
            metadata_items = [dict(item or {}) for item in metadata]
        jobs = max(1, int(n_jobs))
        if jobs == 1 or len(normalized_instances) <= 1:
            return self.explain_batch(normalized_instances, feature_names=feature_names, metadata=metadata_items, **kwargs)
        if not assume_thread_safe:
            raise ValueError(
                "parallel_explain_batch() uses threads. Set assume_thread_safe=True only if the predictor, "
                "solver stack, and downstream libraries are safe for concurrent execution."
            )
        with ThreadPoolExecutor(max_workers=min(jobs, len(normalized_instances))) as executor:
            futures = [
                executor.submit(self.explain, instance, feature_names=feature_names, metadata=metadata_item, **kwargs)
                for instance, metadata_item in zip(normalized_instances, metadata_items)
            ]
            return [future.result() for future in futures]
