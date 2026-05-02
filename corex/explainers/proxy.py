"""GT-first proxy auditing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from corex.games.game_theory import CooperativeGame, GameTheoreticScorer, summarize_coalitions_by_size
from corex.games.value_functions import AblationValueFunction, ProxyValueFunction
from corex.utils.background import resolve_baseline
from corex.utils.explanation import ExplanationResult, ProxyAuditReport
from corex.utils.predictors import resolve_batch_predictor, resolve_scalar_predictor

DEFAULT_PROXY_MAX_COALITION_SIZE = 3
DEFAULT_PROXY_MAX_COALITIONS_PER_SIZE = 1024


@dataclass(init=False)
class ProxyAuditor:
    predictor: Any
    baseline: list[float] | None = None
    scorer: GameTheoreticScorer | None = None
    prediction_method: str | None = None
    target_class: int | None = None
    dataset: Any = None
    default_feature_names: list[str] | None = None
    default_sensitive_attrs: dict[str, Any] | None = None
    default_evaluation_mode: str = "proxy_model"
    threshold: float | None = None
    sampling_budget: int | None = None
    callable_mode: str | None = None
    assume_thread_safe: bool = False
    max_coalitions_per_size: int = DEFAULT_PROXY_MAX_COALITIONS_PER_SIZE

    def __init__(
        self,
        model,
        dataset=None,
        sensitive_attrs=None,
        background=None,
        feature_names=None,
        baseline=None,
        scorer: GameTheoreticScorer | None = None,
        prediction_method: str | None = None,
        target_class: int | None = None,
        evaluation_mode: str = "proxy_model",
        threshold: float | None = None,
        sampling_budget: int | None = None,
        callable_mode: str | None = None,
        assume_thread_safe: bool = False,
        max_coalitions_per_size: int = DEFAULT_PROXY_MAX_COALITIONS_PER_SIZE,
    ) -> None:
        self.predictor = model
        self.baseline = background if background is not None else baseline
        self.scorer = scorer
        self.prediction_method = prediction_method
        self.target_class = target_class
        self.dataset = dataset
        self.default_feature_names = feature_names
        self.default_sensitive_attrs = dict(sensitive_attrs or {}) if sensitive_attrs is not None else None
        self.default_evaluation_mode = evaluation_mode
        self.threshold = threshold
        self.sampling_budget = sampling_budget
        self.callable_mode = callable_mode
        self.assume_thread_safe = assume_thread_safe
        self.max_coalitions_per_size = int(max_coalitions_per_size)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.raw_predictor = self.predictor
        self.baseline = resolve_baseline(self.baseline)
        self.scorer = self.scorer or GameTheoreticScorer(sampling_budget=self.sampling_budget, threshold=self.threshold)
        self.predictor = resolve_scalar_predictor(
            self.predictor,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
            callable_mode=self.callable_mode,
        )

    def _resolved_baseline(self, dataset: list[list[float]] | None = None, width: int | None = None) -> list[float]:
        if self.baseline is not None:
            return resolve_baseline(self.baseline, width=width) or [0.0] * int(width or 0)
        if dataset is not None and len(dataset) > 0:
            return np.asarray(dataset, dtype=float).mean(axis=0).astype(float).tolist()
        return [0.0] * int(width or 0)

    def _resolve_evaluation_mode(self, evaluation_mode: str | None) -> str:
        return evaluation_mode or self.default_evaluation_mode

    def _resolve_scorer(self, threshold: float | None = None) -> GameTheoreticScorer:
        if threshold is None or threshold == self.threshold:
            return self.scorer
        return GameTheoreticScorer(
            concepts=self.scorer.concepts,
            extra_solvers=self.scorer.extra_solvers,
            nucleolus_solver=self.scorer.nucleolus_solver,
            sampling_budget=self.sampling_budget,
            threshold=threshold,
        )

    def _validate_local_proxy_target(self, attribute_name: str, proxy_target: Any) -> float:
        array = np.asarray(proxy_target, dtype=float)
        if array.ndim == 0 or array.size == 1:
            return float(array.reshape(-1)[0])
        raise ValueError(
            f"Local proxy target for {attribute_name!r} must be scalar; "
            f"received shape {array.shape}."
        )

    def _validate_dataset_proxy_targets(self, attribute_name: str, proxy_targets: Any) -> list[float]:
        values = []
        for value in proxy_targets:
            array = np.asarray(value, dtype=float)
            if array.ndim == 0 or array.size == 1:
                values.append(float(array.reshape(-1)[0]))
            else:
                raise ValueError(
                    f"Dataset proxy targets for {attribute_name!r} must be scalar per row; "
                    f"received element with shape {array.shape}."
                )
        return values

    def _resolve_coalition_controls(
        self,
        feature_names: list[str],
        coalition_size_range: tuple[int, int] | int | None = None,
        max_features: int | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> tuple[tuple[int, int], int, int]:
        feature_count = len(feature_names)
        explicit_max_features = max_features is not None
        if max_features is not None:
            coalition_cap = int(max_features)
        elif coalition_size_range is None:
            coalition_cap = min(feature_count, DEFAULT_PROXY_MAX_COALITION_SIZE)
        else:
            coalition_cap = feature_count
        if coalition_cap < 1:
            raise ValueError("max_features must be at least 1.")
        coalition_cap = min(coalition_cap, feature_count)
        resolved_top_k = int(top_k if top_k is not None else (top_n if top_n is not None else 10))
        if resolved_top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if coalition_size_range is None:
            resolved_range = (1, coalition_cap)
        elif isinstance(coalition_size_range, int):
            resolved_range = (int(coalition_size_range), int(coalition_size_range))
        else:
            if len(coalition_size_range) != 2:
                raise ValueError("coalition_size_range must be an int or a (min_size, max_size) tuple.")
            resolved_range = (int(coalition_size_range[0]), int(coalition_size_range[1]))

        min_size, max_size = resolved_range
        if min_size < 1:
            raise ValueError("coalition_size_range minimum must be at least 1.")
        if max_size < min_size:
            raise ValueError("coalition_size_range maximum must be >= minimum.")
        if max_size > coalition_cap:
            if explicit_max_features:
                raise ValueError(
                    f"Requested coalition size {max_size} exceeds max_features={coalition_cap}. "
                    "Increase max_features or lower coalition_size_range."
                )
            resolved_range = (min_size, coalition_cap)
            min_size, max_size = resolved_range
        return (min_size, max_size), resolved_top_k, coalition_cap

    def _resolve_dataset_predictors(
        self,
        proxy_predictor: Callable[[list[float]], float] | None = None,
    ) -> tuple[Callable[[list[float]], float], Callable[[np.ndarray], np.ndarray]]:
        source_predictor = proxy_predictor or self.raw_predictor
        scalar_predictor = resolve_scalar_predictor(
            source_predictor,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
            callable_mode=self.callable_mode,
        )
        has_native_batch = (
            self.prediction_method is not None
            or any(getattr(source_predictor, name, None) is not None for name in ("predict_proba", "decision_function", "predict"))
            or getattr(source_predictor, "__name__", None) in {"predict", "predict_proba", "decision_function"}
        )
        if has_native_batch:
            batch_predictor = resolve_batch_predictor(
                source_predictor,
                prediction_method=self.prediction_method,
                target_class=self.target_class,
                callable_mode=self.callable_mode,
            )
        else:
            def batch_predictor(rows: np.ndarray) -> np.ndarray:
                matrix = np.asarray(rows, dtype=float)
                if matrix.ndim == 1:
                    matrix = matrix.reshape(1, -1)
                return np.asarray([scalar_predictor(row.tolist()) for row in matrix], dtype=float)

        return scalar_predictor, batch_predictor

    def audit_single_instance(
        self,
        instance,
        feature_names,
        sensitive_attributes,
        **kwargs,
    ) -> ProxyAuditReport:
        return self.audit(instance, feature_names, sensitive_attributes, **kwargs)

    def audit_single_coalition(
        self,
        coalition,
        instance=None,
        feature_names=None,
        attribute_name: str | None = None,
        proxy_target=None,
        evaluation_mode: str | None = None,
        proxy_predictor: Callable[[list[float]], float] | None = None,
        sensitive_attributes: dict[str, Any] | None = None,
        sensitive_attrs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if instance is None:
            raise ValueError("audit_single_coalition requires instance=...")
        feature_names = feature_names or self.default_feature_names
        if feature_names is None:
            raise ValueError("audit_single_coalition requires feature_names or constructor defaults.")
        if attribute_name is None or proxy_target is None:
            sensitive_attributes = sensitive_attributes or sensitive_attrs or self.default_sensitive_attrs
            if not sensitive_attributes or len(sensitive_attributes) != 1:
                raise ValueError(
                    "Provide attribute_name/proxy_target explicitly or pass a single sensitive attribute via sensitive_attributes/sensitive_attrs."
                )
            attribute_name, proxy_target = next(iter(sensitive_attributes.items()))
        proxy_target = self._validate_local_proxy_target(attribute_name, proxy_target)
        resolved_mode = self._resolve_evaluation_mode(evaluation_mode)
        coalition_names = frozenset(str(name) for name in coalition)
        if resolved_mode == "proxy_model":
            result = self._build_proxy_result(
                instance,
                feature_names,
                attribute_name,
                proxy_target,
                resolved_mode,
                (len(coalition_names), len(coalition_names)),
                1,
                proxy_predictor=proxy_predictor,
            )
            value = result.game.evaluate(coalition_names)
            return {
                "attribute_name": attribute_name,
                "coalition": sorted(coalition_names),
                "coalition_size": len(coalition_names),
                "value": float(value),
                "evaluation_mode": resolved_mode,
            }
        if resolved_mode == "ablation":
            result = self._build_ablation_result(
                instance,
                feature_names,
                attribute_name,
                resolved_mode,
                (len(coalition_names), len(coalition_names)),
                1,
            )
            value = result.game.evaluate(coalition_names)
            return {
                "attribute_name": attribute_name,
                "coalition": sorted(coalition_names),
                "coalition_size": len(coalition_names),
                "value": float(value),
                "evaluation_mode": resolved_mode,
            }
        if resolved_mode in {"both_sequential", "both_parallel"}:
            return {
                "proxy_model": self.audit_single_coalition(
                    coalition,
                    instance=instance,
                    feature_names=feature_names,
                    attribute_name=attribute_name,
                    proxy_target=proxy_target,
                    evaluation_mode="proxy_model",
                    proxy_predictor=proxy_predictor,
                ),
                "ablation": self.audit_single_coalition(
                    coalition,
                    instance=instance,
                    feature_names=feature_names,
                    attribute_name=attribute_name,
                    proxy_target=proxy_target,
                    evaluation_mode="ablation",
                    proxy_predictor=proxy_predictor,
                ),
            }
        raise ValueError(f"Unsupported evaluation_mode: {resolved_mode}")

    def audit_attribute(
        self,
        feature_names,
        attribute_name: str,
        proxy_target,
        instance=None,
        dataset=None,
        evaluation_mode: str = "proxy_model",
        proxy_predictor: Callable[[list[float]], float] | None = None,
        **kwargs,
    ):
        if proxy_predictor is not None:
            kwargs.setdefault("proxy_predictors", {attribute_name: proxy_predictor})
        if dataset is not None:
            report = self.audit_dataset(
                dataset=dataset,
                feature_names=feature_names,
                sensitive_attribute_series={attribute_name: proxy_target},
                evaluation_mode=evaluation_mode,
                **kwargs,
            )
        elif instance is not None:
            report = self.audit_single_instance(
                instance=instance,
                feature_names=feature_names,
                sensitive_attributes={attribute_name: proxy_target},
                evaluation_mode=evaluation_mode,
                **kwargs,
            )
        else:
            raise ValueError("Provide either instance=... or dataset=... to audit_attribute.")
        return report.attribute_reports[attribute_name]

    def _bootstrap_interval(
        self,
        values: list[float],
        bootstrap_samples: int,
        random_seed: int,
        confidence_level: float,
    ) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        data = np.asarray(values, dtype=float)
        if bootstrap_samples <= 0:
            mean = float(np.mean(data))
            return {"mean": mean, "lower": mean, "upper": mean}
        generator = np.random.default_rng(random_seed)
        sample_means = []
        for _ in range(int(bootstrap_samples)):
            sample = generator.choice(data, size=data.shape[0], replace=True)
            sample_means.append(float(np.mean(sample)))
        alpha = (1.0 - confidence_level) / 2.0
        return {
            "mean": float(np.mean(data)),
            "lower": float(np.quantile(sample_means, alpha)),
            "upper": float(np.quantile(sample_means, 1.0 - alpha)),
        }

    def _dataset_summary_metadata(
        self,
        report: ExplanationResult,
        bootstrap_samples: int,
        random_seed: int,
        confidence_level: float,
    ) -> dict[str, float | int | dict[str, float] | None]:
        game = report.game
        if game is None:
            return {}
        row_scores = report.metadata.get("_dataset_row_scores", {})
        empty_scores = [float(value) for value in row_scores.get("empty", [])]
        grand_scores = [float(value) for value in row_scores.get("grand", [])]
        empty_value = float(np.mean(empty_scores)) if empty_scores else float(game.evaluate_indices(frozenset()))
        grand_value = float(np.mean(grand_scores)) if grand_scores else float(game.evaluate_indices(game.grand_coalition_indices()))
        summary: dict[str, float | int | dict[str, float] | None] = {
            "sample_count": max(len(empty_scores), len(grand_scores)),
            "empty_coalition_value": empty_value,
            "grand_coalition_value": grand_value,
            "lift_over_empty": grand_value - empty_value,
        }
        if "target_mean" in report.metadata:
            summary["target_mean"] = float(report.metadata["target_mean"])
        if bootstrap_samples > 0:
            summary["grand_coalition_ci"] = self._bootstrap_interval(
                grand_scores or [grand_value],
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
                confidence_level=confidence_level,
            )
            summary["empty_coalition_ci"] = self._bootstrap_interval(
                empty_scores or [empty_value],
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed + 1,
                confidence_level=confidence_level,
            )
        return summary

    def _build_proxy_result(
        self,
        instance,
        feature_names,
        attribute_name,
        proxy_target,
        mode,
        coalition_size_range,
        top_n,
        scorer: GameTheoreticScorer | None = None,
        proxy_predictor: Callable[[list[float]], float] | None = None,
    ):
        resolved_predictor = resolve_scalar_predictor(
            proxy_predictor or self.predictor,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
            callable_mode=self.callable_mode,
        )
        value_function = ProxyValueFunction(
            predictor=resolved_predictor,
            instance=instance,
            baseline=self.baseline,
            feature_names=feature_names,
            proxy_target=proxy_target,
            metadata={
                "attribute_name": attribute_name,
                "evaluation_mode": mode,
                "proxy_predictor_mode": "attribute_specific" if proxy_predictor is not None else "primary_predictor_fallback",
            },
        )
        game = value_function.make_game()
        solutions = (scorer or self.scorer).score(game)
        coalition_summary = summarize_coalitions_by_size(
            game,
            min_size=coalition_size_range[0],
            max_size=coalition_size_range[1],
            top_n=top_n,
            max_coalitions_per_size=self.max_coalitions_per_size,
        )
        return ExplanationResult.from_game(
            game,
            solutions,
            metadata={
                "attribute_name": attribute_name,
                "coalition_size_summary": coalition_summary,
                "proxy_predictor_mode": "attribute_specific" if proxy_predictor is not None else "primary_predictor_fallback",
                "audit_scope": "local",
            },
        )

    def _build_ablation_result(
        self,
        instance,
        feature_names,
        attribute_name,
        mode,
        coalition_size_range,
        top_n,
        scorer: GameTheoreticScorer | None = None,
    ):
        value_function = AblationValueFunction(
            predictor=self.predictor,
            instance=instance,
            baseline=self.baseline,
            feature_names=feature_names,
            metadata={"attribute_name": attribute_name, "evaluation_mode": mode},
        )
        game = value_function.make_game(metadata={"proxy_mode": "ablation_impact"})
        solutions = (scorer or self.scorer).score(game)
        coalition_summary = summarize_coalitions_by_size(
            game,
            min_size=coalition_size_range[0],
            max_size=coalition_size_range[1],
            top_n=top_n,
            max_coalitions_per_size=self.max_coalitions_per_size,
        )
        return ExplanationResult.from_game(
            game,
            solutions,
            metadata={"attribute_name": attribute_name, "coalition_size_summary": coalition_summary, "audit_scope": "local"},
        )

    def _build_dataset_proxy_result(
        self,
        dataset,
        feature_names,
        attribute_name,
        proxy_targets,
        mode,
        coalition_size_range,
        top_n,
        scorer: GameTheoreticScorer | None = None,
        proxy_predictor: Callable[[list[float]], float] | None = None,
    ):
        resolved_predictor, batch_predictor = self._resolve_dataset_predictors(proxy_predictor=proxy_predictor)
        rows_array = np.asarray(dataset, dtype=float)
        baseline = self._resolved_baseline(dataset=dataset, width=len(feature_names))
        baseline_array = np.asarray(baseline, dtype=float)
        baseline_matrix = np.repeat(baseline_array.reshape(1, -1), rows_array.shape[0], axis=0)
        feature_index = {name: index for index, name in enumerate(feature_names)}
        rows = rows_array.astype(float).tolist()
        targets = np.asarray([float(value) for value in proxy_targets], dtype=float)
        if len(rows) != len(targets):
            raise ValueError(f"Proxy target series for {attribute_name} must match dataset length.")
        baseline_prediction = float(resolved_predictor(baseline))
        empty_scores = (-np.abs(np.repeat(baseline_prediction, len(targets)) - targets)).astype(float).tolist()
        grand_predictions = batch_predictor(rows_array)
        grand_scores = (-np.abs(grand_predictions - targets)).astype(float).tolist()

        prediction_cache: dict[tuple[int, ...], np.ndarray] = {
            tuple(): np.repeat(baseline_prediction, len(targets)),
            tuple(range(len(feature_names))): grand_predictions,
        }

        def value_function(coalition: frozenset[str]) -> float:
            indices = tuple(sorted(feature_index[name] for name in coalition))
            predictions = prediction_cache.get(indices)
            if predictions is None:
                masked = baseline_matrix.copy()
                if indices:
                    masked[:, list(indices)] = rows_array[:, list(indices)]
                predictions = batch_predictor(masked)
                prediction_cache[indices] = predictions
            errors = -np.abs(predictions - targets)
            return float(np.mean(errors)) if errors.size else 0.0

        game = CooperativeGame(
            players=feature_names,
            value_function=value_function,
            value_function_name="DatasetProxyValueFunction",
            game_metadata={
                "attribute_name": attribute_name,
                "evaluation_mode": mode,
                "audit_scope": "global",
                "sample_count": len(rows),
                "proxy_predictor_mode": "attribute_specific" if proxy_predictor is not None else "primary_predictor_fallback",
            },
        )
        solutions = (scorer or self.scorer).score(game)
        coalition_summary = summarize_coalitions_by_size(
            game,
            min_size=coalition_size_range[0],
            max_size=coalition_size_range[1],
            top_n=top_n,
            max_coalitions_per_size=self.max_coalitions_per_size,
        )
        return ExplanationResult.from_game(
            game,
            solutions,
            metadata={
                "attribute_name": attribute_name,
                "coalition_size_summary": coalition_summary,
                "proxy_predictor_mode": "attribute_specific" if proxy_predictor is not None else "primary_predictor_fallback",
                "audit_scope": "global",
                "sample_count": len(rows),
                "target_mean": float(np.mean(targets)) if len(targets) else 0.0,
                "_dataset_row_scores": {"empty": empty_scores, "grand": grand_scores},
            },
        )

    def _build_dataset_ablation_result(self, dataset, feature_names, attribute_name, mode, coalition_size_range, top_n, scorer: GameTheoreticScorer | None = None):
        _, batch_predictor = self._resolve_dataset_predictors()
        rows_array = np.asarray(dataset, dtype=float)
        baseline = self._resolved_baseline(dataset=dataset, width=len(feature_names))
        baseline_array = np.asarray(baseline, dtype=float)
        baseline_matrix = np.repeat(baseline_array.reshape(1, -1), rows_array.shape[0], axis=0)
        feature_index = {name: index for index, name in enumerate(feature_names)}
        rows = rows_array.astype(float).tolist()
        full_predictions = batch_predictor(rows_array)
        baseline_prediction = float(self.predictor(baseline))
        span = max(float(np.max(np.abs(full_predictions - baseline_prediction))), 1.0) if len(rows) else 1.0
        empty_scores = [0.0 for _ in rows]
        grand_scores = ((full_predictions - baseline_prediction) / span).astype(float).tolist()

        prediction_cache: dict[tuple[int, ...], np.ndarray] = {
            tuple(): np.repeat(baseline_prediction, len(rows)),
            tuple(range(len(feature_names))): full_predictions,
        }

        def value_function(coalition: frozenset[str]) -> float:
            indices = tuple(sorted(feature_index[name] for name in coalition))
            predictions = prediction_cache.get(indices)
            if predictions is None:
                masked = baseline_matrix.copy()
                if indices:
                    masked[:, list(indices)] = rows_array[:, list(indices)]
                predictions = batch_predictor(masked)
                prediction_cache[indices] = predictions
            shifts = (predictions - baseline_prediction) / span
            return float(np.mean(shifts)) if shifts.size else 0.0

        game = CooperativeGame(
            players=feature_names,
            value_function=value_function,
            value_function_name="DatasetAblationValueFunction",
            game_metadata={
                "attribute_name": attribute_name,
                "evaluation_mode": mode,
                "audit_scope": "global",
                "sample_count": len(rows),
                "proxy_mode": "ablation_impact",
            },
        )
        solutions = (scorer or self.scorer).score(game)
        coalition_summary = summarize_coalitions_by_size(
            game,
            min_size=coalition_size_range[0],
            max_size=coalition_size_range[1],
            top_n=top_n,
            max_coalitions_per_size=self.max_coalitions_per_size,
        )
        return ExplanationResult.from_game(
            game,
            solutions,
            metadata={
                "attribute_name": attribute_name,
                "coalition_size_summary": coalition_summary,
                "audit_scope": "global",
                "sample_count": len(rows),
                "_dataset_row_scores": {"empty": empty_scores, "grand": grand_scores},
            },
        )

    def audit(
        self,
        instance=None,
        feature_names=None,
        sensitive_attributes=None,
        evaluation_mode: str | None = None,
        coalition_size_range: tuple[int, int] | int | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        max_features: int | None = None,
        proxy_predictors: dict[str, Callable[[list[float]], float]] | None = None,
        sensitive_attrs: dict[str, Any] | None = None,
        threshold: float | None = None,
        assume_thread_safe: bool | None = None,
    ) -> ProxyAuditReport:
        if instance is None:
            raise ValueError("audit requires instance=...")
        evaluation_mode = self._resolve_evaluation_mode(evaluation_mode)
        scorer = self._resolve_scorer(threshold)
        feature_names = feature_names or self.default_feature_names
        sensitive_attributes = sensitive_attributes or sensitive_attrs or self.default_sensitive_attrs
        if feature_names is None:
            raise ValueError("audit requires feature_names or a default feature_names configured in the constructor.")
        if sensitive_attributes is None:
            raise ValueError("audit requires sensitive_attributes/sensitive_attrs or constructor defaults.")
        coalition_size_range, resolved_top_k, resolved_max_features = self._resolve_coalition_controls(
            feature_names=list(feature_names),
            coalition_size_range=coalition_size_range,
            max_features=max_features,
            top_k=top_k,
            top_n=top_n,
        )
        reports = {}
        proxy_predictors = dict(proxy_predictors or {})
        missing_proxy_predictors = [attribute_name for attribute_name in proxy_predictors if attribute_name not in sensitive_attributes]
        if missing_proxy_predictors:
            raise ValueError(
                "proxy_predictors contains attributes not present in sensitive_attributes: "
                + ", ".join(sorted(missing_proxy_predictors))
            )
        report_metadata = {
            "coalition_size_range": [int(coalition_size_range[0]), int(coalition_size_range[1])],
            "top_k": int(resolved_top_k),
            "top_n": int(resolved_top_k),
            "max_features": int(resolved_max_features),
            "max_coalitions_per_size": int(self.max_coalitions_per_size),
            "coalition_size_analysis": {},
            "proxy_predictor_attributes": sorted(proxy_predictors),
            "audit_scope": "local",
        }
        for attribute_name, proxy_target in sensitive_attributes.items():
            proxy_target = self._validate_local_proxy_target(attribute_name, proxy_target)
            attribute_proxy_predictor = proxy_predictors.get(attribute_name)
            if evaluation_mode == "proxy_model":
                reports[attribute_name] = self._build_proxy_result(
                    instance,
                    feature_names,
                    attribute_name,
                    proxy_target,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                    proxy_predictor=attribute_proxy_predictor,
                )
                report_metadata["coalition_size_analysis"][attribute_name] = reports[attribute_name].metadata["coalition_size_summary"]
            elif evaluation_mode == "ablation":
                reports[attribute_name] = self._build_ablation_result(
                    instance,
                    feature_names,
                    attribute_name,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                )
                report_metadata["coalition_size_analysis"][attribute_name] = reports[attribute_name].metadata["coalition_size_summary"]
            elif evaluation_mode == "both_sequential":
                proxy_result = self._build_proxy_result(
                    instance,
                    feature_names,
                    attribute_name,
                    proxy_target,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                    proxy_predictor=attribute_proxy_predictor,
                )
                ablation_result = self._build_ablation_result(
                    instance,
                    feature_names,
                    attribute_name,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                )
                reports[attribute_name] = {
                    "proxy_model": proxy_result,
                    "ablation": ablation_result,
                }
                report_metadata["coalition_size_analysis"][attribute_name] = {
                    "proxy_model": proxy_result.metadata["coalition_size_summary"],
                    "ablation": ablation_result.metadata["coalition_size_summary"],
                }
            elif evaluation_mode == "both_parallel":
                if not (self.assume_thread_safe if assume_thread_safe is None else assume_thread_safe):
                    raise ValueError(
                        "both_parallel uses threads. Set assume_thread_safe=True only if the predictor "
                        "and solver stack are safe for concurrent execution."
                    )
                with ThreadPoolExecutor(max_workers=2) as executor:
                    proxy_future = executor.submit(
                        self._build_proxy_result,
                        instance,
                        feature_names,
                        attribute_name,
                        proxy_target,
                        evaluation_mode,
                        coalition_size_range,
                        resolved_top_k,
                        scorer,
                        attribute_proxy_predictor,
                    )
                    ablation_future = executor.submit(
                        self._build_ablation_result,
                        instance,
                        feature_names,
                        attribute_name,
                        evaluation_mode,
                        coalition_size_range,
                        resolved_top_k,
                        scorer,
                    )
                    proxy_result = proxy_future.result()
                    ablation_result = ablation_future.result()
                reports[attribute_name] = {
                    "proxy_model": proxy_result,
                    "ablation": ablation_result,
                }
                report_metadata["coalition_size_analysis"][attribute_name] = {
                    "proxy_model": proxy_result.metadata["coalition_size_summary"],
                    "ablation": ablation_result.metadata["coalition_size_summary"],
                }
            else:
                raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
        return ProxyAuditReport(evaluation_mode=evaluation_mode, attribute_reports=reports, metadata=report_metadata)

    def audit_dataset(
        self,
        dataset=None,
        feature_names=None,
        sensitive_attribute_series=None,
        evaluation_mode: str | None = None,
        coalition_size_range: tuple[int, int] | int | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        max_features: int | None = None,
        proxy_predictors: dict[str, Callable[[list[float]], float]] | None = None,
        bootstrap_samples: int = 0,
        confidence_level: float = 0.95,
        random_seed: int = 0,
        sensitive_attrs: dict[str, Any] | None = None,
        threshold: float | None = None,
        assume_thread_safe: bool | None = None,
    ) -> ProxyAuditReport:
        evaluation_mode = self._resolve_evaluation_mode(evaluation_mode)
        scorer = self._resolve_scorer(threshold)
        dataset = dataset if dataset is not None else self.dataset
        feature_names = feature_names or self.default_feature_names
        sensitive_attribute_series = sensitive_attribute_series or sensitive_attrs or self.default_sensitive_attrs
        if dataset is None:
            raise ValueError("audit_dataset requires dataset=... or a default dataset configured in the constructor.")
        if feature_names is None:
            raise ValueError("audit_dataset requires feature_names or a default feature_names configured in the constructor.")
        if sensitive_attribute_series is None:
            raise ValueError("audit_dataset requires sensitive_attribute_series/sensitive_attrs or constructor defaults.")
        coalition_size_range, resolved_top_k, resolved_max_features = self._resolve_coalition_controls(
            feature_names=list(feature_names),
            coalition_size_range=coalition_size_range,
            max_features=max_features,
            top_k=top_k,
            top_n=top_n,
        )
        rows = [list(map(float, row)) for row in dataset]
        proxy_predictors = dict(proxy_predictors or {})
        missing_proxy_predictors = [attribute_name for attribute_name in proxy_predictors if attribute_name not in sensitive_attribute_series]
        if missing_proxy_predictors:
            raise ValueError(
                "proxy_predictors contains attributes not present in sensitive_attribute_series: "
                + ", ".join(sorted(missing_proxy_predictors))
            )
        reports = {}
        report_metadata = {
            "coalition_size_range": [int(coalition_size_range[0]), int(coalition_size_range[1])],
            "top_k": int(resolved_top_k),
            "top_n": int(resolved_top_k),
            "max_features": int(resolved_max_features),
            "max_coalitions_per_size": int(self.max_coalitions_per_size),
            "coalition_size_analysis": {},
            "proxy_predictor_attributes": sorted(proxy_predictors),
            "audit_scope": "global",
            "sample_count": len(rows),
            "bootstrap_samples": int(bootstrap_samples),
            "confidence_level": float(confidence_level),
        }
        for attribute_name, proxy_targets in sensitive_attribute_series.items():
            target_values = self._validate_dataset_proxy_targets(attribute_name, proxy_targets)
            attribute_proxy_predictor = proxy_predictors.get(attribute_name)
            if evaluation_mode == "proxy_model":
                reports[attribute_name] = self._build_dataset_proxy_result(
                    rows,
                    feature_names,
                    attribute_name,
                    target_values,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                    proxy_predictor=attribute_proxy_predictor,
                )
            elif evaluation_mode == "ablation":
                reports[attribute_name] = self._build_dataset_ablation_result(
                    rows,
                    feature_names,
                    attribute_name,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                )
            elif evaluation_mode == "both_sequential":
                proxy_result = self._build_dataset_proxy_result(
                    rows,
                    feature_names,
                    attribute_name,
                    target_values,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                    proxy_predictor=attribute_proxy_predictor,
                )
                ablation_result = self._build_dataset_ablation_result(
                    rows,
                    feature_names,
                    attribute_name,
                    evaluation_mode,
                    coalition_size_range,
                    resolved_top_k,
                    scorer=scorer,
                )
                reports[attribute_name] = {
                    "proxy_model": proxy_result,
                    "ablation": ablation_result,
                }
            elif evaluation_mode == "both_parallel":
                if not (self.assume_thread_safe if assume_thread_safe is None else assume_thread_safe):
                    raise ValueError(
                        "both_parallel uses threads. Set assume_thread_safe=True only if the predictor "
                        "and solver stack are safe for concurrent execution."
                    )
                with ThreadPoolExecutor(max_workers=2) as executor:
                    proxy_future = executor.submit(
                        self._build_dataset_proxy_result,
                        rows,
                        feature_names,
                        attribute_name,
                        target_values,
                        evaluation_mode,
                        coalition_size_range,
                        resolved_top_k,
                        scorer,
                        attribute_proxy_predictor,
                    )
                    ablation_future = executor.submit(
                        self._build_dataset_ablation_result,
                        rows,
                        feature_names,
                        attribute_name,
                        evaluation_mode,
                        coalition_size_range,
                        resolved_top_k,
                        scorer,
                    )
                    proxy_result = proxy_future.result()
                    ablation_result = ablation_future.result()
                reports[attribute_name] = {
                    "proxy_model": proxy_result,
                    "ablation": ablation_result,
                }
            else:
                raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
            if evaluation_mode in {"both_sequential", "both_parallel"}:
                report_metadata["coalition_size_analysis"][attribute_name] = {
                    "proxy_model": reports[attribute_name]["proxy_model"].metadata["coalition_size_summary"],
                    "ablation": reports[attribute_name]["ablation"].metadata["coalition_size_summary"],
                }
                for mode_report in reports[attribute_name].values():
                    mode_report.metadata["dataset_summary"] = self._dataset_summary_metadata(
                        mode_report,
                        bootstrap_samples=bootstrap_samples,
                        random_seed=random_seed,
                        confidence_level=confidence_level,
                    )
            else:
                report_metadata["coalition_size_analysis"][attribute_name] = reports[attribute_name].metadata["coalition_size_summary"]
                reports[attribute_name].metadata["dataset_summary"] = self._dataset_summary_metadata(
                    reports[attribute_name],
                    bootstrap_samples=bootstrap_samples,
                    random_seed=random_seed,
                    confidence_level=confidence_level,
                )
        return ProxyAuditReport(evaluation_mode=evaluation_mode, attribute_reports=reports, metadata=report_metadata)
