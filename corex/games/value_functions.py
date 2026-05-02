"""Value functions that emit cooperative games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from corex.games.game_theory import CooperativeGame
from corex.utils.background import resolve_baseline
from corex.utils.predictors import resolve_scalar_predictor


FeatureMasker = Callable[[list[float], set[int]], list[float]]
PredictFn = Callable[[list[float]], float]


def _default_masker(instance: list[float], coalition: set[int], baseline: list[float]) -> list[float]:
    return [instance[index] if index in coalition else baseline[index] for index in range(len(instance))]


@dataclass
class _BaseValueFunction:
    predictor: Any
    instance: list[float]
    baseline: list[float] | None = None
    feature_names: list[str] | None = None
    masker: FeatureMasker | None = None
    metadata: dict[str, Any] | None = None
    prediction_method: str | None = None
    target_class: int | None = None

    def __post_init__(self) -> None:
        self.instance = [float(value) for value in self.instance]
        self.baseline = resolve_baseline(self.baseline, width=len(self.instance))
        if self.baseline is None:
            self.baseline = [0.0] * len(self.instance)
        self.feature_names = self.feature_names or [f"feature_{index}" for index in range(len(self.instance))]
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique.")
        self._feature_to_index = {name: index for index, name in enumerate(self.feature_names)}
        self.metadata = dict(self.metadata or {})
        self.masker = self.masker or (lambda instance, coalition: _default_masker(instance, coalition, self.baseline or []))
        self.predictor = resolve_scalar_predictor(
            self.predictor,
            prediction_method=self.prediction_method,
            target_class=self.target_class,
        )

    def coalition_vector(self, coalition: Iterable[str]) -> list[float]:
        indices = {self._feature_to_index[player] for player in coalition}
        return self.masker(self.instance, indices)

    def make_game(
        self,
        players: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        cache_mode: str = "auto",
        sample_budget: int | None = None,
    ) -> CooperativeGame:
        players = players or list(self.feature_names)
        merged_metadata = {**self.metadata, **(metadata or {})}
        merged_metadata.setdefault("value_function_class", self.__class__.__name__)
        merged_metadata.setdefault("feature_count", len(players))
        return CooperativeGame(
            players=players,
            value_function=self.evaluate,
            value_function_name=self.__class__.__name__,
            game_metadata=merged_metadata,
            cache_mode=cache_mode,
            sample_budget=sample_budget,
        )


@dataclass
class PredictionValueFunction(_BaseValueFunction):
    def evaluate(self, coalition: frozenset[str]) -> float:
        return float(self.predictor(self.coalition_vector(coalition)))


@dataclass
class LossReductionValueFunction(_BaseValueFunction):
    target: float = 0.0
    loss_fn: Callable[[float, float], float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.loss_fn = self.loss_fn or (lambda prediction, target: (prediction - target) ** 2)
        self.full_loss = float(self.loss_fn(self.predictor(self.instance), self.target))

    def evaluate(self, coalition: frozenset[str]) -> float:
        coalition_loss = float(self.loss_fn(self.predictor(self.coalition_vector(coalition)), self.target))
        return self.full_loss - coalition_loss


@dataclass
class ProxyValueFunction(_BaseValueFunction):
    proxy_target: float = 0.0

    def evaluate(self, coalition: frozenset[str]) -> float:
        prediction = float(self.predictor(self.coalition_vector(coalition)))
        return -abs(prediction - self.proxy_target)

    def make_game(
        self,
        players: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        cache_mode: str = "auto",
        sample_budget: int | None = None,
    ) -> CooperativeGame:
        metadata = dict(metadata or {})
        metadata.setdefault("proxy_mode", "proxy_reconstruction")
        return super().make_game(players=players, metadata=metadata, cache_mode=cache_mode, sample_budget=sample_budget)


@dataclass
class AblationValueFunction(_BaseValueFunction):
    normalize: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reference_prediction = float(self.predictor(self.instance))
        self.baseline_prediction = float(self.predictor(self.baseline or []))
        baseline_prediction = self.baseline_prediction
        self.prediction_span = abs(self.reference_prediction - baseline_prediction) or 1.0

    def evaluate(self, coalition: frozenset[str]) -> float:
        prediction = float(self.predictor(self.coalition_vector(coalition)))
        shift = prediction - self.baseline_prediction
        return shift / self.prediction_span if self.normalize else shift
