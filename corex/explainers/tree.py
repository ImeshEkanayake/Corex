from __future__ import annotations

from dataclasses import dataclass

from corex.explainers.base import BaseExplainer
from corex.games.game_theory import GameTheoreticScorer, Nucleolus
from corex.games.value_functions import PredictionValueFunction


@dataclass(init=False)
class TreeExplainer(BaseExplainer):
    max_exact_players: int = 12

    def __init__(
        self,
        model,
        background=None,
        feature_names=None,
        nsamples: int | None = None,
        gt_budget: int | None = None,
        coalition_budget: int | None = None,
        target_class: int | None = None,
        baseline=None,
        prediction_method: str | None = None,
        callable_mode: str | None = None,
        max_exact_players: int = 12,
        scorer: GameTheoreticScorer | None = None,
    ) -> None:
        self.predictor = model
        self.baseline = background if background is not None else baseline
        self.default_feature_names = feature_names
        self.scorer = scorer
        self.value_function_factory = None
        self.primary_solution = "banzhaf"
        self.cache_mode = "auto"
        self.sample_budget = gt_budget if gt_budget is not None else (coalition_budget if coalition_budget is not None else nsamples)
        self.prediction_method = prediction_method
        self.target_class = target_class
        self.callable_mode = callable_mode
        self.max_exact_players = max_exact_players
        self.__post_init__()

    def __post_init__(self) -> None:
        provided_scorer = self.scorer
        raw_predictor = self.predictor
        super().__post_init__()
        self.tree_model_detected = any(hasattr(raw_predictor, attribute) for attribute in ("tree_", "estimators_", "get_booster"))
        self.value_function_factory = self.value_function_factory or PredictionValueFunction
        if provided_scorer is None:
            self.scorer = GameTheoreticScorer(nucleolus_solver=Nucleolus(max_exact_players=self.max_exact_players))
        self.cache_mode = "sampled"

    def explain(self, instance, feature_names=None, metadata=None, **kwargs):
        metadata = dict(metadata or {})
        metadata.setdefault("tree_model_detected", self.tree_model_detected)
        metadata.setdefault("tree_cache_strategy", "sampled")
        return super().explain(instance, feature_names=feature_names, metadata=metadata, **kwargs)
