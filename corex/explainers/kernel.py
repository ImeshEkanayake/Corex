from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corex.explainers.base import BaseExplainer
from corex.games.game_theory import BanzhafIndex, GameTheoreticScorer, Nucleolus
from corex.games.value_functions import LossReductionValueFunction, PredictionValueFunction


@dataclass(init=False)
class KernelExplainer(BaseExplainer):
    masker: Any = None
    sample_size: int = 512
    random_seed: int = 0
    value_mode: str = "prediction"
    target: float | None = None
    loss_fn: Any = None

    def __init__(
        self,
        model,
        background=None,
        feature_names=None,
        nsamples: int | None = None,
        gt_budget: int | None = None,
        coalition_budget: int | None = None,
        target_class: int | None = None,
        sample_size: int | None = None,
        random_seed: int = 0,
        baseline=None,
        prediction_method: str | None = None,
        callable_mode: str | None = None,
        masker: Any = None,
        value_mode: str = "prediction",
        target: float | None = None,
        loss_fn: Any = None,
        scorer: GameTheoreticScorer | None = None,
    ) -> None:
        self.predictor = model
        self.baseline = background if background is not None else baseline
        self.default_feature_names = feature_names
        self.scorer = scorer
        self.value_function_factory = None
        self.primary_solution = "banzhaf"
        self.cache_mode = "auto"
        self.sample_budget = (
            gt_budget
            if gt_budget is not None
            else (coalition_budget if coalition_budget is not None else (nsamples if nsamples is not None else sample_size))
        )
        self.prediction_method = prediction_method
        self.target_class = target_class
        self.callable_mode = callable_mode
        self.masker = masker
        self.sample_size = sample_size if sample_size is not None else (nsamples if nsamples is not None else 512)
        self.random_seed = random_seed
        self.value_mode = value_mode
        self.target = target
        self.loss_fn = loss_fn
        self.__post_init__()

    def __post_init__(self) -> None:
        provided_scorer = self.scorer
        super().__post_init__()
        if self.value_function_factory is None:
            self.value_function_factory = LossReductionValueFunction if self.value_mode == "loss" else PredictionValueFunction
        if provided_scorer is None:
            self.scorer = GameTheoreticScorer(
                banzhaf_solver=BanzhafIndex(sample_size=self.sample_size, random_seed=self.random_seed),
                nucleolus_solver=Nucleolus(max_exact_players=10),
            )
        self.cache_mode = "sampled"
        self.sample_budget = self.sample_budget or self.sample_size

    def _build_value_function(self, instance, feature_names=None, metadata=None, **kwargs):
        kwargs.setdefault("masker", self.masker)
        if self.value_mode == "loss":
            kwargs.setdefault("target", 0.0 if self.target is None else self.target)
            kwargs.setdefault("loss_fn", self.loss_fn)
        return super()._build_value_function(instance, feature_names=feature_names, metadata=metadata, **kwargs)
