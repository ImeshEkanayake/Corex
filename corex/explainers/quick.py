from __future__ import annotations

from dataclasses import dataclass

from corex.explainers.base import BaseExplainer
from corex.explainers.tree import TreeExplainer
from corex.games.game_theory import BanzhafIndex, GameTheoreticScorer, Nucleolus
from corex.games.value_functions import PredictionValueFunction


@dataclass(init=False)
class QuickExplainer(BaseExplainer):
    sample_size: int = 128
    random_seed: int = 0

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
        self.sample_size = sample_size if sample_size is not None else (nsamples if nsamples is not None else 128)
        self.random_seed = random_seed
        self.__post_init__()

    def __post_init__(self) -> None:
        provided_scorer = self.scorer
        super().__post_init__()
        self.value_function_factory = self.value_function_factory or PredictionValueFunction
        if provided_scorer is None:
            self.scorer = GameTheoreticScorer(
                banzhaf_solver=BanzhafIndex(sample_size=self.sample_size, random_seed=self.random_seed),
                nucleolus_solver=Nucleolus(max_exact_players=8),
            )
        self.cache_mode = "sampled"
        self.sample_budget = self.sample_budget or self.sample_size
