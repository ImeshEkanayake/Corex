"""Coalition explainer with least-core primary support."""

from __future__ import annotations

from dataclasses import dataclass

from corex.explainers.base import BaseExplainer
from corex.games.game_theory import GameTheoreticScorer
from corex.games.value_functions import PredictionValueFunction
from corex.solvers.core import CoreSolver
from corex.solvers.least_core import LeastCoreSolver


@dataclass(init=False)
class CoalitionExplainer(BaseExplainer):
    primary_solution: str = "least_core"
    max_exact_players: int = 10

    def __init__(
        self,
        model,
        background=None,
        feature_names=None,
        solution: str | None = None,
        sampling_budget: int | None = None,
        gt_budget: int | None = None,
        max_exact_players: int = 10,
        target_class: int | None = None,
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
        self.primary_solution = solution or "least_core"
        self.cache_mode = "auto"
        self.sample_budget = sampling_budget if sampling_budget is not None else gt_budget
        self.prediction_method = prediction_method
        self.target_class = target_class
        self.callable_mode = callable_mode
        self.max_exact_players = max_exact_players
        self.__post_init__()

    def __post_init__(self) -> None:
        provided_scorer = self.scorer
        super().__post_init__()
        self.value_function_factory = self.value_function_factory or PredictionValueFunction
        extra_solvers = {
            "least_core": LeastCoreSolver(max_exact_players=self.max_exact_players),
            "core": CoreSolver(max_exact_players=self.max_exact_players),
        }
        if provided_scorer is None:
            self.scorer = GameTheoreticScorer(extra_solvers=extra_solvers)

    def explain(self, instance, feature_names=None, metadata=None, **kwargs):
        game = self._build_game(instance, feature_names=feature_names, metadata=metadata, **kwargs)
        solutions = self._score_game(game)
        return self._make_result(game, solutions, primary_solution=self.primary_solution, metadata=metadata)
