"""Core-related solver helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from corex.games.game_theory import CooperativeGame, GameSolution, _standardize_solver_diagnostics
from corex.solvers.least_core import LeastCoreSolver


@dataclass
class CoreSolver:
    tolerance: float = 1e-9
    max_exact_players: int = 10

    def solve(self, game: CooperativeGame) -> GameSolution:
        started = time.perf_counter()
        least_core = LeastCoreSolver(tolerance=self.tolerance, max_exact_players=self.max_exact_players).solve(game)
        epsilon = float(least_core.solver_diagnostics.get("epsilon", 0.0))
        feasible = epsilon <= self.tolerance
        allocations = np.asarray(least_core.allocations, dtype=float)
        mode = "exact" if feasible and least_core.exact_or_approximate == "exact" else "approximate"
        return GameSolution(
            solution_concept="core",
            allocations=allocations,
            exact_or_approximate=mode,
            solver_diagnostics=_standardize_solver_diagnostics(
                success=feasible,
                method="least_core_feasibility",
                exact_or_approximate=mode,
                epsilon=epsilon,
                runtime_ms=(time.perf_counter() - started) * 1000.0,
                fallback_used=least_core.exact_or_approximate != "exact",
                tolerance=self.tolerance,
            ),
            game=game,
        )
