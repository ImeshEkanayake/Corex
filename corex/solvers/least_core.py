"""Least-core solver with an exact LP path for small games."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from corex.games.game_theory import (
    BanzhafIndex,
    CooperativeGame,
    GameSolution,
    _standardize_solver_diagnostics,
    solve_least_core_lp,
)


@dataclass
class LeastCoreSolver:
    tolerance: float = 1e-9
    max_exact_players: int = 10

    def _compute_banzhaf_approximation(self, game: CooperativeGame) -> np.ndarray:
        return np.asarray(BanzhafIndex().solve(game).allocations, dtype=float)

    def _banzhaf_fallback(self, game: CooperativeGame) -> GameSolution:
        return self._solve_heuristic(game)

    def solve(self, game: CooperativeGame) -> GameSolution:
        started = time.perf_counter()
        if game.n_players <= self.max_exact_players:
            result = solve_least_core_lp(game, tolerance=self.tolerance)
            if result["success"]:
                return GameSolution(
                    solution_concept="least_core",
                    allocations=result["allocations"],
                    exact_or_approximate="exact",
                    solver_diagnostics=_standardize_solver_diagnostics(
                        success=True,
                        method="linear_program",
                        exact_or_approximate="exact",
                        epsilon=float(result["epsilon"]),
                        runtime_ms=(time.perf_counter() - started) * 1000.0,
                        fallback_used=False,
                        status=int(result["status"]),
                        uses_shapley_fallback=False,
                    ),
                    game=game,
                )

        return self._banzhaf_fallback(game)

    def _solve_heuristic(self, game: CooperativeGame) -> GameSolution:
        started = time.perf_counter()
        grand_value = game.evaluate(game.grand_coalition())
        seed = self._compute_banzhaf_approximation(game)
        seed_total = float(np.sum(seed))
        if abs(seed_total) <= self.tolerance:
            allocations = np.full(game.n_players, grand_value / max(game.n_players, 1), dtype=float)
        else:
            allocations = np.asarray(seed, dtype=float) * grand_value / seed_total
        deficits = []
        for coalition, value in game.coalition_values_indexed().items():
            allocated = float(np.sum(allocations[list(coalition)])) if coalition else 0.0
            deficits.append(value - allocated)
        epsilon = max(deficits) if deficits else 0.0
        return GameSolution(
            solution_concept="least_core",
            allocations=allocations,
            exact_or_approximate="approximate",
            solver_diagnostics=_standardize_solver_diagnostics(
                success=True,
                method="banzhaf_warm_start",
                exact_or_approximate="approximate",
                epsilon=float(epsilon),
                runtime_ms=(time.perf_counter() - started) * 1000.0,
                fallback_used=True,
                uses_shapley_fallback=False,
                reason=f"player_count_exceeds_exact_limit:{self.max_exact_players}",
            ),
            game=game,
        )
