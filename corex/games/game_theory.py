"""Core cooperative game and solver primitives."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
import random
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import linprog


Coalition = frozenset[int]
NamedCoalition = frozenset[str]
MAX_SAFE_EXHAUSTIVE_PLAYERS = 12


def _powerset_indices(count: int, min_size: int = 0, max_size: int | None = None) -> Iterable[tuple[int, ...]]:
    upper = count if max_size is None else min(count, max_size)
    for size in range(max(min_size, 0), upper + 1):
        for coalition in combinations(range(count), size):
            yield coalition


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, dict):
        return {str(key): _serialize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, set):
        return [_serialize_value(item) for item in sorted(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _coalition_indicator(coalition: Coalition, n_players: int) -> np.ndarray:
    row = np.zeros(n_players, dtype=float)
    for index in coalition:
        row[index] = 1.0
    return row


def _coalition_label(players: list[str], coalition: Coalition) -> str:
    names = [players[index] for index in sorted(coalition)]
    return "{" + ",".join(names) + "}"


def _standardize_solver_diagnostics(
    *,
    success: bool,
    method: str,
    exact_or_approximate: str,
    epsilon: float | None = None,
    samples: int | None = None,
    runtime_ms: float | None = None,
    fallback_used: bool = False,
    warning: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    diagnostics = {
        "success": bool(success),
        "method": method,
        "exact_or_approximate": exact_or_approximate,
        "epsilon": None if epsilon is None else float(epsilon),
        "samples": None if samples is None else int(samples),
        "runtime_ms": None if runtime_ms is None else float(runtime_ms),
        "fallback_used": bool(fallback_used),
        "warning": warning,
    }
    diagnostics.update(extra)
    return diagnostics


@dataclass
class CooperativeGame:
    players: list[str]
    value_function: Callable[[NamedCoalition | Coalition], float]
    value_function_name: str = "value_function"
    game_metadata: dict[str, Any] | None = None
    cache_mode: str = "auto"
    sample_budget: int | None = None
    coalition_representation: str = "names"
    max_cache_size: int | None = None
    _coalition_cache: OrderedDict[Coalition, float] = field(default_factory=OrderedDict, init=False, repr=False)
    _player_to_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.players = [str(player) for player in self.players]
        if not self.players:
            raise ValueError("CooperativeGame requires at least one player.")
        if len(set(self.players)) != len(self.players):
            raise ValueError("CooperativeGame players must be unique.")
        self._player_to_index = {player: index for index, player in enumerate(self.players)}
        self.game_metadata = dict(self.game_metadata or {})
        self.sample_budget = self.sample_budget or max(32, len(self.players) * 16)
        if self.cache_mode == "auto":
            self.cache_mode = "exhaustive" if len(self.players) <= 10 else "sampled"
        if self.max_cache_size is None:
            self.max_cache_size = None if self.cache_mode == "exhaustive" else max(128, int(self.sample_budget) * 2)
        if self.coalition_representation not in {"names", "indices"}:
            raise ValueError("coalition_representation must be 'names' or 'indices'.")
        if self.cache_mode == "exhaustive":
            self._ensure_exhaustive_cache_is_safe()

    @property
    def n_players(self) -> int:
        return len(self.players)

    def player_index(self, player: str) -> int:
        return self._player_to_index[player]

    def player_name(self, index: int) -> str:
        return self.players[index]

    def coalition_to_indices(self, coalition: Iterable[str | int]) -> Coalition:
        if isinstance(coalition, frozenset) and not coalition:
            return frozenset()
        indices: set[int] = set()
        for member in coalition:
            if isinstance(member, int):
                if member < 0 or member >= self.n_players:
                    raise IndexError(f"Coalition index {member} is out of bounds for {self.n_players} players.")
                indices.add(member)
            else:
                if member not in self._player_to_index:
                    raise KeyError(f"Unknown player: {member}")
                indices.add(self._player_to_index[member])
        return frozenset(indices)

    def coalition_to_names(self, coalition: Iterable[int]) -> NamedCoalition:
        return frozenset(self.players[index] for index in coalition)

    def _call_value_function(self, coalition: Coalition) -> float:
        if self.coalition_representation == "indices":
            value = self.value_function(coalition)
        else:
            value = self.value_function(self.coalition_to_names(coalition))
        return float(value)

    def _enforce_cache_limit(self) -> None:
        if self.cache_mode == "exhaustive" or self.max_cache_size is None:
            return
        pinned = {frozenset(), self.grand_coalition_indices()}
        while len(self._coalition_cache) > self.max_cache_size:
            oldest_key = next(iter(self._coalition_cache))
            if oldest_key in pinned and len(self._coalition_cache) > len(pinned):
                self._coalition_cache.move_to_end(oldest_key)
                continue
            self._coalition_cache.pop(oldest_key, None)

    def _ensure_exhaustive_cache_is_safe(self) -> None:
        if self.n_players <= MAX_SAFE_EXHAUSTIVE_PLAYERS:
            return
        if bool(self.game_metadata.get("allow_exhaustive_cache")):
            return
        raise ValueError(
            "Exhaustive coalition caching is disabled for games with more than "
            f"{MAX_SAFE_EXHAUSTIVE_PLAYERS} players. Use cache_mode='sampled' or "
            "set game_metadata={'allow_exhaustive_cache': True} to opt in explicitly."
        )

    def evaluate_indices(self, coalition: Coalition) -> float:
        coalition_key = frozenset(coalition)
        cached = self._coalition_cache.get(coalition_key)
        if cached is not None:
            if self.cache_mode != "exhaustive":
                self._coalition_cache.move_to_end(coalition_key)
            return cached
        self._coalition_cache[coalition_key] = self._call_value_function(coalition_key)
        self._enforce_cache_limit()
        return self._coalition_cache[coalition_key]

    def evaluate(self, coalition: Iterable[str | int]) -> float:
        return self.evaluate_indices(self.coalition_to_indices(coalition))

    def coalition_cache_size(self) -> int:
        return len(self._coalition_cache)

    def grand_coalition_indices(self) -> Coalition:
        return frozenset(range(self.n_players))

    def grand_coalition(self) -> NamedCoalition:
        return self.coalition_to_names(self.grand_coalition_indices())

    def coalition_values_indexed(self) -> dict[Coalition, float]:
        if self.cache_mode == "exhaustive":
            for coalition in _powerset_indices(self.n_players):
                self.evaluate_indices(frozenset(coalition))
        return dict(self._coalition_cache)

    @property
    def coalition_values(self) -> dict[Coalition, float]:
        return self.coalition_values_indexed()

    def named_coalition_values(self) -> dict[NamedCoalition, float]:
        return {self.coalition_to_names(coalition): value for coalition, value in self.coalition_values_indexed().items()}

    def build_coalition_cache(self, max_samples: int | None = None) -> dict[Coalition, float]:
        if max_samples is None or max_samples >= 2 ** self.n_players:
            self._ensure_exhaustive_cache_is_safe()
            for coalition in _powerset_indices(self.n_players):
                self.evaluate_indices(frozenset(coalition))
            return dict(self._coalition_cache)
        generator = random.Random(0)
        sampled: set[Coalition] = {frozenset(), self.grand_coalition_indices()}
        while len(sampled) < max_samples:
            size = generator.randint(0, self.n_players)
            coalition = frozenset(generator.sample(range(self.n_players), k=size))
            sampled.add(coalition)
        for coalition in sampled:
            self.evaluate_indices(coalition)
        return dict(self._coalition_cache)

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": list(self.players),
            "n_players": self.n_players,
            "value_function_name": self.value_function_name,
            "game_metadata": dict(self.game_metadata),
            "cache_mode": self.cache_mode,
            "sample_budget": self.sample_budget,
            "max_cache_size": self.max_cache_size,
            "coalition_representation": self.coalition_representation,
            "coalition_cache_size": self.coalition_cache_size(),
            "grand_coalition_value": self.evaluate_indices(self.grand_coalition_indices()),
        }


def summarize_coalitions_by_size(
    game: CooperativeGame,
    min_size: int = 2,
    max_size: int = 10,
    top_n: int = 5,
    max_coalitions_per_size: int | None = 2048,
    random_seed: int = 0,
) -> dict[str, Any]:
    player_count = game.n_players
    lower = max(min_size, 0)
    upper = min(max_size, player_count)
    generator = random.Random(random_seed)
    summary: dict[str, Any] = {
        "player_count": player_count,
        "requested_size_range": [int(min_size), int(max_size)],
        "evaluated_size_range": [int(lower), int(upper)],
        "sizes": {},
    }
    for size in range(lower, upper + 1):
        coalition_records = []
        total_coalitions = comb(player_count, size)
        sampled = max_coalitions_per_size is not None and total_coalitions > max_coalitions_per_size
        if sampled:
            seen: set[tuple[str, ...]] = set()
            coalition_iterable = []
            while len(coalition_iterable) < int(max_coalitions_per_size):
                coalition = tuple(sorted(generator.sample(game.players, size)))
                if coalition in seen:
                    continue
                seen.add(coalition)
                coalition_iterable.append(coalition)
        else:
            coalition_iterable = combinations(game.players, size)
        for coalition in coalition_iterable:
            coalition_names = frozenset(coalition)
            coalition_records.append(
                {
                    "coalition": list(coalition),
                    "value": float(game.evaluate(coalition_names)),
                }
            )
        coalition_records.sort(key=lambda item: (-item["value"], item["coalition"]))
        values = [item["value"] for item in coalition_records]
        if coalition_records:
            best = coalition_records[0]
            worst = coalition_records[-1]
            average = sum(values) / len(values)
        else:
            best = {"coalition": [], "value": 0.0}
            worst = {"coalition": [], "value": 0.0}
            average = 0.0
        summary["sizes"][str(size)] = {
            "coalition_count": len(coalition_records),
            "total_coalition_count": int(total_coalitions),
            "sampled": bool(sampled),
            "average_value": float(average),
            "best_coalition": best,
            "best_value": float(best["value"]),
            "worst_coalition": worst,
            "worst_value": float(worst["value"]),
            "top_coalitions": coalition_records[:top_n],
        }
    return summary


@dataclass(init=False)
class GameSolution:
    solution_concept: str
    allocations: np.ndarray | Iterable[float]
    exact_or_approximate: str
    solver_diagnostics: dict[str, Any]
    game: CooperativeGame | None = None

    def __init__(
        self,
        solution_concept: str | None = None,
        allocations: np.ndarray | Iterable[float] | None = None,
        exact_or_approximate: str | CooperativeGame | None = None,
        solver_diagnostics: dict[str, Any] | str | None = None,
        game: CooperativeGame | dict[str, Any] | None = None,
        concept: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if concept is not None and solution_concept is None:
            solution_concept = concept
        if diagnostics is not None and solver_diagnostics is None:
            solver_diagnostics = diagnostics
        if isinstance(exact_or_approximate, CooperativeGame):
            compatibility_game = exact_or_approximate
            compatibility_mode = solver_diagnostics
            compatibility_diagnostics = game
            game = compatibility_game
            exact_or_approximate = compatibility_mode
            solver_diagnostics = compatibility_diagnostics
        if solution_concept is None or allocations is None or exact_or_approximate is None:
            raise ValueError("GameSolution requires solution_concept/concept, allocations, and exact_or_approximate.")
        if solver_diagnostics is None:
            solver_diagnostics = {}
        self.solution_concept = str(solution_concept)
        self.allocations = allocations
        self.exact_or_approximate = str(exact_or_approximate)
        self.solver_diagnostics = dict(solver_diagnostics)
        self.game = game if isinstance(game, CooperativeGame) or game is None else None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.allocations = _as_float_array(self.allocations)
        self.solver_diagnostics = dict(self.solver_diagnostics or {})

    @property
    def epsilon(self) -> float | None:
        value = self.solver_diagnostics.get("epsilon")
        if value is None and self.solution_concept == "nucleolus":
            value = self.solver_diagnostics.get("max_excess")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_concept": self.solution_concept,
            "allocations": self.allocations.astype(float).tolist(),
            "exact_or_approximate": self.exact_or_approximate,
            "solver_diagnostics": _serialize_value(self.solver_diagnostics),
            "has_game": self.game is not None,
        }


class BanzhafIndex:
    def __init__(
        self,
        sample_size: int | None = None,
        random_seed: int = 0,
        sampling_budget: int | None = None,
        threshold: float | None = None,
        mode: str = "marginal",
    ) -> None:
        self.sample_size = sample_size if sample_size is not None else sampling_budget
        self.random_seed = random_seed
        self.sampling_budget = self.sample_size
        self.threshold = threshold
        self.mode = mode

    def solve(self, game: CooperativeGame) -> GameSolution:
        started = time.perf_counter()
        if self.sample_size is None and game.n_players <= 10:
            allocations = self._solve_exact(game)
            mode = "exact"
            diagnostics = _standardize_solver_diagnostics(
                success=True,
                method="exact_enumeration",
                exact_or_approximate=mode,
                samples=0,
                threshold=self.threshold,
                mode=self.mode,
            )
        else:
            samples = self.sample_size or game.sample_budget or 128
            allocations = self._solve_sampled(game, samples=samples)
            mode = "approximate"
            diagnostics = _standardize_solver_diagnostics(
                success=True,
                method="monte_carlo",
                exact_or_approximate=mode,
                samples=samples,
                seed=self.random_seed,
                threshold=self.threshold,
                mode=self.mode,
            )
        diagnostics["runtime_ms"] = (time.perf_counter() - started) * 1000.0
        return GameSolution("banzhaf", allocations, mode, diagnostics, game=game)

    def _solve_exact(self, game: CooperativeGame) -> np.ndarray:
        raw = np.zeros(game.n_players, dtype=float)
        for coalition in _powerset_indices(game.n_players):
            coalition_indices = frozenset(coalition)
            coalition_value = game.evaluate_indices(coalition_indices)
            coalition_set = set(coalition)
            for index in range(game.n_players):
                if index in coalition_set:
                    continue
                raw[index] += game.evaluate_indices(frozenset(set(coalition_indices) | {index})) - coalition_value
        scale = 1.0 / (2 ** max(game.n_players - 1, 0))
        return raw * scale

    def _solve_sampled(self, game: CooperativeGame, samples: int) -> np.ndarray:
        generator = random.Random(self.random_seed)
        raw = np.zeros(game.n_players, dtype=float)
        for _ in range(samples):
            coalition_indices = frozenset(index for index in range(game.n_players) if generator.random() < 0.5)
            coalition_value = game.evaluate_indices(coalition_indices)
            for index in range(game.n_players):
                if index in coalition_indices:
                    continue
                raw[index] += game.evaluate_indices(frozenset(set(coalition_indices) | {index})) - coalition_value
        return (2.0 * raw) / max(samples, 1)


def solve_least_core_lp(
    game: CooperativeGame,
    fixed_constraints: list[tuple[Coalition, float]] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    fixed_constraints = list(fixed_constraints or [])
    n_players = game.n_players
    grand_value = game.evaluate_indices(game.grand_coalition_indices())

    objective = np.zeros(n_players + 1, dtype=float)
    objective[-1] = 1.0

    coalitions = [frozenset(coalition) for coalition in _powerset_indices(n_players, min_size=1, max_size=n_players - 1)]
    fixed_coalitions = {coalition for coalition, _ in fixed_constraints}

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for coalition in coalitions:
        if coalition in fixed_coalitions:
            continue
        row = np.zeros(n_players + 1, dtype=float)
        row[:n_players] = -_coalition_indicator(coalition, n_players)
        row[-1] = -1.0
        a_ub.append(row)
        b_ub.append(-game.evaluate_indices(coalition))

    a_eq: list[np.ndarray] = [np.append(np.ones(n_players, dtype=float), 0.0)]
    b_eq: list[float] = [grand_value]
    for coalition, excess_value in fixed_constraints:
        row = np.zeros(n_players + 1, dtype=float)
        row[:n_players] = _coalition_indicator(coalition, n_players)
        a_eq.append(row)
        b_eq.append(game.evaluate_indices(coalition) - float(excess_value))

    result = linprog(
        c=objective,
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
        A_eq=np.vstack(a_eq),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=[(None, None)] * (n_players + 1),
        method="highs",
    )

    if not result.success:
        return {
            "success": False,
            "status": int(result.status),
            "message": result.message,
        }

    solution_vector = np.asarray(result.x, dtype=float)
    allocations = solution_vector[:n_players]
    epsilon = float(solution_vector[-1])
    excess_by_coalition = _coalition_excesses(game, allocations)
    active_coalitions = []
    for coalition in coalitions:
        excess = game.evaluate_indices(coalition) - float(np.sum(allocations[list(coalition)]))
        if abs(excess - epsilon) <= max(tolerance, 1e-7):
            active_coalitions.append(coalition)

    return {
        "success": True,
        "allocations": allocations,
        "epsilon": epsilon,
        "status": int(result.status),
        "message": result.message,
        "coalitions": coalitions,
        "active_coalitions": active_coalitions,
        "excess_by_coalition": excess_by_coalition,
    }


def _select_independent_coalitions(
    candidate_coalitions: list[Coalition],
    existing_rows: list[np.ndarray],
    n_players: int,
) -> list[Coalition]:
    selected: list[Coalition] = []
    base_rank = np.linalg.matrix_rank(np.vstack(existing_rows))
    rows = list(existing_rows)
    for coalition in candidate_coalitions:
        row = _coalition_indicator(coalition, n_players)
        candidate_rank = np.linalg.matrix_rank(np.vstack(rows + [row]))
        if candidate_rank > base_rank:
            selected.append(coalition)
            rows.append(row)
            base_rank = candidate_rank
        if base_rank >= n_players:
            break
    return selected


def _heuristic_nucleolus(game: CooperativeGame) -> GameSolution:
    started = time.perf_counter()
    grand_value = game.evaluate_indices(game.grand_coalition_indices())
    singleton_values = np.asarray([game.evaluate({player}) for player in game.players], dtype=float)
    remainder = grand_value - float(np.sum(singleton_values))
    banzhaf = BanzhafIndex(sample_size=None).solve(game).allocations
    positive_mass = np.clip(banzhaf, 0.0, None)
    total_mass = float(np.sum(positive_mass))
    if total_mass > 0.0:
        residual = remainder * positive_mass / total_mass
    else:
        residual = np.full(game.n_players, remainder / max(game.n_players, 1), dtype=float)
    allocations = singleton_values + residual
    excesses = _coalition_excesses(game, allocations)
    max_dissatisfaction = max(excesses.values()) if excesses else 0.0
    exact = _is_additive_game(game)
    mode = "exact" if exact else "approximate"
    diagnostics = _standardize_solver_diagnostics(
        success=True,
        method="singleton_plus_banzhaf_residual",
        exact_or_approximate=mode,
        epsilon=float(max_dissatisfaction),
        runtime_ms=(time.perf_counter() - started) * 1000.0,
        fallback_used=not exact,
        max_excess=float(max_dissatisfaction),
        excess_values=sorted(excesses.values()),
        excess_by_coalition=excesses,
        heuristic=True,
    )
    return GameSolution("nucleolus", allocations, mode, diagnostics, game=game)


class Nucleolus:
    """Lexicographic excess minimization with an exact small-game LP path."""

    def __init__(self, max_exact_players: int = 10, tolerance: float = 1e-9) -> None:
        self.max_exact_players = max_exact_players
        self.tolerance = tolerance

    def solve(self, game: CooperativeGame) -> GameSolution:
        started = time.perf_counter()
        if game.n_players > self.max_exact_players:
            heuristic = _heuristic_nucleolus(game)
            heuristic.solver_diagnostics["reason"] = f"player_count_exceeds_exact_limit:{self.max_exact_players}"
            heuristic.solver_diagnostics["fallback_used"] = True
            return heuristic

        fixed_constraints: list[tuple[Coalition, float]] = []
        existing_rows = [np.ones(game.n_players, dtype=float)]
        lexicographic_epsilons: list[float] = []
        last_result: dict[str, Any] | None = None

        while np.linalg.matrix_rank(np.vstack(existing_rows)) < game.n_players:
            result = solve_least_core_lp(game, fixed_constraints=fixed_constraints, tolerance=self.tolerance)
            if not result["success"]:
                heuristic = _heuristic_nucleolus(game)
                heuristic.solver_diagnostics["fallback_from"] = "exact_iterative_lp"
                heuristic.solver_diagnostics["lp_failure"] = {"status": result.get("status"), "message": result.get("message")}
                heuristic.solver_diagnostics["runtime_ms"] = (time.perf_counter() - started) * 1000.0
                heuristic.solver_diagnostics["fallback_used"] = True
                return heuristic

            last_result = result
            epsilon = float(result["epsilon"])
            lexicographic_epsilons.append(epsilon)
            active = [coalition for coalition in result["active_coalitions"] if coalition not in {item[0] for item in fixed_constraints}]
            selected = _select_independent_coalitions(active, existing_rows, game.n_players)
            if not selected:
                break
            for coalition in selected:
                fixed_constraints.append((coalition, epsilon))
                existing_rows.append(_coalition_indicator(coalition, game.n_players))

        if last_result is None:
            return _heuristic_nucleolus(game)

        allocations = np.asarray(last_result["allocations"], dtype=float)
        excesses = _coalition_excesses(game, allocations)
        max_excess = max(excesses.values()) if excesses else 0.0
        exact = np.linalg.matrix_rank(np.vstack(existing_rows)) >= game.n_players
        mode = "exact" if exact else "approximate"
        diagnostics = _standardize_solver_diagnostics(
            success=True,
            method="exact_iterative_lp" if exact else "partial_iterative_lp",
            exact_or_approximate=mode,
            epsilon=float(last_result["epsilon"]),
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            fallback_used=not exact,
            max_excess=float(max_excess),
            excess_values=sorted(excesses.values()),
            excess_by_coalition=excesses,
            lexicographic_epsilons=lexicographic_epsilons,
            fixed_coalition_count=len(fixed_constraints),
            max_exact_players=self.max_exact_players,
        )
        return GameSolution("nucleolus", allocations, mode, diagnostics, game=game)


def _coalition_excesses(game: CooperativeGame, allocations: np.ndarray | Iterable[float]) -> dict[str, float]:
    allocations_array = _as_float_array(allocations)
    excesses: dict[str, float] = {}
    for coalition in _powerset_indices(game.n_players):
        coalition_key = frozenset(coalition)
        allocated = float(np.sum(allocations_array[list(coalition)])) if coalition else 0.0
        excesses[_coalition_label(game.players, coalition_key)] = game.evaluate_indices(coalition_key) - allocated
    return excesses


def _is_additive_game(game: CooperativeGame, tolerance: float = 1e-9) -> bool:
    singleton_total = {index: game.evaluate_indices(frozenset({index})) for index in range(game.n_players)}
    for coalition in _powerset_indices(game.n_players):
        coalition_key = frozenset(coalition)
        if abs(game.evaluate_indices(coalition_key) - sum(singleton_total[index] for index in coalition_key)) > tolerance:
            return False
    return True


class GameTheoreticScorer:
    def __init__(
        self,
        banzhaf_solver: BanzhafIndex | None = None,
        nucleolus_solver: Nucleolus | None = None,
        extra_solvers: dict[str, Any] | None = None,
        concepts: list[str] | None = None,
        sampling_budget: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self.banzhaf_solver = banzhaf_solver or BanzhafIndex(sampling_budget=sampling_budget, threshold=threshold)
        self.nucleolus_solver = nucleolus_solver or Nucleolus()
        self.extra_solvers = extra_solvers or {}
        self.concepts = list(concepts) if concepts is not None else None

    def score(self, game: CooperativeGame) -> dict[str, GameSolution]:
        requested = self.concepts or ["banzhaf", "nucleolus", *self.extra_solvers.keys()]
        solutions: dict[str, GameSolution] = {}
        for name in requested:
            if name == "banzhaf":
                solutions[name] = self.banzhaf_solver.solve(game)
            elif name == "nucleolus":
                solutions[name] = self.nucleolus_solver.solve(game)
            elif name in self.extra_solvers:
                solutions[name] = self.extra_solvers[name].solve(game)
            elif name == "least_core":
                from corex.solvers.least_core import LeastCoreSolver

                solutions[name] = LeastCoreSolver().solve(game)
            elif name == "core":
                from corex.solvers.core import CoreSolver

                solutions[name] = CoreSolver().solve(game)
            else:
                raise ValueError(f"Unsupported concept: {name}")
        return solutions
