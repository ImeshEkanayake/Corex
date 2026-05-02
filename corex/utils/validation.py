"""Validation and analytics helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from corex.utils.explanation import ExplanationResult


ALPHA_RELEASE_ITEMS = {
    "api_surface_present": True,
    "gt_core_present": True,
    "value_functions_emit_games": True,
    "explainers_emit_gt_results": True,
    "plots_present": True,
    "tests_present": True,
    "docs_version_synced": True,
}


def validate_game_definition(
    game_definition: dict[str, Any] | ExplanationResult,
    solution_concepts: dict[str, Any] | None = None,
    check_efficiency: bool = False,
    check_properties: bool = False,
) -> dict[str, Any]:
    result: ExplanationResult | None = None
    primary_solution_name: str | None = None
    if isinstance(game_definition, ExplanationResult):
        result = game_definition
        solution_concepts = result.solution_concepts
        primary_solution_name = result.primary_solution
        game_definition = result.game_definition

    players = list(game_definition.get("players", []))
    missing = [name for name in ("players", "value_function_name", "game_metadata") if name not in game_definition]
    errors: list[str] = []
    warnings: list[str] = []
    efficiency_checks: dict[str, bool] = {}

    if not players:
        errors.append("players must be non-empty")
    if len(set(players)) != len(players):
        errors.append("players must be unique")
    if "n_players" in game_definition and int(game_definition["n_players"]) != len(players):
        errors.append("n_players must match len(players)")
    if not isinstance(game_definition.get("game_metadata", {}), dict):
        errors.append("game_metadata must be a dict")
    if "grand_coalition_value" in game_definition and not isinstance(game_definition["grand_coalition_value"], (int, float)):
        errors.append("grand_coalition_value must be numeric")
    if solution_concepts is not None and not solution_concepts:
        errors.append("solution_concepts must be non-empty")
    if primary_solution_name is not None and solution_concepts is not None and primary_solution_name not in solution_concepts:
        errors.append("primary_solution must exist in solution_concepts")
    if result is not None and result.game is not None and result.players != players:
        errors.append("attached game player order must match serialized players")

    if solution_concepts:
        for name, solution in solution_concepts.items():
            allocations = np.asarray(getattr(solution, "allocations", []), dtype=float)
            if allocations.shape != (len(players),):
                errors.append(f"{name} allocations must have shape ({len(players)},)")
                continue
            if np.isnan(allocations).any():
                errors.append(f"{name} allocations must not contain NaN values")
            exactness = getattr(solution, "exact_or_approximate", None)
            if exactness not in {"exact", "approximate"}:
                errors.append(f"{name} exact_or_approximate must be 'exact' or 'approximate'")
            diagnostics = getattr(solution, "solver_diagnostics", None)
            if not isinstance(diagnostics, dict):
                errors.append(f"{name} solver_diagnostics must be a dict")
            elif "success" not in diagnostics:
                warnings.append(f"{name} solver_diagnostics is missing 'success'")
            if check_efficiency and "grand_coalition_value" in game_definition and allocations.shape == (len(players),):
                grand_value = float(game_definition["grand_coalition_value"])
                efficient = bool(np.isclose(float(np.sum(allocations)), grand_value, atol=1e-6))
                efficiency_checks[name] = efficient
                if not efficient:
                    warnings.append(f"{name} allocations are not efficient against the grand coalition value")

    property_checks: dict[str, Any] = {}
    if check_properties and result is not None and result.game is not None:
        game = result.game
        null_players = []
        for index, player in enumerate(game.players):
            is_null = True
            for coalition in game.coalition_values_indexed():
                if index in coalition:
                    continue
                with_player = frozenset(set(coalition) | {index})
                if not np.isclose(
                    game.evaluate_indices(with_player),
                    game.evaluate_indices(coalition),
                    atol=1e-8,
                ):
                    is_null = False
                    break
            if is_null:
                null_players.append(player)

        symmetric_pairs = []
        for left_index, left in enumerate(game.players):
            for right_index in range(left_index + 1, game.n_players):
                right = game.players[right_index]
                symmetric = True
                for coalition in game.coalition_values_indexed():
                    if left_index in coalition or right_index in coalition:
                        continue
                    left_value = game.evaluate_indices(frozenset(set(coalition) | {left_index}))
                    right_value = game.evaluate_indices(frozenset(set(coalition) | {right_index}))
                    if not np.isclose(left_value, right_value, atol=1e-8):
                        symmetric = False
                        break
                if symmetric:
                    symmetric_pairs.append((left, right))
        property_checks = {
            "null_players": null_players,
            "symmetric_pairs": symmetric_pairs,
        }

    return {
        "valid": not missing and not errors,
        "all_pass": not missing and not errors,
        "has_game_definition": bool(game_definition),
        "has_players": bool(players),
        "has_value_function_name": "value_function_name" in game_definition,
        "has_solution_concepts": bool(solution_concepts),
        "missing": missing,
        "errors": errors,
        "warnings": warnings,
        "player_count": len(players),
        "primary_solution_present": primary_solution_name in solution_concepts if primary_solution_name and solution_concepts else None,
        "efficiency_checks": efficiency_checks,
        "property_checks": property_checks,
    }


def compare_explainers(*results: ExplanationResult) -> dict[str, Any]:
    if not results:
        return {"comparisons": [], "player_order": []}
    player_order = results[0].players
    for result in results[1:]:
        if result.players != player_order:
            raise ValueError("compare_explainers requires identical feature/player order across results.")
    baseline = np.asarray(results[0].primary_solution_result.allocations, dtype=float)
    baseline_banzhaf = results[0].banzhaf_scores
    baseline_nucleolus = results[0].nucleolus_allocation
    comparisons = []
    for index, result in enumerate(results[1:], start=1):
        allocations = np.asarray(result.primary_solution_result.allocations, dtype=float)
        delta = allocations - baseline
        rank_correlation = spearmanr(baseline, allocations).statistic if len(baseline) > 1 else 1.0
        banzhaf_correlation = None
        if baseline_banzhaf is not None and result.banzhaf_scores is not None:
            banzhaf_correlation = float(np.corrcoef(baseline_banzhaf, result.banzhaf_scores)[0, 1]) if len(baseline_banzhaf) > 1 else 1.0
            if np.isnan(banzhaf_correlation):
                banzhaf_correlation = None
        nucleolus_correlation = None
        if baseline_nucleolus is not None and result.nucleolus_allocation is not None:
            nucleolus_correlation = float(np.corrcoef(baseline_nucleolus, result.nucleolus_allocation)[0, 1]) if len(baseline_nucleolus) > 1 else 1.0
            if np.isnan(nucleolus_correlation):
                nucleolus_correlation = None
        comparisons.append({"index": index, "solution": result.primary_solution, "delta": delta.tolist()})
        comparisons[-1]["banzhaf_correlation"] = banzhaf_correlation
        comparisons[-1]["nucleolus_correlation"] = nucleolus_correlation
        comparisons[-1]["rank_correlation"] = None if np.isnan(rank_correlation) else float(rank_correlation)
        comparisons[-1]["epsilon_comparison"] = {
            "baseline": results[0].epsilon,
            "current": result.epsilon,
        }
        comparisons[-1]["exactness"] = {
            "baseline": results[0].exact_or_approximate,
            "current": result.exact_or_approximate,
        }
    return {"comparisons": comparisons, "player_order": player_order}


def compute_coalition_interactions(result: ExplanationResult) -> dict[str, float]:
    if result.game is None:
        raise ValueError("compute_coalition_interactions requires an ExplanationResult with an attached game object.")

    game = result.game
    empty_value = game.evaluate_indices(frozenset())
    interactions: dict[str, float] = {}
    for left_index, right_index in combinations(range(game.n_players), 2):
        pair_value = game.evaluate_indices(frozenset({left_index, right_index}))
        left_value = game.evaluate_indices(frozenset({left_index}))
        right_value = game.evaluate_indices(frozenset({right_index}))
        interactions[f"{game.player_name(left_index)}|{game.player_name(right_index)}"] = float(
            pair_value - left_value - right_value + empty_value
        )
    return interactions


def release_gate_status() -> dict[str, Any]:
    ready = all(ALPHA_RELEASE_ITEMS.values())
    return {"ready": ready, "items": dict(ALPHA_RELEASE_ITEMS)}
