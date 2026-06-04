"""Serializable GT result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from corex.games.game_theory import CooperativeGame, GameSolution


@dataclass
class ExplanationResult:
    game_definition: dict[str, Any]
    solution_concepts: dict[str, GameSolution]
    primary_solution: str = "banzhaf"
    secondary_solutions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    game: CooperativeGame | None = field(default=None, repr=False)

    @classmethod
    def from_game(
        cls,
        game: CooperativeGame,
        solutions: dict[str, GameSolution],
        primary_solution: str = "banzhaf",
        metadata: dict[str, Any] | None = None,
    ) -> "ExplanationResult":
        return cls(
            game_definition=game.to_dict(),
            solution_concepts=solutions,
            primary_solution=primary_solution,
            secondary_solutions=[name for name in solutions if name != primary_solution],
            metadata=dict(metadata or {}),
            game=game,
        )

    @property
    def players(self) -> list[str]:
        if self.game is not None:
            return list(self.game.players)
        if "players" in self.game_definition:
            return list(self.game_definition["players"])
        if "proxy_game" in self.game_definition:
            return list(self.game_definition["proxy_game"].get("players", []))
        return []

    @property
    def primary_solution_result(self) -> GameSolution:
        return self.solution_concepts[self.primary_solution]

    @property
    def feature_values(self) -> np.ndarray:
        return np.asarray(self.primary_solution_result.allocations, dtype=float)

    @property
    def banzhaf_scores(self) -> np.ndarray | None:
        solution = self.solution_concepts.get("banzhaf")
        return None if solution is None else np.asarray(solution.allocations, dtype=float)

    @property
    def nucleolus_allocation(self) -> np.ndarray | None:
        solution = self.solution_concepts.get("nucleolus")
        return None if solution is None else np.asarray(solution.allocations, dtype=float)

    @property
    def epsilon(self) -> float | None:
        if "least_core" in self.solution_concepts:
            return self.solution_concepts["least_core"].epsilon
        return self.primary_solution_result.epsilon

    @property
    def exact_or_approximate(self) -> str:
        return self.primary_solution_result.exact_or_approximate

    def summary(self) -> dict[str, Any]:
        primary = self.primary_solution_result
        grand_coalition_value = self.game_definition.get("grand_coalition_value")
        allocations_by_player = {
            player: float(value)
            for player, value in zip(self.players, primary.allocations.astype(float).tolist())
        }
        ranked = sorted(allocations_by_player.items(), key=lambda item: item[1], reverse=True)
        return {
            "players": self.players,
            "player_count": len(self.players),
            "value_function": self.game_definition.get("value_function_name", "composite"),
            "grand_coalition_value": grand_coalition_value,
            "primary_solution": self.primary_solution,
            "allocations": primary.allocations.astype(float).tolist(),
            "allocations_by_player": allocations_by_player,
            "epsilon": self.epsilon,
            "max_excess": float(self.solution_concepts["nucleolus"].solver_diagnostics.get("max_excess", 0.0))
            if "nucleolus" in self.solution_concepts
            else None,
            "exact_or_approximate": primary.exact_or_approximate,
            "solver_diagnostics": dict(primary.solver_diagnostics),
            "available_solutions": list(self.solution_concepts),
            "top_positive_contributors": [{"player": name, "value": float(value)} for name, value in ranked[:3]],
            "top_negative_contributors": [
                {"player": name, "value": float(value)}
                for name, value in sorted(allocations_by_player.items(), key=lambda item: item[1])[:3]
            ],
            "metadata": dict(self.metadata),
        }

    def validate(self, check_efficiency: bool = False, check_properties: bool = False) -> dict[str, Any]:
        from corex.utils.validation import validate_game_definition

        return validate_game_definition(
            self,
            check_efficiency=check_efficiency,
            check_properties=check_properties,
        )

    def plot_banzhaf(self, show: bool = False):
        from corex.plots.banzhaf import plot_banzhaf

        fig = plot_banzhaf(self)
        if show:
            fig.show()
        return fig

    def plot_nucleolus(self, show: bool = False):
        from corex.plots.nucleolus import plot_nucleolus

        fig = plot_nucleolus(self)
        if show:
            fig.show()
        return fig

    def plot_stability(self, show: bool = False):
        from corex.plots.stability import plot_stability

        fig = plot_stability(self)
        if show:
            fig.show()
        return fig

    def plot_excess_distribution(self, show: bool = False):
        from corex.plots.stability import plot_excess_distribution

        fig = plot_excess_distribution(self)
        if show:
            fig.show()
        return fig

    def plot_force(self, max_display: int = 10, show: bool = False):
        from corex.plots.summary import force_plot

        return force_plot(self, max_display=max_display, show=show)

    def plot_waterfall(
        self,
        max_display: int = 10,
        orientation: str = "vertical",
        show: bool = False,
    ):
        from corex.plots.summary import waterfall_plot

        return waterfall_plot(self, max_display=max_display, orientation=orientation, show=show)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_definition": dict(self.game_definition),
            "players": list(self.players),
            "feature_values": self.feature_values.astype(float).tolist(),
            "banzhaf_scores": None if self.banzhaf_scores is None else self.banzhaf_scores.astype(float).tolist(),
            "nucleolus_allocation": None if self.nucleolus_allocation is None else self.nucleolus_allocation.astype(float).tolist(),
            "epsilon": self.epsilon,
            "exact_or_approximate": self.exact_or_approximate,
            "solution_concepts": {name: solution.to_dict() for name, solution in self.solution_concepts.items()},
            "primary_solution": self.primary_solution,
            "secondary_solutions": list(self.secondary_solutions),
            "summary": self.summary(),
            "metadata": dict(self.metadata),
        }


@dataclass
class ProxyAuditReport:
    evaluation_mode: str
    attribute_reports: dict[str, ExplanationResult | dict[str, ExplanationResult]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _attribute_modes(self, attribute_name: str) -> dict[str, ExplanationResult]:
        if attribute_name not in self.attribute_reports:
            raise KeyError(f"Unknown attribute report: {attribute_name}")
        report = self.attribute_reports[attribute_name]
        if isinstance(report, ExplanationResult):
            return {self.evaluation_mode if self.evaluation_mode in {"proxy_model", "ablation"} else "proxy_model": report}
        return dict(report)

    def summary(self) -> dict[str, Any]:
        attribute_summaries = {}
        for name in self.attribute_reports:
            modes = self._attribute_modes(name)
            attribute_summaries[name] = {
                "available_modes": self.available_modes(name),
                "epsilon": self.epsilon(name),
                "local_fairness_risk": self.local_fairness_risk(name),
                "top_proxy_coalitions": self.top_proxy_coalitions(name),
                "mode_summaries": {
                    mode: {
                        "epsilon": report.epsilon,
                        "local_fairness_risk": self.local_fairness_risk(name, mode=mode),
                        "dataset_summary": report.metadata.get("dataset_summary"),
                        "top_proxy_coalitions": self.top_proxy_coalitions(name, mode=mode),
                    }
                    for mode, report in modes.items()
                },
            }
        return {
            "evaluation_mode": self.evaluation_mode,
            "attributes": list(self.attribute_reports),
            "attribute_count": len(self.attribute_reports),
            "audit_scope": self.metadata.get("audit_scope", "local"),
            "top_k": self.metadata.get("top_k", self.metadata.get("top_n", 10)),
            "coalition_size_range": self.metadata.get("coalition_size_range"),
            "max_features": self.metadata.get("max_features"),
            "attribute_summaries": attribute_summaries,
            "metadata": dict(self.metadata),
        }

    def _attribute_report(self, attribute_name: str, mode: str | None = None) -> ExplanationResult:
        modes = self._attribute_modes(attribute_name)
        if mode is None:
            if "proxy_model" in modes:
                return modes["proxy_model"]
            return next(iter(modes.values()))
        if mode not in modes:
            raise KeyError(f"Unknown mode {mode!r} for attribute {attribute_name!r}")
        return modes[mode]

    def banzhaf_scores(self, attribute_name: str, mode: str | None = None):
        return self._attribute_report(attribute_name, mode=mode).banzhaf_scores

    def nucleolus_allocation(self, attribute_name: str, mode: str | None = None):
        return self._attribute_report(attribute_name, mode=mode).nucleolus_allocation

    def epsilon(self, attribute_name: str, mode: str | None = None) -> float | None:
        return self._attribute_report(attribute_name, mode=mode).epsilon

    def local_fairness_risk(self, attribute_name: str, mode: str | None = None) -> float | None:
        details = self.local_fairness_risk_details(attribute_name, mode=mode)
        if details is None:
            return None
        return float(details["lfr"])

    def local_fairness_risk_details(self, attribute_name: str, mode: str | None = None) -> dict[str, Any] | None:
        report = self._attribute_report(attribute_name, mode=mode)
        details = report.metadata.get("local_fairness_risk")
        return dict(details) if isinstance(details, dict) else None

    def available_modes(self, attribute_name: str) -> list[str]:
        return sorted(self._attribute_modes(attribute_name))

    def top_proxy_coalitions(self, attribute_name: str, mode: str | None = None, size: int | None = None) -> list[dict[str, Any]]:
        report = self._attribute_report(attribute_name, mode=mode)
        summary = report.metadata.get("coalition_size_summary") or {}
        if isinstance(summary, dict) and "proxy_model" in summary:
            summary = summary.get("proxy_model") or {}
        sizes = summary.get("sizes", {}) if isinstance(summary, dict) else {}
        if size is not None:
            size_summary = sizes.get(str(size), {})
            return list(size_summary.get("top_coalitions", []))
        collected = []
        for key in sorted(sizes, key=lambda value: int(value)):
            collected.extend(sizes[key].get("top_coalitions", []))
        return collected

    def plot_proxy_overlaps(
        self,
        top_k: int | None = None,
        dataset_name: str | None = None,
        target_description: str | None = None,
        show: bool = False,
    ):
        from corex.plots.proxy import plot_proxy_overlaps

        fig = plot_proxy_overlaps(
            self,
            top_k=top_k,
            dataset_name=dataset_name,
            target_description=target_description,
        )
        if show:
            fig.show()
        return fig

    def plot_banzhaf(self, attribute_name: str, mode: str | None = None, show: bool = False):
        fig = self._attribute_report(attribute_name, mode=mode).plot_banzhaf(show=show)
        return fig

    def plot_nucleolus(self, attribute_name: str, mode: str | None = None, show: bool = False):
        fig = self._attribute_report(attribute_name, mode=mode).plot_nucleolus(show=show)
        return fig

    def plot_excess(self, attribute_name: str, mode: str | None = None, show: bool = False):
        fig = self._attribute_report(attribute_name, mode=mode).plot_excess_distribution(show=show)
        return fig

    def plot_coalition_size_summary(self, attribute_name: str, mode: str | None = None, metric: str = "best_value", show: bool = False):
        from corex.plots.coalition import plot_coalition_size_summary

        summary = self._attribute_report(attribute_name, mode=mode).metadata.get("coalition_size_summary")
        if isinstance(summary, dict) and "proxy_model" in summary:
            summary = summary["proxy_model"]
        fig = plot_coalition_size_summary(summary, metric=metric)
        if show:
            fig.show()
        return fig

    def to_dict(self) -> dict[str, Any]:
        serialized_attribute_reports = {}
        for name, report in self.attribute_reports.items():
            if isinstance(report, ExplanationResult):
                serialized_attribute_reports[name] = report.to_dict()
            else:
                serialized_attribute_reports[name] = {mode: mode_report.to_dict() for mode, mode_report in report.items()}
        return {
            "evaluation_mode": self.evaluation_mode,
            "audit_scope": self.metadata.get("audit_scope", "local"),
            "summary": self.summary(),
            "attribute_reports": serialized_attribute_reports,
            "top_proxy_coalitions": {
                name: self.top_proxy_coalitions(name)
                for name in self.attribute_reports
            },
            "metadata": dict(self.metadata),
        }
