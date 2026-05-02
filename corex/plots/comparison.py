from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from corex.plots.common import COREX_COLORS, apply_corex_theme
from corex.utils.explanation import ExplanationResult


def _validate_feature_space(results: Sequence[ExplanationResult]) -> list[str]:
    players = results[0].players
    for result in results[1:]:
        if result.players != players:
            raise ValueError("comparison plots require identical feature/player order across results.")
    return players


def comparison_plot(
    results: Sequence[ExplanationResult],
    feature_names: Sequence[str] | None = None,
    show: bool = False,
):
    if not results:
        raise ValueError("results must not be empty")
    default_names = _validate_feature_space(results)
    names = list(feature_names) if feature_names is not None else default_names
    z_values = []
    y_labels = []
    for index, result in enumerate(results):
        y_labels.append(f"Result {index}")
        allocation_map = dict(zip(result.players, result.feature_values))
        z_values.append([allocation_map.get(name, 0.0) for name in names])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=names,
                y=y_labels,
                colorscale="RdBu",
                colorbar={"title": "Allocation"},
                hovertemplate="Result: %{y}<br>Feature: %{x}<br>Allocation: %{z:.6f}<extra></extra>",
            )
        ]
    )
    fig = apply_corex_theme(fig, title="Explanation Comparison", xaxis_title="Feature", yaxis_title="Result")
    if show:
        fig.show()
    return fig


def stability_plot(
    results: Sequence[ExplanationResult],
    feature_names: Sequence[str] | None = None,
    show: bool = False,
):
    if not results:
        raise ValueError("results must not be empty")
    default_names = _validate_feature_space(results)
    names = list(feature_names) if feature_names is not None else default_names
    fig = go.Figure()
    for name in names:
        values = []
        for result in results:
            allocation_map = dict(zip(result.players, result.feature_values))
            values.append(allocation_map.get(name, 0.0))
        fig.add_trace(
            go.Box(
                y=values,
                name=name,
                marker_color=COREX_COLORS["primary"],
                boxmean=True,
                hovertemplate=f"<b>{name}</b><br>Allocation: %{{y:.6f}}<extra></extra>",
            )
        )
    fig = apply_corex_theme(fig, title="Explanation Stability", xaxis_title="Feature", yaxis_title="Allocation Across Runs")
    if show:
        fig.show()
    return fig
