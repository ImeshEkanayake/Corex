from __future__ import annotations

import plotly.graph_objects as go

from corex.plots.common import apply_corex_theme, sign_colors
from corex.utils.explanation import ExplanationResult


def plot_nucleolus(result: ExplanationResult) -> go.Figure:
    solution = result.solution_concepts["nucleolus"]
    players = result.players
    allocations = solution.allocations
    fig = go.Figure(
        data=[
            go.Bar(
                x=players,
                y=allocations,
                marker_color=sign_colors(allocations),
                text=[f"{value:.4f}" for value in allocations],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Nucleolus allocation: %{y:.6f}<extra></extra>",
            )
        ]
    )
    fig.update_traces(cliponaxis=False)
    return apply_corex_theme(fig, title="Nucleolus Allocation", xaxis_title="Players", yaxis_title="Allocation")
