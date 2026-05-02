from __future__ import annotations

import plotly.graph_objects as go

from corex.plots.common import COREX_COLORS, apply_corex_theme
from corex.utils.explanation import ExplanationResult


def plot_stability(result: ExplanationResult) -> go.Figure:
    if "nucleolus" in result.solution_concepts:
        diagnostics = result.solution_concepts["nucleolus"].solver_diagnostics
        value = float(diagnostics.get("max_excess", result.epsilon or 0.0))
        title = "Maximum Excess"
    elif "least_core" in result.solution_concepts:
        diagnostics = result.solution_concepts["least_core"].solver_diagnostics
        value = float(diagnostics.get("epsilon", result.epsilon or 0.0))
        title = "Least-Core Epsilon"
    else:
        diagnostics = result.primary_solution_result.solver_diagnostics
        value = float(diagnostics.get("epsilon", result.epsilon or 0.0) or 0.0)
        title = "Stability Proxy"
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=value,
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, max(value * 1.25, 0.1)]},
                "bar": {"color": COREX_COLORS["secondary"]},
                "steps": [
                    {"range": [0, max(value * 0.5, 0.02)], "color": "#d1fae5"},
                    {"range": [max(value * 0.5, 0.02), max(value * 1.25, 0.1)], "color": "#fef3c7"},
                ],
            },
            title={"text": title},
        )
    )
    fig.update_layout(
        title={"text": "Coalition Stability", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 14, "color": COREX_COLORS["dark"]},
        margin={"l": 48, "r": 48, "t": 72, "b": 32},
        paper_bgcolor="white",
    )
    return fig


def plot_excess_distribution(result: ExplanationResult) -> go.Figure:
    if "nucleolus" in result.solution_concepts:
        diagnostics = result.solution_concepts["nucleolus"].solver_diagnostics
        values = diagnostics.get("excess_values", [diagnostics.get("max_excess", 0.0)])
    elif "least_core" in result.solution_concepts:
        diagnostics = result.solution_concepts["least_core"].solver_diagnostics
        values = [diagnostics.get("epsilon", result.epsilon or 0.0)]
    else:
        diagnostics = result.primary_solution_result.solver_diagnostics
        values = [diagnostics.get("epsilon", result.epsilon or 0.0) or 0.0]
    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=min(max(len(values), 10), 40),
                marker={"color": COREX_COLORS["accent"]},
                hovertemplate="Excess: %{x:.6f}<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    return apply_corex_theme(fig, title="Excess Distribution", xaxis_title="Excess", yaxis_title="Coalition Count")
