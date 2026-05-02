from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from corex.plots.common import COREX_COLORS, apply_corex_theme


def plot_coalition_size_summary(summary: dict[str, Any], metric: str = "average_value", title: str | None = None) -> go.Figure:
    sizes = sorted(summary["sizes"], key=lambda item: int(item))
    x_values = [int(size) for size in sizes]
    if metric == "average_value":
        y_values = [summary["sizes"][size]["average_value"] for size in sizes]
        yaxis_title = "Average Coalition Value"
        series_name = "Average Value"
        color = COREX_COLORS["secondary"]
    elif metric == "best_value":
        y_values = [summary["sizes"][size]["best_coalition"]["value"] for size in sizes]
        yaxis_title = "Best Coalition Value"
        series_name = "Best Value"
        color = COREX_COLORS["accent"]
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    hover_text = [
        "<br>".join(
            [
                f"Size: {size}",
                f"Coalitions: {summary['sizes'][size]['coalition_count']}",
                f"Best coalition: {', '.join(summary['sizes'][size]['best_coalition']['coalition'])}",
                f"Best value: {summary['sizes'][size]['best_coalition']['value']:.6f}",
                f"Average value: {summary['sizes'][size]['average_value']:.6f}",
            ]
        )
        for size in sizes
    ]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                line={"color": color, "width": 4},
                marker={"size": 10},
                name=series_name,
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )
    return apply_corex_theme(
        fig,
        title=title or f"Coalition Size Summary: {series_name}",
        xaxis_title="Coalition Size",
        yaxis_title=yaxis_title,
    )
