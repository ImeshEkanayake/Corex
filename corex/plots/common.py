from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go


COREX_COLORS = {
    "primary": "#1f5aa6",
    "secondary": "#2a7f62",
    "accent": "#7f3c8d",
    "negative": "#c03d3e",
    "neutral": "#6b7280",
    "light": "#e5e7eb",
    "dark": "#1f2937",
}


def apply_corex_theme(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 14, "color": COREX_COLORS["dark"]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 72, "r": 32, "t": 72, "b": 64},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0, "title": {"text": legend_title}},
    )
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title, showgrid=False, zeroline=False)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title, gridcolor="#e5e7eb", zerolinecolor="#d1d5db")
    return fig


def sign_colors(values: Iterable[float]) -> list[str]:
    return [COREX_COLORS["primary"] if value >= 0 else COREX_COLORS["negative"] for value in values]
