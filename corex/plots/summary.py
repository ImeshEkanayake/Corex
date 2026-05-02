from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corex.plots.common import COREX_COLORS, apply_corex_theme, sign_colors
from corex.utils.explanation import ExplanationResult


def _to_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def _to_vector(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _feature_names(count: int, feature_names: Sequence[str] | None) -> list[str]:
    return list(feature_names) if feature_names is not None else [f"feature_{index}" for index in range(count)]


def _top_feature_indices(matrix: list[list[float]], max_display: int) -> list[int]:
    n_features = len(matrix[0]) if matrix else 0
    means = []
    for index in range(n_features):
        column = [abs(row[index]) for row in matrix]
        means.append((sum(column) / max(len(column), 1), index))
    means.sort(reverse=True)
    return [index for _, index in means[:max_display]]


def _symmetric_rank(index: int) -> int:
    if index == 0:
        return 0
    step = (index + 1) // 2
    return step if index % 2 else -step


def _beeswarm_offsets(values: list[float], max_width: float = 0.38, nbins: int = 40) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.0 for _ in values]
    span = maximum - minimum
    bin_width = span / max(nbins, 1)
    bins: dict[int, list[int]] = {}
    for index, value in enumerate(values):
        bucket = int((value - minimum) / bin_width) if bin_width > 0 else 0
        bins.setdefault(bucket, []).append(index)
    offsets = [0.0 for _ in values]
    for bucket_indices in bins.values():
        bucket_indices.sort(key=lambda idx: values[idx])
        levels = max(len(bucket_indices) - 1, 1)
        for local_position, original_index in enumerate(bucket_indices):
            rank = _symmetric_rank(local_position)
            offsets[original_index] = (rank / levels) * max_width
    return offsets


def _normalize_feature_colors(feature_values: list[float]) -> tuple[list[float], dict[str, object]]:
    if not feature_values:
        return [], {"size": 9, "color": COREX_COLORS["primary"], "opacity": 0.85}
    sorted_values = sorted(feature_values)
    count = len(sorted_values)
    if count == 1 or math.isclose(sorted_values[0], sorted_values[-1]):
        normalized = [0.5 for _ in feature_values]
    else:
        normalized = []
        for value in feature_values:
            left = 0
            right = count
            while left < right:
                middle = (left + right) // 2
                if sorted_values[middle] < value:
                    left = middle + 1
                else:
                    right = middle
            normalized.append(left / max(count - 1, 1))
    return normalized, {
        "size": 8,
        "color": normalized,
        "colorscale": [
            [0.0, "#284858"],
            [0.5, "#438D94"],
            [1.0, "#FF8623"],
        ],
        "showscale": True,
        "colorbar": {
            "title": "Feature Value",
            "thickness": 14,
            "len": 0.78,
            "outlinewidth": 0,
        },
        "opacity": 0.9,
    }


def summary_plot(
    attribution_matrix: Sequence[Sequence[float]],
    feature_values: Sequence[Sequence[float]] | None = None,
    feature_names: Sequence[str] | None = None,
    outcome_values: Sequence[float] | None = None,
    plot_type: str = "bar",
    max_display: int = 20,
    show: bool = False,
):
    matrix = _to_matrix(attribution_matrix)
    if not matrix:
        raise ValueError("attribution_matrix must not be empty")
    names = _feature_names(len(matrix[0]), feature_names)
    top_indices = _top_feature_indices(matrix, max_display)
    ordered_names = [names[index] for index in reversed(top_indices)]
    mean_abs = [
        sum(abs(row[index]) for row in matrix) / len(matrix)
        for index in reversed(top_indices)
    ]
    if plot_type == "bar":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=mean_abs,
                    y=ordered_names,
                    orientation="h",
                    marker_color=COREX_COLORS["primary"],
                    hovertemplate="<b>%{y}</b><br>Mean |allocation|: %{x:.6f}<extra></extra>",
                )
            ]
        )
        fig = apply_corex_theme(fig, title="Summary Plot", xaxis_title="Mean |Allocation|", yaxis_title="Feature")
    elif plot_type == "dot":
        fig = go.Figure()
        feature_matrix = _to_matrix(feature_values) if feature_values is not None else None
        for row_offset, index in enumerate(reversed(top_indices)):
            x_values = [row[index] for row in matrix]
            if feature_matrix is not None:
                color_values = [row[index] for row in feature_matrix]
                normalized_colors, marker = _normalize_feature_colors(color_values)
                marker["showscale"] = row_offset == 0
            else:
                normalized_colors = None
                marker = {"size": 8, "color": COREX_COLORS["primary"], "opacity": 0.82}
            y_offsets = _beeswarm_offsets(x_values)
            y_values = [row_offset + offset for offset in y_offsets]
            customdata = list(zip(color_values, normalized_colors or [None for _ in x_values])) if feature_matrix is not None else None
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    marker=marker,
                    name=names[index],
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>{names[index]}</b><br>"
                        "Allocation: %{x:.6f}<br>"
                        "Feature value: %{customdata[0]:.6f}<extra></extra>"
                        if feature_matrix is not None
                        else f"<b>{names[index]}</b><br>Allocation: %{{x:.6f}}<extra></extra>"
                    ),
                )
            )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(ordered_names))),
            ticktext=ordered_names,
        )
        fig.add_vline(x=0.0, line_width=1.2, line_color=COREX_COLORS["neutral"], opacity=0.9)
        fig.update_traces(cliponaxis=False)
        fig = apply_corex_theme(fig, title="Summary Beeswarm Plot", xaxis_title="Allocation", yaxis_title="Feature")
    elif plot_type == "violin":
        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            column_widths=[0.28, 0.72],
        )
        feature_matrix = _to_matrix(feature_values) if feature_values is not None else None
        outcomes = [float(value) for value in outcome_values] if outcome_values is not None else None
        if outcomes is not None and len(outcomes) != len(matrix):
            raise ValueError("outcome_values must have the same number of rows as attribution_matrix")
        row_positions = list(range(len(ordered_names)))
        fig.add_trace(
            go.Bar(
                x=mean_abs,
                y=row_positions,
                orientation="h",
                marker_color="rgba(31, 41, 55, 0.82)",
                hovertemplate="<b>%{customdata}</b><br>Mean |allocation|: %{x:.6f}<extra></extra>",
                customdata=ordered_names,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for row_offset, index in enumerate(reversed(top_indices)):
            x_values = [row[index] for row in matrix]
            row_feature_values = [row[index] for row in feature_matrix] if feature_matrix is not None else [None for _ in x_values]
            if outcomes is not None:
                top_points = [
                    (value, feature_value)
                    for value, feature_value, outcome in zip(x_values, row_feature_values, outcomes)
                    if outcome >= 0.5
                ]
                bottom_points = [
                    (value, feature_value)
                    for value, feature_value, outcome in zip(x_values, row_feature_values, outcomes)
                    if outcome < 0.5
                ]
                top_label = "Final class = 1"
                bottom_label = "Final class = 0"
            else:
                top_points = [(value, feature_value) for value, feature_value in zip(x_values, row_feature_values) if value >= 0.0]
                bottom_points = [(value, feature_value) for value, feature_value in zip(x_values, row_feature_values) if value < 0.0]
                top_label = "Positive allocation"
                bottom_label = "Negative allocation"
            top_values = [value for value, _ in top_points]
            bottom_values = [value for value, _ in bottom_points]
            if top_values:
                fig.add_trace(
                    go.Violin(
                        x=top_values,
                        y=[row_offset for _ in top_values],
                        orientation="h",
                        side="positive",
                        line={"color": "#438D94", "width": 1.0},
                        fillcolor="rgba(67, 141, 148, 0.24)",
                        opacity=0.9,
                        points=False,
                        box_visible=False,
                        meanline_visible=False,
                        spanmode="hard",
                        showlegend=False,
                        width=0.82,
                        hovertemplate=f"<b>{names[index]}</b><br>{top_label}: %{{x:.6f}}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )
            if bottom_values:
                fig.add_trace(
                    go.Violin(
                        x=bottom_values,
                        y=[row_offset for _ in bottom_values],
                        orientation="h",
                        side="negative",
                        line={"color": "#F66C1D", "width": 1.0},
                        fillcolor="rgba(246, 108, 29, 0.24)",
                        opacity=0.9,
                        points=False,
                        box_visible=False,
                        meanline_visible=False,
                        spanmode="hard",
                        showlegend=False,
                        width=0.82,
                        hovertemplate=f"<b>{names[index]}</b><br>{bottom_label}: %{{x:.6f}}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )
            if top_values:
                top_offsets = _beeswarm_offsets([value for value, _ in top_points], max_width=0.18, nbins=28)
                if feature_matrix is not None:
                    _, top_marker = _normalize_feature_colors([feature_value for _, feature_value in top_points])
                    top_marker["showscale"] = row_offset == 0
                    top_marker["size"] = 5.5
                    top_marker["opacity"] = 0.72
                    if row_offset == 0:
                        top_marker["colorbar"]["x"] = 1.02
                        top_marker["colorbar"]["y"] = 0.5
                        top_marker["colorbar"]["len"] = 0.82
                else:
                    top_marker = {"size": 5.5, "color": "#438D94", "opacity": 0.65}
                fig.add_trace(
                    go.Scatter(
                        x=[value for value, _ in top_points],
                        y=[row_offset + abs(offset) + 0.02 for offset in top_offsets],
                        mode="markers",
                        marker=top_marker,
                        customdata=[[feature_value] for _, feature_value in top_points],
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{names[index]}</b><br>"
                            "Influence: %{x:.6f}<br>"
                            "Feature value: %{customdata[0]:.6f}<br>"
                            f"{top_label}<extra></extra>"
                            if feature_matrix is not None
                            else f"<b>{names[index]}</b><br>Influence: %{{x:.6f}}<br>{top_label}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
                )
            if bottom_points:
                bottom_offsets = _beeswarm_offsets([value for value, _ in bottom_points], max_width=0.18, nbins=28)
                if feature_matrix is not None:
                    _, bottom_marker = _normalize_feature_colors([feature_value for _, feature_value in bottom_points])
                    bottom_marker["showscale"] = False
                    bottom_marker["size"] = 5.5
                    bottom_marker["opacity"] = 0.72
                else:
                    bottom_marker = {"size": 5.5, "color": "#F66C1D", "opacity": 0.65}
                fig.add_trace(
                    go.Scatter(
                        x=[value for value, _ in bottom_points],
                        y=[row_offset - abs(offset) - 0.02 for offset in bottom_offsets],
                        mode="markers",
                        marker=bottom_marker,
                        customdata=[[feature_value] for _, feature_value in bottom_points],
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{names[index]}</b><br>"
                            "Influence: %{x:.6f}<br>"
                            "Feature value: %{customdata[0]:.6f}<br>"
                            f"{bottom_label}<extra></extra>"
                            if feature_matrix is not None
                            else f"<b>{names[index]}</b><br>Influence: %{{x:.6f}}<br>{bottom_label}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
                )
        fig.add_vline(x=0.0, line_width=1.2, line_color=COREX_COLORS["neutral"], opacity=0.9, row=1, col=2)
        fig.update_traces(cliponaxis=False, selector={"type": "scatter"})
        fig = apply_corex_theme(fig, title="Summary Violin Plot", xaxis_title="Allocation", yaxis_title="Feature")
        fig.update_xaxes(title_text="Mean |Allocation|", row=1, col=1, autorange="reversed", showgrid=False, zeroline=False)
        fig.update_xaxes(title_text="Allocation", row=1, col=2, showgrid=False, zeroline=False)
        fig.update_yaxes(
            row=1,
            col=1,
            tickmode="array",
            tickvals=row_positions,
            ticktext=ordered_names,
            title_text="Feature",
            gridcolor="#e5e7eb",
        )
        fig.update_yaxes(
            row=1,
            col=2,
            tickmode="array",
            tickvals=row_positions,
            ticktext=ordered_names,
            showticklabels=False,
            title_text=None,
            gridcolor="#e5e7eb",
        )
        fig.update_layout(
            violingap=0,
            violinmode="overlay",
            bargap=0.18,
            margin={"l": 128, "r": 56, "t": 72, "b": 64},
        )
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")
    if show:
        fig.show()
    return fig


def dependence_plot(
    feature: int | str,
    attribution_matrix: Sequence[Sequence[float]],
    feature_values: Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
    interaction_feature: int | str | None = None,
    show: bool = False,
):
    matrix = _to_matrix(attribution_matrix)
    values = _to_matrix(feature_values)
    names = _feature_names(len(matrix[0]), feature_names)
    feature_index = names.index(feature) if isinstance(feature, str) else int(feature)
    if interaction_feature is None:
        interaction_index = feature_index
    else:
        interaction_index = names.index(interaction_feature) if isinstance(interaction_feature, str) else int(interaction_feature)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[row[feature_index] for row in values],
                y=[row[feature_index] for row in matrix],
                mode="markers",
                marker={
                    "size": 10,
                    "color": [row[interaction_index] for row in values],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": names[interaction_index]},
                },
                hovertemplate=f"<b>{names[feature_index]}</b><br>Feature value: %{{x:.6f}}<br>Allocation: %{{y:.6f}}<extra></extra>",
            )
        ]
    )
    fig = apply_corex_theme(
        fig,
        title=f"Dependence Plot: {names[feature_index]}",
        xaxis_title=f"{names[feature_index]} value",
        yaxis_title="Allocation",
    )
    if show:
        fig.show()
    return fig


def scatter_plot(
    attribution_matrix: Sequence[Sequence[float]],
    feature_values: Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
    feature: int | str | None = None,
    show: bool = False,
):
    names = _feature_names(len(attribution_matrix[0]), feature_names) if attribution_matrix else []
    selected = feature if feature is not None else 0
    return dependence_plot(selected, attribution_matrix, feature_values, feature_names=names, show=show)


def heatmap_plot(
    attribution_matrix: Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
    instance_order: Sequence[str] | None = None,
    max_display: int = 20,
    show: bool = False,
):
    matrix = _to_matrix(attribution_matrix)
    names = _feature_names(len(matrix[0]), feature_names)
    top_indices = _top_feature_indices(matrix, max_display)
    z_values = [[row[index] for index in top_indices] for row in matrix]
    y_labels = list(instance_order) if instance_order is not None else [f"Instance {i}" for i in range(len(matrix))]
    x_labels = [names[index] for index in top_indices]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale="RdBu",
                colorbar={"title": "Allocation"},
                hovertemplate="Instance: %{y}<br>Feature: %{x}<br>Allocation: %{z:.6f}<extra></extra>",
            )
        ]
    )
    fig = apply_corex_theme(fig, title="Attribution Heatmap", xaxis_title="Feature", yaxis_title="Instance")
    if show:
        fig.show()
    return fig


def _top_entries(result: ExplanationResult, max_display: int) -> tuple[list[str], list[float]]:
    pairs = sorted(zip(result.players, result.feature_values), key=lambda item: abs(item[1]), reverse=True)[:max_display]
    names = [name for name, _ in pairs]
    values = [value for _, value in pairs]
    return names, values


def _force_display_entries(result: ExplanationResult, max_display: int) -> list[tuple[str, float]]:
    pairs = sorted(
        [(str(name), float(value)) for name, value in zip(result.players, result.feature_values) if not math.isclose(float(value), 0.0)],
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    if not pairs:
        return []
    if len(pairs) <= max_display:
        return pairs
    kept = pairs[: max(1, max_display - 1)]
    remainder = pairs[max(1, max_display - 1) :]
    kept.append((f"Other ({len(remainder)})", float(sum(value for _, value in remainder))))
    return kept


def _force_reference_values(result: ExplanationResult) -> tuple[float, float]:
    base_value = None
    output_value = None
    if result.game is not None:
        try:
            base_value = float(result.game.evaluate_indices(frozenset()))
            output_value = float(result.game.evaluate_indices(result.game.grand_coalition_indices()))
        except Exception:
            base_value = None
            output_value = None
    if output_value is None:
        maybe_output = result.game_definition.get("grand_coalition_value")
        if maybe_output is not None:
            output_value = float(maybe_output)
    contribution_sum = float(np.sum(result.feature_values))
    if output_value is None:
        output_value = contribution_sum
    if base_value is None:
        base_value = output_value - contribution_sum
    return float(base_value), float(output_value)


def _force_band_points(x0: float, x1: float, positive: bool) -> tuple[list[float], list[float]]:
    span = abs(x1 - x0)
    tip = min(max(span * 0.16, 0.02), max(span * 0.4, 0.02))
    if math.isclose(span, 0.0):
        tip = 0.02
    inner = 0.08
    outer = 0.48
    if positive:
        if x1 >= x0:
            xs = [x0, max(x0, x1 - tip), x1, max(x0, x1 - tip), x0]
        else:
            xs = [x0, min(x0, x1 + tip), x1, min(x0, x1 + tip), x0]
        ys = [inner, inner, 0.26, outer, outer]
    else:
        if x1 <= x0:
            xs = [x0, min(x0, x1 + tip), x1, min(x0, x1 + tip), x0]
        else:
            xs = [x0, max(x0, x1 - tip), x1, max(x0, x1 - tip), x0]
        ys = [-inner, -inner, -0.26, -outer, -outer]
    return xs, ys


def force_plot(result: ExplanationResult, max_display: int = 10, show: bool = False):
    entries = _force_display_entries(result, max_display)
    base_value, output_value = _force_reference_values(result)
    positive_color = "#F66C1D"
    negative_color = "#438D94"

    positive_entries = [(name, value) for name, value in entries if value > 0]
    negative_entries = [(name, value) for name, value in entries if value < 0]
    negative_span = float(sum(abs(value) for _, value in negative_entries))
    positive_span = float(sum(value for _, value in positive_entries))

    fig = go.Figure()
    x_extents = [base_value, output_value, base_value - negative_span, base_value + positive_span]
    band_bottom = -0.34
    band_top = 0.34
    left_cursor = base_value - negative_span
    right_cursor = base_value

    for index, (name, value) in enumerate(negative_entries):
        x0 = left_cursor
        x1 = left_cursor + abs(value)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[band_bottom, band_bottom, band_top, band_top, band_bottom],
                mode="lines",
                fill="toself",
                line={"color": negative_color, "width": 1.4},
                fillcolor="rgba(67, 141, 148, 0.86)",
                hovertemplate=f"<b>{name}</b><br>Contribution: {value:+.6f}<extra></extra>",
                showlegend=False,
            )
        )
        if index > 0:
            fig.add_vline(x=x0, line_width=2.0, line_color="rgba(255,255,255,0.92)")
        segment_width = abs(x1 - x0)
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=0.0,
            text=f"{name}<br>Impact: {value:+.4f}",
            showarrow=False,
            textangle=-90,
            font={"size": 8 if segment_width < 0.03 else 10 if segment_width < 0.06 else 12, "color": "white"},
            xanchor="center",
            yanchor="middle",
        )
        left_cursor = x1

    for index, (name, value) in enumerate(positive_entries):
        x0 = right_cursor
        x1 = right_cursor + value
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[band_bottom, band_bottom, band_top, band_top, band_bottom],
                mode="lines",
                fill="toself",
                line={"color": positive_color, "width": 1.4},
                fillcolor="rgba(246, 108, 29, 0.86)",
                hovertemplate=f"<b>{name}</b><br>Contribution: {value:+.6f}<extra></extra>",
                showlegend=False,
            )
        )
        if index > 0:
            fig.add_vline(x=x0, line_width=2.0, line_color="rgba(255,255,255,0.92)")
        segment_width = abs(x1 - x0)
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=0.0,
            text=f"{name}<br>Impact: {value:+.4f}",
            showarrow=False,
            textangle=-90,
            font={"size": 8 if segment_width < 0.03 else 10 if segment_width < 0.06 else 12, "color": "white"},
            xanchor="center",
            yanchor="middle",
        )
        right_cursor = x1

    x_min = min(x_extents)
    x_max = max(x_extents)
    padding = max((x_max - x_min) * 0.08, 0.25)

    fig.add_vline(x=base_value, line_width=2.2, line_color=COREX_COLORS["neutral"], line_dash="dash", opacity=0.95)
    fig.add_annotation(
        x=base_value,
        y=0.46,
        text=f"<b>Base:</b> {base_value:.3f}",
        showarrow=False,
        font={"size": 13, "color": COREX_COLORS["dark"]},
        xanchor="center",
        yanchor="bottom",
    )
    fig.add_annotation(
        x=output_value,
        y=-0.46,
        text=f"<b>Output:</b> {output_value:.3f}",
        showarrow=False,
        font={"size": 12, "color": COREX_COLORS["dark"]},
        xanchor="center",
        yanchor="top",
    )

    if not entries:
        fig.add_annotation(
            x=base_value,
            y=0.0,
            text="No non-zero contributions",
            showarrow=False,
            font={"size": 13, "color": COREX_COLORS["neutral"]},
        )

    fig = apply_corex_theme(fig, title="COREX Force Plot", xaxis_title="Model Output Value", yaxis_title=None)
    fig.update_layout(
        title={"text": "COREX Force Plot", "x": 0.68, "xanchor": "center"},
        height=max(520, 260 + 28 * max(len(entries), 1)),
        margin={"l": 72, "r": 32, "t": 96, "b": 96},
    )
    fig.update_xaxes(range=[x_min - padding, x_max + padding], showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(
        range=[-0.62, 0.62],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text=None,
    )
    if show:
        fig.show()
    return fig


def waterfall_plot(
    result: ExplanationResult,
    max_display: int = 10,
    orientation: str = "vertical",
    show: bool = False,
):
    feature_names, contributions = _top_entries(result, max_display)
    measure = ["relative"] * len(feature_names) + ["total"]
    labels = feature_names + ["Output"]
    values = contributions + [sum(contributions)]
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    trace_kwargs = {
        "measure": measure,
        "increasing": {"marker": {"color": COREX_COLORS["primary"]}},
        "decreasing": {"marker": {"color": COREX_COLORS["negative"]}},
        "totals": {"marker": {"color": COREX_COLORS["dark"]}},
        "connector": {"line": {"color": COREX_COLORS["neutral"]}},
    }
    if orientation == "horizontal":
        trace_kwargs.update({"orientation": "h", "y": labels, "x": values})
        fig = go.Figure(go.Waterfall(**trace_kwargs))
        fig = apply_corex_theme(fig, title="Waterfall Plot", xaxis_title="Contribution", yaxis_title="Feature")
    else:
        trace_kwargs.update({"x": labels, "y": values})
        fig = go.Figure(go.Waterfall(**trace_kwargs))
        fig = apply_corex_theme(fig, title="Waterfall Plot", xaxis_title="Feature", yaxis_title="Contribution")
    if show:
        fig.show()
    return fig


def waterfall_plot_vertical(result: ExplanationResult, max_display: int = 10, show: bool = False):
    return waterfall_plot(result, max_display=max_display, orientation="vertical", show=show)


def waterfall_plot_horizontal(result: ExplanationResult, max_display: int = 10, show: bool = False):
    return waterfall_plot(result, max_display=max_display, orientation="horizontal", show=show)
