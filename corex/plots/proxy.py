from __future__ import annotations

from collections import Counter

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from corex.plots.common import COREX_COLORS
from corex.utils.explanation import ProxyAuditReport


def _resolve_coalition_summary(result) -> dict | None:
    if isinstance(result, dict):
        nested = result.get("proxy_model") or result.get("ablation")
        return _resolve_coalition_summary(nested) if nested is not None else None
    summary = result.metadata.get("coalition_size_summary")
    if summary is None:
        return None
    if isinstance(summary, dict) and "sizes" in summary:
        return summary
    if isinstance(summary, dict):
        return summary.get("proxy_model") or summary.get("ablation")
    return None


def _resolve_player_names(report: ProxyAuditReport) -> list[str]:
    for result in report.attribute_reports.values():
        if isinstance(result, dict):
            result = result.get("proxy_model") or result.get("ablation")
            if result is None:
                continue
        players = getattr(result, "players", None) or result.game_definition.get("players", [])
        if players:
            return list(players)
    return []


def plot_proxy_overlaps(
    report: ProxyAuditReport,
    top_k: int | None = None,
    dataset_name: str | None = None,
    target_description: str | None = None,
) -> go.Figure:
    attribute_names = list(report.attribute_reports)
    player_names = _resolve_player_names(report)
    coalition_candidates = []
    feature_candidate_counts = Counter()
    dataset_name = dataset_name or report.metadata.get("dataset_name")
    target_description = target_description or report.metadata.get("target_description")
    resolved_top_k = int(top_k if top_k is not None else report.metadata.get("top_k", report.metadata.get("top_n", 10)))

    for attribute_name, result in report.attribute_reports.items():
        summary = _resolve_coalition_summary(result)
        if not summary:
            continue
        for size_key in sorted(summary.get("sizes", {}), key=lambda key: int(key)):
            size_block = summary["sizes"][size_key]
            for rank, item in enumerate(size_block.get("top_coalitions", [])[:resolved_top_k], start=1):
                coalition_candidates.append(
                    {
                        "attribute": attribute_name,
                        "size": int(size_key),
                        "rank": rank,
                        "coalition": list(item.get("coalition", [])),
                        "value": float(item.get("value", 0.0)),
                    }
                )
                feature_candidate_counts.update(item.get("coalition", []))

    coalition_candidates.sort(
        key=lambda item: (item["size"], item["rank"], -item["value"], item["attribute"], item["coalition"])
    )

    if not player_names:
        player_names = sorted({member for item in coalition_candidates for member in item["coalition"]})

    player_names = sorted(
        player_names,
        key=lambda name: (-feature_candidate_counts.get(name, 0), name),
    )

    feature_count = max(len(player_names), 1)
    candidate_count = max(len(coalition_candidates), 1)
    max_label_length = max((len(name) for name in player_names), default=12)
    figure_height = max(620, 240 + feature_count * 68 + candidate_count * 10)
    figure_width = max(1100, 280 + candidate_count * 42)
    left_margin = min(max(120, max_label_length * 11), 260)
    show_bar_text = candidate_count <= 18
    marker_size = 11 if candidate_count <= 24 else 9
    player_positions = {name: index for index, name in enumerate(player_names)}

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.52, 0.48],
        column_widths=[0.25, 0.75],
        vertical_spacing=0.1,
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    )

    fig.add_trace(
        go.Bar(
            x=[feature_candidate_counts.get(name, 0) for name in player_names],
            y=player_names,
            orientation="h",
            marker_color=COREX_COLORS["neutral"],
            hovertemplate="<b>%{y}</b><br>Displayed in %{x} coalition columns<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    x_positions = list(range(len(coalition_candidates)))
    x_range = [-0.5, len(x_positions) - 0.5] if x_positions else [-0.5, 0.5]
    x_labels = [f"{item['attribute']}<br>k={item['size']} #{item['rank']}" for item in coalition_candidates]
    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=[item["value"] for item in coalition_candidates],
            marker_color=COREX_COLORS["dark"],
            text=[f"{item['value']:.3f}" for item in coalition_candidates] if show_bar_text else None,
            textposition="outside" if show_bar_text else "none",
            textfont={"size": 10},
            hovertext=[
                "<br>".join(
                    [
                        f"Proxy: {item['attribute']}",
                        f"Coalition size: {item['size']}",
                        f"Rank within size: {item['rank']}",
                        f"Coalition: {', '.join(item['coalition'])}",
                        f"Value: {item['value']:.6f}",
                    ]
                )
                for item in coalition_candidates
            ],
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    row_positions = list(range(len(player_names)))
    for row_index, player_name in enumerate(player_names):
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=[row_index] * len(x_positions),
                mode="markers",
                marker={"size": marker_size, "color": COREX_COLORS["light"]},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    for index, item in enumerate(coalition_candidates):
        coalition_rows = sorted(player_positions[name] for name in item["coalition"] if name in player_positions)
        if coalition_rows:
            if len(coalition_rows) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=[index, index],
                        y=[coalition_rows[0], coalition_rows[-1]],
                        mode="lines",
                        line={"width": 2, "color": COREX_COLORS["dark"]},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )
            fig.add_trace(
                go.Scatter(
                    x=[index] * len(coalition_rows),
                    y=coalition_rows,
                    mode="markers",
                    marker={"size": marker_size + 1, "color": COREX_COLORS["dark"]},
                    hovertext=[
                        "<br>".join(
                            [
                                f"Proxy: {item['attribute']}",
                                f"Coalition size: {item['size']}",
                                f"Rank within size: {item['rank']}",
                                f"Feature: {player_names[row]}",
                                f"Coalition members: {', '.join(item['coalition'])}",
                                f"Value: {item['value']:.6f}",
                            ]
                        )
                        for row in coalition_rows
                    ],
                    hovertemplate="%{hovertext}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    size_boundaries = []
    previous_size = None
    for index, item in enumerate(coalition_candidates):
        if previous_size is not None and item["size"] != previous_size:
            size_boundaries.append(index - 0.5)
        previous_size = item["size"]
    for boundary in size_boundaries:
        fig.add_vline(x=boundary, line_width=1, line_dash="dash", line_color="#cbd5e1", row=1, col=2)
        fig.add_vline(x=boundary, line_width=1, line_dash="dash", line_color="#cbd5e1", row=2, col=2)

    fig.update_layout(
        title={"text": "Proxy Coalition Overlaps Across Sizes", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 14, "color": COREX_COLORS["dark"]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=figure_width,
        height=figure_height,
        margin={"l": left_margin, "r": 28, "t": 78, "b": 96},
        bargap=0.18,
        dragmode=False,
        showlegend=False,
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(
        title_text="Coalition Columns Grouped by Size",
        showticklabels=False,
        showgrid=False,
        range=x_range,
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Coalition Value", gridcolor="#e5e7eb", row=1, col=2)
    fig.update_xaxes(title_text="Coalition Membership Count", gridcolor="#e5e7eb", row=2, col=1)
    fig.update_yaxes(title_text="Feature", autorange="reversed", row=2, col=1)
    fig.update_xaxes(
        title_text="Proxy Target / Coalition Size",
        tickmode="array",
        tickvals=x_positions,
        ticktext=x_labels,
        tickangle=90,
        automargin=True,
        range=x_range,
        matches="x2",
        row=2,
        col=2,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=row_positions,
        ticktext=player_names,
        autorange="reversed",
        automargin=True,
        row=2,
        col=2,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_yaxes(automargin=True, row=2, col=1)
    context_lines = []
    if dataset_name:
        context_lines.append(f"<b>Dataset</b><br>{dataset_name}")
    if target_description:
        context_lines.append(f"<b>Proxy Targets</b><br>{target_description}")
    if context_lines:
        top_left_x_domain = list(fig.layout.xaxis.domain) if fig.layout.xaxis.domain else [0.0, 0.25]
        top_left_y_domain = list(fig.layout.yaxis.domain) if fig.layout.yaxis.domain else [0.52, 1.0]
        annotation_x = top_left_x_domain[0] + 0.02 * (top_left_x_domain[1] - top_left_x_domain[0])
        annotation_y = top_left_y_domain[1] - 0.08 * (top_left_y_domain[1] - top_left_y_domain[0])
        fig.add_annotation(
            x=annotation_x,
            y=annotation_y,
            xref="paper",
            yref="paper",
            text="<br><br>".join(context_lines),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font={"size": 13, "color": COREX_COLORS["dark"]},
            bgcolor="rgba(255,255,255,0.92)",
            borderpad=4,
        )
    return fig
