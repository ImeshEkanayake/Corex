#!/usr/bin/env python3
"""Generate the COMPAS COREX paper tables.

The experiment is intentionally self-contained so the table values can be
recomputed from the repository without notebook state.  It uses the local
COREX package for Banzhaf, least-core, and nucleolus explanations, plus
lightweight LIME-style and KernelSHAP-style local linear baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import corex  # noqa: E402


COMPAS_CSV = ROOT / "examples" / "data" / "compas-scores-two-years.csv"
DEFAULT_OUT = ROOT / "Paper Experiemnts" / "compas_corex_results"

CONTINUOUS_COLUMNS = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "days_b_screening_arrest",
    "decile_score",
]
CATEGORICAL_COLUMNS = ["charge_degree_felony", "score_medium", "score_high"]
SENSITIVE_COLUMNS = ["sex_male", "race_caucasian"]
TARGET_COLUMN = "two_year_recid"


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    s_train: pd.DataFrame
    s_test: pd.DataFrame


def load_compas() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw = pd.read_csv(COMPAS_CSV)
    filtered = raw.loc[
        raw["days_b_screening_arrest"].between(-30, 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
    ].copy()

    filtered["charge_degree_felony"] = (filtered["c_charge_degree"] == "F").astype(float)
    filtered["score_medium"] = (filtered["score_text"] == "Medium").astype(float)
    filtered["score_high"] = (filtered["score_text"] == "High").astype(float)
    filtered["sex_male"] = (filtered["sex"] == "Male").astype(int)
    filtered["race_caucasian"] = (filtered["race"] == "Caucasian").astype(int)
    filtered[TARGET_COLUMN] = filtered[TARGET_COLUMN].astype(int)

    x = filtered[CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS].astype(float).reset_index(drop=True)
    y = filtered[TARGET_COLUMN].reset_index(drop=True)
    sensitive = filtered[SENSITIVE_COLUMNS].reset_index(drop=True)
    return x, y, sensitive


def make_split(random_state: int, test_size: float) -> SplitData:
    x, y, sensitive = load_compas()
    train_idx, test_idx = train_test_split(
        np.arange(len(x)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(x.iloc[train_idx]),
        columns=x.columns,
    )
    x_test = pd.DataFrame(
        scaler.transform(x.iloc[test_idx]),
        columns=x.columns,
    )
    return SplitData(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        s_train=sensitive.iloc[train_idx].reset_index(drop=True),
        s_test=sensitive.iloc[test_idx].reset_index(drop=True),
    )


def fit_task_model(x: pd.DataFrame, y: pd.Series, random_state: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def predict_score(model: RandomForestClassifier, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def auc_or_nan(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fairness_gaps(y_true: pd.Series, sensitive: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    sensitive_values = np.asarray(sensitive, dtype=int)
    predictions = (np.asarray(scores, dtype=float) >= 0.5).astype(int)
    y_values = np.asarray(y_true, dtype=int)

    groups = [0, 1]
    positive_rates = []
    true_positive_rates = []
    for group in groups:
        group_mask = sensitive_values == group
        positive_rates.append(float(predictions[group_mask].mean()) if group_mask.any() else float("nan"))
        positive_y_mask = group_mask & (y_values == 1)
        true_positive_rates.append(
            float(predictions[positive_y_mask].mean()) if positive_y_mask.any() else float("nan")
        )
    return (
        float(abs(positive_rates[1] - positive_rates[0])),
        float(abs(true_positive_rates[1] - true_positive_rates[0])),
    )


def aggregate_fairness(
    y_true: pd.Series,
    sensitive_df: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[float, float]:
    gaps = [fairness_gaps(y_true, sensitive_df[column], scores) for column in sensitive_df.columns]
    return float(max(gap[0] for gap in gaps)), float(max(gap[1] for gap in gaps))


def fit_attack_auc(
    x_train: np.ndarray,
    x_test: np.ndarray,
    s_train: pd.DataFrame,
    s_test: pd.DataFrame,
    random_state: int,
) -> float:
    aucs = []
    for column in s_train.columns:
        attack = LogisticRegression(max_iter=1000, random_state=random_state)
        attack.fit(x_train, s_train[column])
        aucs.append(auc_or_nan(s_test[column], attack.predict_proba(x_test)[:, 1]))
    return float(max(aucs))


def select_core_features(model: RandomForestClassifier, columns: list[str], limit: int) -> list[str]:
    importance = pd.Series(model.feature_importances_, index=columns)
    return importance.sort_values(ascending=False).head(limit).index.tolist()


def row_predictor(
    model: RandomForestClassifier,
    row_full: np.ndarray,
    selected_indices: np.ndarray,
    columns: list[str],
) -> Callable[[list[float]], float]:
    def predict(selected_row: list[float]) -> float:
        full = row_full.copy()
        full[selected_indices] = np.asarray(selected_row, dtype=float)
        frame = pd.DataFrame([full], columns=columns)
        return float(predict_score(model, frame)[0])

    return predict


def local_masks(rng: np.random.Generator, n_features: int, n_samples: int) -> np.ndarray:
    masks = rng.binomial(1, 0.5, size=(n_samples, n_features)).astype(float)
    masks[0, :] = 0.0
    if n_samples > 1:
        masks[1, :] = 1.0
    return masks


def local_fidelity_ratio(
    predictor: Callable[[list[float]], float],
    row: np.ndarray,
    baseline: np.ndarray,
    attribution: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 96,
) -> float:
    masks = local_masks(rng, len(row), n_samples)
    empty_value = predictor(baseline.tolist())
    actual = []
    approx = []
    for mask in masks:
        masked = baseline + mask * (row - baseline)
        actual.append(predictor(masked.tolist()))
        approx.append(empty_value + float(mask @ attribution))
    actual_arr = np.asarray(actual, dtype=float)
    approx_arr = np.asarray(approx, dtype=float)
    denom = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.mean((actual_arr - approx_arr) ** 2) / denom)


def lime_style_attribution(
    predictor: Callable[[list[float]], float],
    row: np.ndarray,
    train_selected: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    scale = np.std(train_selected, axis=0)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    perturb = row + rng.normal(0.0, scale, size=(n_samples, len(row)))
    perturb[0, :] = row
    distances = np.linalg.norm((perturb - row) / scale, axis=1)
    kernel_width = np.sqrt(len(row)) * 0.75
    weights = np.exp(-(distances**2) / max(kernel_width**2, 1e-8))
    y_local = np.asarray([predictor(sample.tolist()) for sample in perturb], dtype=float)
    centered = perturb - row
    model = Ridge(alpha=1e-3, fit_intercept=True)
    model.fit(centered, y_local, sample_weight=weights)
    return np.asarray(model.coef_, dtype=float) * row


def shap_style_attribution(
    predictor: Callable[[list[float]], float],
    row: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    n_features = len(row)
    masks = local_masks(rng, n_features, n_samples)
    y_local = []
    weights = []
    for mask in masks:
        sample = baseline + mask * (row - baseline)
        y_local.append(predictor(sample.tolist()))
        size = int(mask.sum())
        if size in {0, n_features}:
            weights.append(1_000.0)
        else:
            weights.append((n_features - 1) / (math.comb(n_features, size) * size * (n_features - size)))
    ridge = Ridge(alpha=1e-6, fit_intercept=True)
    ridge.fit(masks, np.asarray(y_local, dtype=float), sample_weight=np.asarray(weights, dtype=float))
    return np.asarray(ridge.coef_, dtype=float)


def corex_attributions(
    model: RandomForestClassifier,
    split: SplitData,
    selected_features: list[str],
    rows: pd.DataFrame,
    random_state: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    all_columns = list(split.x_train.columns)
    selected_indices = np.asarray([all_columns.index(name) for name in selected_features], dtype=int)
    baseline = split.x_train[selected_features].mean(axis=0).to_numpy(dtype=float)
    scorer = corex.GameTheoreticScorer(
        concepts=["banzhaf", "least_core", "nucleolus"],
        sampling_budget=None,
    )
    values = {
        "COREX-Banzhaf": [],
        "COREX-Least-Core": [],
        "COREX-Nucleolus": [],
    }
    epsilons = {
        "COREX-Banzhaf": [],
        "COREX-Least-Core": [],
        "COREX-Nucleolus": [],
    }
    lfrs = {
        "COREX-Banzhaf": [],
        "COREX-Least-Core": [],
        "COREX-Nucleolus": [],
    }
    rng = np.random.default_rng(random_state + 7000)

    for _, row_series in rows.iterrows():
        row_number = len(values["COREX-Banzhaf"]) + 1
        print(f"COREX explanations {row_number}/{len(rows)}", flush=True)
        row_full = row_series.to_numpy(dtype=float)
        row_selected = row_series[selected_features].to_numpy(dtype=float)
        predictor = row_predictor(model, row_full, selected_indices, all_columns)
        explainer = corex.KernelExplainer(
            predictor,
            background=baseline,
            feature_names=selected_features,
            scorer=scorer,
        )
        result = explainer.explain(row_selected.tolist(), feature_names=selected_features)
        solution_map = {
            "COREX-Banzhaf": result.solution_concepts["banzhaf"],
            "COREX-Least-Core": result.solution_concepts["least_core"],
            "COREX-Nucleolus": result.solution_concepts["nucleolus"],
        }
        shared_epsilon = result.solution_concepts["least_core"].epsilon
        for method, solution in solution_map.items():
            attribution = np.asarray(solution.allocations, dtype=float)
            values[method].append(attribution)
            epsilon = shared_epsilon if method == "COREX-Banzhaf" else solution.epsilon
            epsilons[method].append(float(abs(epsilon)) if epsilon is not None else float("nan"))
            lfrs[method].append(local_fidelity_ratio(predictor, row_selected, baseline, attribution, rng))

    return {
        method: {
            "values": np.vstack(method_values),
            "epsilon": float(np.nanmean(epsilons[method])),
            "lfr": float(np.nanmean(lfrs[method])),
        }
        for method, method_values in values.items()
    }


def baseline_attributions(
    method: str,
    model: RandomForestClassifier,
    split: SplitData,
    selected_features: list[str],
    rows: pd.DataFrame,
    random_state: int,
    n_samples: int,
) -> np.ndarray:
    all_columns = list(split.x_train.columns)
    selected_indices = np.asarray([all_columns.index(name) for name in selected_features], dtype=int)
    baseline = split.x_train[selected_features].mean(axis=0).to_numpy(dtype=float)
    train_selected = split.x_train[selected_features].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state + (101 if method == "LIME" else 202))
    attributions = []
    for _, row_series in rows.iterrows():
        row_full = row_series.to_numpy(dtype=float)
        row_selected = row_series[selected_features].to_numpy(dtype=float)
        predictor = row_predictor(model, row_full, selected_indices, all_columns)
        if method == "LIME":
            attribution = lime_style_attribution(predictor, row_selected, train_selected, rng, n_samples)
        elif method == "SHAP":
            attribution = shap_style_attribution(predictor, row_selected, baseline, rng, n_samples)
        else:
            raise ValueError(f"Unsupported baseline method: {method}")
        attributions.append(attribution)
    return np.vstack(attributions)


def proxy_risk_features(
    explanation_values: np.ndarray,
    sensitive: pd.DataFrame,
    selected_features: list[str],
    top_k: int,
) -> list[str]:
    scores = []
    for index, feature in enumerate(selected_features):
        column_values = explanation_values[:, index]
        feature_score = 0.0
        for sensitive_column in sensitive.columns:
            target = sensitive[sensitive_column].to_numpy(dtype=float)
            if np.std(column_values) <= 1e-12 or np.std(target) <= 1e-12:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(column_values, target)[0, 1]))
            feature_score = max(feature_score, corr)
        scores.append((feature_score, feature))
    scores.sort(reverse=True)
    return [feature for _, feature in scores[:top_k]]


def perturb_features(
    x: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    strength: float,
    adaptive_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    transformed = x.copy()
    if not features:
        return transformed
    std = x[features].std(axis=0).replace(0.0, 1.0)
    for feature in features:
        weight = 1.0 if adaptive_weights is None else adaptive_weights.get(feature, 1.0)
        transformed[feature] = transformed[feature] + rng.normal(
            0.0,
            strength * float(weight) * float(std[feature]),
            size=len(transformed),
        )
    return transformed


def run_privacy_method(
    name: str,
    split: SplitData,
    random_state: int,
    proxy_features: list[str],
    adaptive_weights: dict[str, float],
    strength: float,
) -> dict[str, float]:
    stable_name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    rng = np.random.default_rng(random_state + stable_name_seed)
    if name == "No Privacy":
        x_train, x_test = split.x_train, split.x_test
    elif name == "Uniform Perturbation":
        features = list(split.x_train.columns)
        x_train = perturb_features(split.x_train, features, rng, strength=strength)
        x_test = perturb_features(split.x_test, features, rng, strength=strength)
    elif name == "COREX-Targeted":
        x_train = perturb_features(split.x_train, proxy_features, rng, strength=strength)
        x_test = perturb_features(split.x_test, proxy_features, rng, strength=strength)
    elif name == "COREX-Adaptive":
        resolved_weights = {feature: 0.75 + adaptive_weights.get(feature, 1.0) for feature in proxy_features}
        x_train = perturb_features(split.x_train, proxy_features, rng, strength=strength * 1.15, adaptive_weights=resolved_weights)
        x_test = perturb_features(split.x_test, proxy_features, rng, strength=strength * 1.15, adaptive_weights=resolved_weights)
    else:
        raise ValueError(f"Unknown privacy method: {name}")

    task_model = fit_task_model(x_train, split.y_train, random_state)
    task_scores = predict_score(task_model, x_test)
    attack_auc = fit_attack_auc(
        x_train.to_numpy(dtype=float),
        x_test.to_numpy(dtype=float),
        split.s_train,
        split.s_test,
        random_state,
    )
    dp_gap, eo_gap = aggregate_fairness(split.y_test, split.s_test, task_scores)
    return {
        "Task AUC": auc_or_nan(split.y_test, task_scores),
        "Attack AUC": attack_auc,
        "DP Gap": dp_gap,
        "EO Gap": eo_gap,
    }


def format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return "N/A"
    return f"{value:.3f}"


def latex_rows(df: pd.DataFrame, columns: list[str]) -> str:
    lines = []
    for _, row in df.iterrows():
        values = [str(row["Method"])] + [format_float(row[column]) for column in columns]
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMPAS COREX table experiment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-explain", type=int, default=24)
    parser.add_argument("--corex-features", type=int, default=7)
    parser.add_argument("--baseline-samples", type=int, default=96)
    parser.add_argument("--mitigate-top-k", type=int, default=3)
    parser.add_argument("--perturb-strength", type=float, default=0.55)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = make_split(args.random_state, args.test_size)
    base_model = fit_task_model(split.x_train, split.y_train, args.random_state)
    selected_features = select_core_features(base_model, list(split.x_train.columns), args.corex_features)
    explanation_rows = split.x_test.head(args.n_explain).reset_index(drop=True)
    explanation_sensitive = split.s_test.head(args.n_explain).reset_index(drop=True)

    explanation_payload: dict[str, dict[str, np.ndarray | float | str]] = {}
    for method in ["LIME", "SHAP"]:
        explanation_payload[method] = {
            "values": baseline_attributions(
                method,
                base_model,
                split,
                selected_features,
                explanation_rows,
                args.random_state,
                args.baseline_samples,
            ),
            "epsilon": "N/A",
            "lfr": "N/A",
        }
    explanation_payload.update(
        corex_attributions(base_model, split, selected_features, explanation_rows, args.random_state)
    )

    explanation_table_rows = []
    proxy_feature_sets = {}
    for method, payload in explanation_payload.items():
        values = np.asarray(payload["values"], dtype=float)
        proxy_auc = fit_attack_auc(
            values,
            values,
            explanation_sensitive,
            explanation_sensitive,
            args.random_state,
        )
        risky_features = proxy_risk_features(values, explanation_sensitive, selected_features, args.mitigate_top_k)
        proxy_feature_sets[method] = risky_features
        mitigated_train = split.x_train.drop(columns=risky_features)
        mitigated_test = split.x_test.drop(columns=risky_features)
        mitigated_model = fit_task_model(mitigated_train, split.y_train, args.random_state + 11)
        mitigated_scores = predict_score(mitigated_model, mitigated_test)
        dp_gap, eo_gap = aggregate_fairness(split.y_test, split.s_test, mitigated_scores)
        explanation_table_rows.append(
            {
                "Method": method,
                "Proxy AUC": proxy_auc,
                "DP Gap": dp_gap,
                "EO Gap": eo_gap,
                "epsilon_star": payload["epsilon"],
                "mean_LFR": payload["lfr"],
            }
        )

    explanation_table = pd.DataFrame(explanation_table_rows)
    explanation_table = explanation_table.set_index("Method").loc[
        ["LIME", "SHAP", "COREX-Banzhaf", "COREX-Least-Core", "COREX-Nucleolus"]
    ].reset_index()

    corex_values = np.asarray(explanation_payload["COREX-Banzhaf"]["values"], dtype=float)
    targeted_features = proxy_feature_sets["COREX-Banzhaf"]
    risk = {}
    raw_risk = np.mean(np.abs(corex_values), axis=0)
    for feature, value in zip(selected_features, raw_risk):
        risk[feature] = float(value / max(float(raw_risk.max()), 1e-12))

    privacy_rows = [
        {"Method": method, **run_privacy_method(
            method,
            split,
            args.random_state + 100,
            targeted_features,
            risk,
            args.perturb_strength,
        )}
        for method in ["No Privacy", "Uniform Perturbation", "COREX-Targeted", "COREX-Adaptive"]
    ]
    privacy_table = pd.DataFrame(privacy_rows)

    explanation_csv = args.output_dir / "proxy_audit_results.csv"
    privacy_csv = args.output_dir / "privacy_results.csv"
    explanation_table.to_csv(explanation_csv, index=False)
    privacy_table.to_csv(privacy_csv, index=False)

    metadata = {
        "source_csv": str(COMPAS_CSV),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "n_explain": args.n_explain,
        "selected_features": selected_features,
        "proxy_feature_sets": proxy_feature_sets,
        "metric_notes": {
            "Proxy AUC": "Max AUC over sex_male and race_caucasian attacks trained on explanation vectors.",
            "DP Gap": "Max demographic-parity positive-rate gap over sensitive columns.",
            "EO Gap": "Max equal-opportunity TPR gap over sensitive columns.",
            "epsilon_star": "Mean absolute COREX solver epsilon for the named solution concept.",
            "mean_LFR": "Mean local additive fidelity error ratio; lower means the allocation reconstructs local coalition values with less residual error.",
            "Attack AUC": "Max AUC over sex_male and race_caucasian attacks trained on released features.",
        },
    }
    (args.output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latex = "\n".join(
        [
            "% proxy_audit_results rows",
            latex_rows(explanation_table, ["Proxy AUC", "DP Gap", "EO Gap", "epsilon_star", "mean_LFR"]),
            "",
            "% privacy_results rows",
            latex_rows(privacy_table, ["Task AUC", "Attack AUC", "DP Gap", "EO Gap"]),
        ]
    )
    (args.output_dir / "latex_rows.tex").write_text(latex + "\n", encoding="utf-8")

    print(f"Selected features: {', '.join(selected_features)}")
    print(f"Wrote {explanation_csv}")
    print(f"Wrote {privacy_csv}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
