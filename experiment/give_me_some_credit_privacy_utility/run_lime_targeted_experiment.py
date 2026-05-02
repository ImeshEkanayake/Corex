#!/usr/bin/env python3
"""Standalone Give Me Some Credit lime_targeted privacy--utility experiment.

This source contains the full Table 2 implementation and writes outputs under lime_targeted/outputs by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
DATA_URL = "https://raw.githubusercontent.com/aashish-bidap/Give-Me-Some-Credit/master/cs-training.csv"
TARGET = "SeriousDlqin2yrs"
AGE = "age"
EXPECTED_COLUMNS = [
    TARGET,
    AGE,
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
METHOD_ORDER = [
    "No Privacy",
    "Uniform Perturbation",
    "LIME-Targeted",
    "SHAP-Targeted",
    "COREX-Targeted",
    "COREX-Adaptive",
]


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 2 privacy--utility experiment.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "lime_targeted" / "outputs")
    parser.add_argument("--ranking-source-dir", type=Path, default=REPO_ROOT / "Paper Experiemnts" / "give_me_some_credit_corex" / "outputs" / "rankings")
    parser.add_argument("--base-noise-scale", type=float, default=0.10)
    parser.add_argument("--robustness-scales", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    parser.add_argument("--dp-privacy-budgets", type=float, nargs="*", default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": output_dir / "tables",
        "splits": output_dir / "splits",
        "rankings": output_dir / "rankings",
        "logs": output_dir / "logs",
        "data": output_dir / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(
        "%s:%s: %s: %s", filename, lineno, category.__name__, message
    )


def ensure_dataset(data_path: Path | None, data_dir: Path) -> Path:
    if data_path is not None:
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        return data_path
    destination = data_dir / "cs-training.csv"
    if not destination.exists():
        logging.info("Downloading dataset to %s", destination)
        urlretrieve(DATA_URL, destination)
    return destination


def load_data(raw_path: Path, dirs: dict[str, Path], test_size: float, seed: int):
    raw = pd.read_csv(raw_path)
    if raw.columns[0].startswith("Unnamed"):
        raw = raw.drop(columns=[raw.columns[0]])
    for column in ["id", "ID", "Id"]:
        if column in raw.columns:
            raw = raw.drop(columns=[column])
    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    raw = raw[EXPECTED_COLUMNS].copy()
    raw[TARGET] = raw[TARGET].astype(int)
    age_median = float(raw[AGE].median())
    sensitive = (raw[AGE] >= age_median).astype(int).rename("A_age_ge_median")
    features = raw.drop(columns=[TARGET, AGE])
    target = raw[TARGET]

    train_idx, test_idx = train_test_split(
        np.arange(len(raw)),
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
    x_train_raw = features.iloc[train_idx].reset_index(drop=True)
    x_test_raw = features.iloc[test_idx].reset_index(drop=True)
    y_train = target.iloc[train_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)
    a_train = sensitive.iloc[train_idx].reset_index(drop=True)
    a_test = sensitive.iloc[test_idx].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(x_train_raw)), columns=features.columns)
    x_test = pd.DataFrame(scaler.transform(imputer.transform(x_test_raw)), columns=features.columns)

    pd.DataFrame({"train_index": train_idx}).to_csv(dirs["splits"] / "train_indices.csv", index=False)
    pd.DataFrame({"test_index": test_idx}).to_csv(dirs["splits"] / "test_indices.csv", index=False)
    pd.DataFrame({"feature": list(features.columns)}).to_csv(dirs["data"].parent / "feature_names.csv", index=False)

    logging.info("Dataset shape: %s", raw.shape)
    logging.info("Train/test sizes: %s / %s", len(train_idx), len(test_idx))
    logging.info("Final feature list: %s", list(features.columns))
    logging.info("Sensitive distribution train=%s test=%s", a_train.value_counts(normalize=True).to_dict(), a_test.value_counts(normalize=True).to_dict())
    logging.info("Target distribution train=%s test=%s", y_train.value_counts(normalize=True).to_dict(), y_test.value_counts(normalize=True).to_dict())
    return x_train, x_test, y_train, y_test, a_train, a_test, list(features.columns)


def copy_rankings_if_available(source_dir: Path, ranking_dir: Path) -> None:
    names = ["lime_ranking.csv", "shap_ranking.csv", "corex_banzhaf_ranking.csv", "corex_least_core_ranking.csv", "corex_nucleolus_ranking.csv"]
    if not source_dir.exists():
        logging.warning("Ranking source directory not found: %s", source_dir)
        return
    for name in names:
        source = source_dir / name
        destination = ranking_dir / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
            logging.info("Copied ranking %s", source)


def require_rankings(ranking_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "lime": ranking_dir / "lime_ranking.csv",
        "shap": ranking_dir / "shap_ranking.csv",
        "corex": ranking_dir / "corex_banzhaf_ranking.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Table 1 rankings. Run the Table 1 pipeline first or pass --ranking-source-dir. Missing: "
            + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in files.items()}


def score_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if column not in {"feature", "rank"} and pd.api.types.is_numeric_dtype(df[column]):
            return column
    raise ValueError(f"No numeric score column found in ranking columns: {list(df.columns)}")


def top_features(ranking: pd.DataFrame, k: int) -> list[str]:
    ordered = ranking.sort_values("rank") if "rank" in ranking.columns else ranking.copy()
    return ordered.head(k)["feature"].astype(str).tolist()


def make_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def compute_dp_gap(y_pred: np.ndarray, a: pd.Series) -> float:
    groups = []
    a_values = a.to_numpy(dtype=int)
    for group in [0, 1]:
        mask = a_values == group
        groups.append(float(np.mean(y_pred[mask])) if mask.any() else np.nan)
    return float(abs(groups[1] - groups[0]))


def compute_eo_gap(y_true: pd.Series, y_pred: np.ndarray, a: pd.Series) -> float:
    tprs = []
    y_values = y_true.to_numpy(dtype=int)
    a_values = a.to_numpy(dtype=int)
    for group in [0, 1]:
        mask = (a_values == group) & (y_values == 1)
        if not mask.any():
            logging.warning("EO gap undefined: group A=%s has no positive true labels.", group)
            return float("nan")
        tprs.append(float(np.mean(y_pred[mask])))
    return float(abs(tprs[1] - tprs[0]))


def evaluate_setting(method_name: str, x_train_method: pd.DataFrame, x_test_method: pd.DataFrame, y_train, y_test, a_train, a_test, perturbed_features: list[str], noise_config: dict, seed: int) -> dict[str, object]:
    predictive_model = make_model(seed)
    predictive_model.fit(x_train_method, y_train)
    task_proba = predictive_model.predict_proba(x_test_method)[:, 1]
    y_pred = (task_proba >= 0.5).astype(int)

    attack_model = make_model(seed)
    attack_model.fit(x_train_method, a_train)
    attack_proba = attack_model.predict_proba(x_test_method)[:, 1]

    row = {
        "method": method_name,
        "task_auc": float(roc_auc_score(y_test, task_proba)),
        "attack_auc": float(roc_auc_score(a_test, attack_proba)),
        "dp_gap": compute_dp_gap(y_pred, a_test),
        "eo_gap": compute_eo_gap(y_test, y_pred, a_test),
        "perturbed_features": ";".join(perturbed_features) if perturbed_features else "none",
        "noise_config": json.dumps(noise_config, sort_keys=True),
    }
    logging.info("Metrics %s: %s", method_name, row)
    return row


def add_laplace_noise(x: pd.DataFrame, scales: dict[str, float], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = x.copy()
    for feature, scale in scales.items():
        if scale > 0:
            out[feature] = out[feature] + rng.laplace(loc=0.0, scale=float(scale), size=len(out))
    return out


def dp_feature_scales(x_train: pd.DataFrame, features: list[str], epsilon: float) -> dict[str, float]:
    if epsilon <= 0:
        raise ValueError("DP privacy budget epsilon must be positive.")
    scales = {feature: 0.0 for feature in x_train.columns}
    for feature in features:
        sensitivity = float(x_train[feature].max() - x_train[feature].min())
        scales[feature] = sensitivity / float(epsilon)
    return scales


def dp_adaptive_scales(x_train: pd.DataFrame, corex_ranking: pd.DataFrame, feature_names: list[str], epsilon: float) -> dict[str, float]:
    base_scales = dp_feature_scales(x_train, feature_names, epsilon)
    column = score_column(corex_ranking)
    score_map = dict(zip(corex_ranking["feature"].astype(str), corex_ranking[column].astype(float)))
    positive = {feature: max(float(score_map.get(feature, 0.0)), 0.0) for feature in feature_names}
    max_positive = max(positive.values()) if positive else 0.0
    if max_positive <= 1e-12:
        return base_scales
    # Higher proxy risk receives stronger noise; low-risk features still receive
    # a small DP noise floor so every non-sensitive feature participates.
    return {
        feature: base_scales[feature] * (0.10 + 0.90 * positive[feature] / max_positive)
        for feature in feature_names
    }


def adaptive_scales(corex_ranking: pd.DataFrame, feature_names: list[str], min_scale: float = 0.02, max_scale: float = 0.20) -> dict[str, float]:
    column = score_column(corex_ranking)
    score_map = dict(zip(corex_ranking["feature"].astype(str), corex_ranking[column].astype(float)))
    positive = {feature: max(float(score_map.get(feature, 0.0)), 0.0) for feature in feature_names}
    total = sum(positive.values()) + 1e-12
    normalized = {feature: value / total for feature, value in positive.items()}
    max_norm = max(normalized.values()) if normalized else 1.0
    return {
        feature: float(min_scale + (max_scale - min_scale) * normalized[feature] / max(max_norm, 1e-12))
        for feature in feature_names
    }


def method_scales(method: str, feature_names: list[str], rankings: dict[str, pd.DataFrame], top_k: int, base_noise_scale: float):
    zero = {feature: 0.0 for feature in feature_names}
    if method == "No Privacy":
        return zero, [], {"type": "none"}
    if method == "Uniform Perturbation":
        return {feature: base_noise_scale for feature in feature_names}, list(feature_names), {"type": "uniform_laplace", "base_noise_scale": base_noise_scale}
    ranking_key = {"LIME-Targeted": "lime", "SHAP-Targeted": "shap", "COREX-Targeted": "corex"}.get(method)
    if ranking_key:
        selected = top_features(rankings[ranking_key], top_k)
        scales = {feature: (base_noise_scale if feature in selected else 0.0) for feature in feature_names}
        return scales, selected, {"type": "targeted_laplace", "base_noise_scale": base_noise_scale, "top_k": top_k, "ranking": ranking_key}
    if method == "COREX-Adaptive":
        scales = adaptive_scales(rankings["corex"], feature_names)
        return scales, list(feature_names), {"type": "adaptive_laplace", "min_noise_scale": 0.02, "max_noise_scale": 0.20, "ranking": "corex_banzhaf"}
    raise ValueError(method)


def method_dp_scales(method: str, x_train: pd.DataFrame, feature_names: list[str], rankings: dict[str, pd.DataFrame], top_k: int, epsilon: float):
    zero = {feature: 0.0 for feature in feature_names}
    if method == "No Privacy":
        return zero, [], {"type": "none", "privacy_budget": "none"}
    if method == "Uniform Perturbation":
        return dp_feature_scales(x_train, feature_names, epsilon), list(feature_names), {"type": "uniform_dp_laplace", "epsilon": epsilon}
    ranking_key = {"LIME-Targeted": "lime", "SHAP-Targeted": "shap", "COREX-Targeted": "corex"}.get(method)
    if ranking_key:
        selected = top_features(rankings[ranking_key], top_k)
        return dp_feature_scales(x_train, selected, epsilon), selected, {
            "type": "targeted_dp_laplace",
            "epsilon": epsilon,
            "top_k": top_k,
            "ranking": ranking_key,
        }
    if method == "COREX-Adaptive":
        return dp_adaptive_scales(x_train, rankings["corex"], feature_names, epsilon), list(feature_names), {
            "type": "adaptive_dp_laplace",
            "epsilon": epsilon,
            "ranking": "corex_banzhaf",
        }
    raise ValueError(method)


def run_methods(x_train, x_test, y_train, y_test, a_train, a_test, feature_names, rankings, top_k: int, base_noise_scale: float, seed: int):
    rows = []
    noise_rows = []
    for index, method in enumerate(METHOD_ORDER):
        scales, perturbed_features, noise_config = method_scales(method, feature_names, rankings, top_k, base_noise_scale)
        x_train_method = add_laplace_noise(x_train, scales, seed + 1000 + index)
        x_test_method = add_laplace_noise(x_test, scales, seed + 2000 + index)
        rows.append(evaluate_setting(method, x_train_method, x_test_method, y_train, y_test, a_train, a_test, perturbed_features, noise_config, seed))
        source = noise_config.get("ranking", "none" if method == "No Privacy" else "uniform")
        for feature in feature_names:
            noise_rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "noise_scale": scales[feature],
                    "was_perturbed": bool(scales[feature] > 0),
                    "source_ranking": source,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(noise_rows)


def run_dp_budget_methods(x_train, x_test, y_train, y_test, a_train, a_test, feature_names, rankings, top_k: int, epsilon: float, seed: int) -> pd.DataFrame:
    rows = []
    for index, method in enumerate(METHOD_ORDER):
        scales, perturbed_features, noise_config = method_dp_scales(method, x_train, feature_names, rankings, top_k, epsilon)
        x_train_method = add_laplace_noise(x_train, scales, seed + 10_000 + int(epsilon * 1_000_000) + index)
        x_test_method = add_laplace_noise(x_test, scales, seed + 20_000 + int(epsilon * 1_000_000) + index)
        row = evaluate_setting(method, x_train_method, x_test_method, y_train, y_test, a_train, a_test, perturbed_features, noise_config, seed)
        row = {"privacy_budget": epsilon, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def paper_ready(table: pd.DataFrame) -> pd.DataFrame:
    out = table[["method", "task_auc", "attack_auc", "dp_gap", "eo_gap"]].copy()
    out.columns = ["Method", "Task AUC", "Attack AUC", "DP Gap", "EO Gap"]
    for column in ["Task AUC", "Attack AUC", "DP Gap", "EO Gap"]:
        out[column] = out[column].map(lambda value: round(float(value), 3))
    return out


def latex_table(paper: pd.DataFrame) -> str:
    rows = []
    for _, row in paper.iterrows():
        rows.append(
            f"{row['Method']} & {row['Task AUC']:.3f} & {row['Attack AUC']:.3f} & {row['DP Gap']:.3f} & {row['EO Gap']:.3f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Privacy--utility results on Give Me Some Credit. Higher task AUC is better; lower attack AUC, DP gap, and EO gap are better.}",
            r"\label{tab:privacy_results}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Task AUC & Attack AUC & DP Gap & EO Gap \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def interpretation(table: pd.DataFrame) -> str:
    best_task = table.loc[table["task_auc"].idxmax()]
    best_attack = table.loc[table["attack_auc"].idxmin()]
    best_dp = table.loc[table["dp_gap"].idxmin()]
    best_eo = table.loc[table["eo_gap"].idxmin()]
    adaptive = table.loc[table["method"] == "COREX-Adaptive"].iloc[0]
    uniform = table.loc[table["method"] == "Uniform Perturbation"].iloc[0]
    targeted = table.loc[table["method"] == "COREX-Targeted"].iloc[0]
    balance = adaptive["task_auc"] > uniform["task_auc"] and (
        adaptive["attack_auc"] <= targeted["attack_auc"] or adaptive["dp_gap"] <= targeted["dp_gap"] or adaptive["eo_gap"] <= targeted["eo_gap"]
    )
    lines = [
        f"Highest task AUC: {best_task['method']} ({best_task['task_auc']:.3f}).",
        f"Lowest attack AUC: {best_attack['method']} ({best_attack['attack_auc']:.3f}).",
        f"Lowest DP gap: {best_dp['method']} ({best_dp['dp_gap']:.3f}).",
        f"Lowest EO gap: {best_eo['method']} ({best_eo['eo_gap']:.3f}).",
        f"COREX-Adaptive gives the best privacy--utility balance: {'yes' if balance else 'no'}.",
    ]
    if adaptive["attack_auc"] > uniform["attack_auc"] or adaptive["attack_auc"] > targeted["attack_auc"]:
        lines.append("Warning: COREX-Adaptive does not improve attack AUC compared with Uniform Perturbation or COREX-Targeted.")
    return "\n".join(lines) + "\n"


def write_notebook(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# Table 2 Privacy--Utility Experiment\n"]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["%run ../scripts/run_table2_privacy_utility.py\n"]},
                ],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    dirs = make_dirs(args.output_dir)
    setup_logging(dirs["logs"] / "table2_privacy_utility.log")
    write_notebook(EXPERIMENT_DIR / "notebooks" / "table2_privacy_utility.ipynb")
    (args.output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    raw_path = ensure_dataset(args.data_path, dirs["data"])
    x_train, x_test, y_train, y_test, a_train, a_test, feature_names = load_data(raw_path, dirs, args.test_size, args.seed)
    copy_rankings_if_available(args.ranking_source_dir, dirs["rankings"])
    rankings = require_rankings(dirs["rankings"])
    logging.info("Top-5 LIME features: %s", top_features(rankings["lime"], args.top_k))
    logging.info("Top-5 SHAP features: %s", top_features(rankings["shap"], args.top_k))
    logging.info("Top-5 COREX features: %s", top_features(rankings["corex"], args.top_k))

    table, noise_summary = run_methods(
        x_train, x_test, y_train, y_test, a_train, a_test, feature_names, rankings, args.top_k, args.base_noise_scale, args.seed
    )
    table.to_csv(dirs["tables"] / "table2_privacy_utility.csv", index=False)
    paper = paper_ready(table)
    paper.to_csv(dirs["tables"] / "table2_privacy_utility_paper_ready.csv", index=False)
    (dirs["tables"] / "table2_privacy_utility_latex.txt").write_text(latex_table(paper), encoding="utf-8")
    (dirs["tables"] / "table2_interpretation.txt").write_text(interpretation(table), encoding="utf-8")
    noise_summary.to_csv(dirs["tables"] / "table2_noise_summary.csv", index=False)

    robust_rows = []
    for scale in args.robustness_scales:
        robust, _ = run_methods(x_train, x_test, y_train, y_test, a_train, a_test, feature_names, rankings, args.top_k, scale, args.seed)
        robust.insert(0, "noise_scale_setting", scale)
        robust_rows.append(robust[["noise_scale_setting", "method", "task_auc", "attack_auc", "dp_gap", "eo_gap"]])
    if robust_rows:
        pd.concat(robust_rows, ignore_index=True).to_csv(dirs["tables"] / "table2_privacy_utility_robustness.csv", index=False)

    dp_rows = []
    for epsilon in args.dp_privacy_budgets:
        logging.info("Running DP privacy budget sweep epsilon=%s", epsilon)
        dp_rows.append(run_dp_budget_methods(x_train, x_test, y_train, y_test, a_train, a_test, feature_names, rankings, args.top_k, epsilon, args.seed))
    pd.concat(dp_rows, ignore_index=True).to_csv(dirs["tables"] / "table2_privacy_utility_dp_sweep.csv", index=False)
    logging.info("Wrote Table 2 outputs to %s", dirs["tables"])


if __name__ == "__main__":
    main()
