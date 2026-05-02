#!/usr/bin/env python3
"""Optimize only the COREX-Adaptive privacy mechanism for Table 2."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


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
    "COREX-Adaptive Original",
    "COREX-Adaptive Optimized",
]
PRIVACY_BUDGETS = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize COREX-Adaptive privacy mechanism.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--attack-tolerance", type=float, default=0.01)
    parser.add_argument("--dp-tolerance", type=float, default=0.005)
    parser.add_argument("--eo-tolerance", type=float, default=0.005)
    parser.add_argument("--task-tolerance", type=float, default=0.03)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--optuna-trials", type=int, default=100)
    parser.add_argument("--nsga-trials", type=int, default=100)
    parser.add_argument("--search-train-sample", type=int, default=3000)
    parser.add_argument("--search-val-sample", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "adaptive": output_dir / "adaptive_optimization",
        "logs": output_dir / "logs",
        "data": output_dir / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(
        "%s:%s: %s: %s", filename, lineno, category.__name__, message
    )


def ensure_dataset(args: argparse.Namespace, dirs: dict[str, Path]) -> Path:
    if args.data_path is not None:
        return args.data_path
    path = dirs["data"] / "cs-training.csv"
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return path


def load_data(path: Path, seed: int):
    raw = pd.read_csv(path)
    if raw.columns[0].startswith("Unnamed"):
        raw = raw.drop(columns=[raw.columns[0]])
    for column in ["id", "ID", "Id"]:
        if column in raw.columns:
            raw = raw.drop(columns=[column])
    raw = raw[EXPECTED_COLUMNS].copy()
    y = raw[TARGET].astype(int)
    a = (raw[AGE] >= float(raw[AGE].median())).astype(int)
    x_raw = raw.drop(columns=[TARGET, AGE])
    train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=seed, stratify=y)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(x_raw.iloc[train_idx])), columns=x_raw.columns)
    x_test = pd.DataFrame(scaler.transform(imputer.transform(x_raw.iloc[test_idx])), columns=x_raw.columns)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    a_train, a_test = a.iloc[train_idx].reset_index(drop=True), a.iloc[test_idx].reset_index(drop=True)
    return x_train, x_test, y_train, y_test, a_train, a_test, list(x_raw.columns)


def model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)


def dp_gap(y_pred: np.ndarray, a: pd.Series) -> float:
    av = a.to_numpy(dtype=int)
    rates = [float(y_pred[av == g].mean()) for g in [0, 1]]
    return abs(rates[1] - rates[0])


def eo_gap(y_true: pd.Series, y_pred: np.ndarray, a: pd.Series) -> float:
    yv, av = y_true.to_numpy(dtype=int), a.to_numpy(dtype=int)
    tprs = []
    for g in [0, 1]:
        mask = (av == g) & (yv == 1)
        if not mask.any():
            logging.warning("EO undefined for group %s", g)
            return float("nan")
        tprs.append(float(y_pred[mask].mean()))
    return abs(tprs[1] - tprs[0])


def evaluate_frames(x_train, x_eval, y_train, y_eval, a_train, a_eval, seed: int) -> dict[str, float]:
    f = model(seed)
    f.fit(x_train, y_train)
    task_p = f.predict_proba(x_eval)[:, 1]
    pred = (task_p >= 0.5).astype(int)
    g = model(seed)
    g.fit(x_train, a_train)
    attack_p = g.predict_proba(x_eval)[:, 1]
    return {
        "task_auc": float(roc_auc_score(y_eval, task_p)),
        "attack_auc": float(roc_auc_score(a_eval, attack_p)),
        "dp_gap": dp_gap(pred, a_eval),
        "eo_gap": eo_gap(y_eval, pred, a_eval),
    }


def laplace_perturb(x: pd.DataFrame, scales: dict[str, float], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = x.copy()
    for feature, scale in scales.items():
        if scale > 0:
            out[feature] = out[feature] + rng.laplace(0.0, float(scale), size=len(out))
    return out


def score_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col not in {"feature", "rank"} and pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError("COREX ranking has no numeric score column.")


def load_risk(output_dir: Path, feature_names: list[str]) -> dict[str, float]:
    path = output_dir / "rankings" / "corex_banzhaf_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ranking: {path}. Run Table 2 pipeline first.")
    ranking = pd.read_csv(path)
    col = score_column(ranking)
    raw = dict(zip(ranking["feature"].astype(str), ranking[col].astype(float)))
    risk = {f: max(float(raw.get(f, 0.0)), 0.0) for f in feature_names}
    total = sum(risk.values())
    if total <= 1e-12:
        logging.warning("All COREX risk scores are zero; using uniform risk.")
        return {f: 1.0 / len(feature_names) for f in feature_names}
    return {f: risk[f] / total for f in feature_names}


def task_importance(x_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str], seed: int) -> dict[str, float]:
    f = model(seed)
    f.fit(x_train, y_train)
    imp = np.asarray(f.feature_importances_, dtype=float)
    imp = imp / (imp.sum() + 1e-12)
    return dict(zip(feature_names, imp))


def transform_risk(config: dict, risk: dict[str, float], feature_names: list[str]) -> dict[str, float]:
    arr = np.asarray([risk[f] for f in feature_names], dtype=float)
    kind = config.get("transform", "linear")
    if kind == "power":
        arr = np.power(arr, float(config.get("gamma", 1.0)))
    elif kind == "softmax":
        temp = float(config.get("temperature", 1.0))
        z = arr / max(temp, 1e-12)
        z = z - z.max()
        arr = np.exp(z)
    elif kind == "rank":
        gamma = float(config.get("gamma", 1.0))
        ranks = np.arange(1, len(feature_names) + 1, dtype=float)
        order = np.argsort(-np.asarray([risk[f] for f in feature_names]))
        weights = np.zeros(len(feature_names), dtype=float)
        weights[order] = 1.0 / np.power(ranks, gamma)
        arr = weights
    total = arr.sum()
    if total <= 1e-12:
        arr = np.ones(len(feature_names), dtype=float) / len(feature_names)
    else:
        arr = arr / total
    return dict(zip(feature_names, arr))


def scales_for_config(config: dict, risk: dict[str, float], task_imp: dict[str, float], feature_names: list[str]):
    variant = config["adaptive_variant"]
    if variant == "utility_aware":
        alpha, beta, lam = float(config["alpha_proxy"]), float(config["beta_utility"]), float(config["lambda_utility"])
        arr = np.asarray([
            (risk[f] ** alpha) / ((task_imp[f] + 1e-12) ** beta + lam)
            for f in feature_names
        ])
        arr = arr / (arr.sum() + 1e-12)
        transformed = dict(zip(feature_names, arr))
    else:
        transformed = transform_risk(config, risk, feature_names)
    min_n, max_n = float(config["min_noise"]), float(config["max_noise"])
    max_t = max(transformed.values()) if transformed else 1.0
    scales = {f: min_n + (max_n - min_n) * transformed[f] / max(max_t, 1e-12) for f in feature_names}
    if variant == "sparse":
        threshold = float(config.get("risk_threshold", 0.0))
        scales = {f: (scales[f] if transformed[f] >= threshold else 0.0) for f in feature_names}
    elif variant == "top_m":
        m = config.get("m", "all")
        if m != "all":
            selected = set(sorted(feature_names, key=lambda f: transformed[f], reverse=True)[: int(m)])
            scales = {f: (scales[f] if f in selected else 0.0) for f in feature_names}
    elif variant == "bounded":
        cap = float(config.get("cap", max_n))
        scales = {f: min(scales[f], cap) for f in feature_names}
    selected = [f for f in feature_names if scales[f] > 0]
    return scales, transformed, selected


def candidate_score(metrics: dict[str, float], target: dict[str, float], tol: dict[str, float]) -> tuple[bool, float]:
    attack_limit = target["attack_auc"] + tol["attack"]
    dp_limit = target["dp_gap"] + tol["dp"]
    eo_limit = target["eo_gap"] + tol["eo"]
    task_floor = max(0.70, target["task_auc"] - tol["task"])
    feasible = (
        metrics["attack_auc"] <= attack_limit
        and metrics["dp_gap"] <= dp_limit
        and metrics["eo_gap"] <= eo_limit
        and metrics["task_auc"] >= task_floor
    )
    score = (
        metrics["task_auc"]
        - 2.0 * max(0.0, metrics["attack_auc"] - attack_limit)
        - max(0.0, metrics["dp_gap"] - dp_limit)
        - max(0.0, metrics["eo_gap"] - eo_limit)
        - 2.0 * max(0.0, task_floor - metrics["task_auc"])
    )
    return feasible, float(score)


def evaluate_adaptive_config(config: dict, x_train, x_eval, y_train, y_eval, a_train, a_eval, risk, task_imp, feature_names, seed: int):
    scales, transformed, selected = scales_for_config(config, risk, task_imp, feature_names)
    x_train_p = laplace_perturb(x_train, scales, seed + int(config["config_id"]) * 2 + 1)
    x_eval_p = laplace_perturb(x_eval, scales, seed + int(config["config_id"]) * 2 + 2)
    metrics = evaluate_frames(x_train_p, x_eval_p, y_train, y_eval, a_train, a_eval, seed)
    return metrics, scales, transformed, selected


def base_configs() -> list[dict]:
    configs = []
    cid = 0
    transforms = [("linear", {}), *[("power", {"gamma": g}) for g in [0.5, 1.0, 2.0]], *[("rank", {"gamma": g}) for g in [0.5, 1.0, 2.0]]]
    for variant in ["dense", "bounded"]:
        for (tr, extra), min_n, max_n in itertools.product(transforms, [0.0, 0.01, 0.02], [0.05, 0.10, 0.20]):
            if max_n <= min_n:
                continue
            cfg = {"config_id": cid, "adaptive_variant": variant, "transform": tr, "min_noise": min_n, "max_noise": max_n, **extra}
            if variant == "bounded":
                cfg["cap"] = max_n
            configs.append(cfg)
            cid += 1
    for threshold in [0.02, 0.05, 0.10, 0.15]:
        for min_n, max_n in [(0.0, 0.05), (0.005, 0.075), (0.01, 0.10), (0.02, 0.15)]:
            configs.append({"config_id": cid, "adaptive_variant": "sparse", "transform": "power", "gamma": 1.0, "min_noise": min_n, "max_noise": max_n, "risk_threshold": threshold})
            cid += 1
    for m in [1, 2, 3, 4, 5, 6, 7, "all"]:
        for min_n, max_n in [(0.0, 0.05), (0.005, 0.075), (0.01, 0.10), (0.02, 0.15)]:
            configs.append({"config_id": cid, "adaptive_variant": "top_m", "transform": "rank", "gamma": 1.0, "min_noise": min_n, "max_noise": max_n, "m": m})
            cid += 1
    for lam, alpha, beta, min_n, max_n in itertools.product([0.05, 0.25, 1.0], [0.5, 1.0, 2.0], [0.5, 1.0], [0.0, 0.01], [0.05, 0.10]):
        configs.append({"config_id": cid, "adaptive_variant": "utility_aware", "min_noise": min_n, "max_noise": max_n, "lambda_utility": lam, "alpha_proxy": alpha, "beta_utility": beta})
        cid += 1
    return configs


def random_configs(start_id: int, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = []
    variants = ["dense", "sparse", "top_m", "utility_aware", "bounded"]
    for i in range(n):
        variant = str(rng.choice(variants))
        min_n = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.05]))
        max_n = float(rng.choice([0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]))
        if max_n <= min_n:
            max_n = min_n + 0.05
        cfg = {"config_id": start_id + i, "adaptive_variant": variant, "min_noise": min_n, "max_noise": max_n}
        if variant == "utility_aware":
            cfg.update({
                "lambda_utility": float(rng.choice([0.01, 0.05, 0.10, 0.25, 0.50, 1.00])),
                "alpha_proxy": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
                "beta_utility": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            })
        else:
            transform = str(rng.choice(["linear", "power", "softmax", "rank"]))
            cfg["transform"] = transform
            if transform in {"power", "rank"}:
                cfg["gamma"] = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
            if transform == "softmax":
                cfg["temperature"] = float(rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.0]))
            if variant == "sparse":
                cfg["risk_threshold"] = float(rng.choice([0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]))
            if variant == "top_m":
                cfg["m"] = rng.choice([1, 2, 3, 4, 5, 6, 7, "all"]).item()
            if variant == "bounded":
                cfg["cap"] = max_n
        configs.append(cfg)
    return configs


def optuna_configs(start_id: int, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    if optuna is None:
        logging.warning("Optuna unavailable; skipping Bayesian and NSGA-II searches.")
        return [], []
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    configs = []
    def objective(trial):
        cfg = suggest_config(trial, start_id + len(configs), "bayesian")
        configs.append(cfg)
        return 0.0
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    nsga_sampler = optuna.samplers.NSGAIISampler(seed=seed)
    nsga = optuna.create_study(directions=["maximize", "minimize", "minimize", "minimize"], sampler=nsga_sampler)
    nsga_configs = []
    def mo_objective(trial):
        cfg = suggest_config(trial, start_id + n + len(nsga_configs), "multi_objective")
        nsga_configs.append(cfg)
        return 0.0, 1.0, 1.0, 1.0
    nsga.optimize(mo_objective, n_trials=n, show_progress_bar=False)
    return configs, nsga_configs


def suggest_config(trial, cid: int, search_name: str) -> dict:
    variant = trial.suggest_categorical("variant", ["dense", "sparse", "top_m", "utility_aware", "bounded"])
    min_n = trial.suggest_categorical("min_noise", [0.0, 0.005, 0.01, 0.02, 0.05])
    max_n = trial.suggest_categorical("max_noise", [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
    if max_n <= min_n:
        max_n = min_n + 0.05
    cfg = {"config_id": cid, "adaptive_variant": variant, "min_noise": min_n, "max_noise": max_n, "search_name": search_name}
    if variant == "utility_aware":
        cfg.update({
            "lambda_utility": trial.suggest_categorical("lambda_utility", [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]),
            "alpha_proxy": trial.suggest_categorical("alpha_proxy", [0.5, 1.0, 1.5, 2.0]),
            "beta_utility": trial.suggest_categorical("beta_utility", [0.5, 1.0, 1.5, 2.0]),
        })
    else:
        tr = trial.suggest_categorical("transform", ["linear", "power", "softmax", "rank"])
        cfg["transform"] = tr
        if tr in {"power", "rank"}:
            cfg["gamma"] = trial.suggest_categorical("gamma", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
        if tr == "softmax":
            cfg["temperature"] = trial.suggest_categorical("temperature", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
        if variant == "sparse":
            cfg["risk_threshold"] = trial.suggest_categorical("risk_threshold", [0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20])
        if variant == "top_m":
            cfg["m"] = trial.suggest_categorical("m", [1, 2, 3, 4, 5, 6, 7, "all"])
        if variant == "bounded":
            cfg["cap"] = max_n
    return cfg


def load_reference(output_dir: Path) -> pd.DataFrame:
    dp_path = output_dir / "tables" / "table2_privacy_utility_dp_sweep.csv"
    if not dp_path.exists():
        raise FileNotFoundError(f"Missing DP sweep: {dp_path}. Run run_table2_privacy_utility.py first.")
    df = pd.read_csv(dp_path)
    return df[df["privacy_budget"].isin(PRIVACY_BUDGETS)].copy()


def write_notebook(path: Path) -> None:
    path.write_text(json.dumps({
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Optimize COREX-Adaptive\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["%run ../scripts/optimize_corex_adaptive.py\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    dirs = make_dirs(args.output_dir)
    setup_logging(dirs["logs"] / "adaptive_optimization.log")
    write_notebook(EXPERIMENT_DIR / "notebooks" / "optimize_corex_adaptive.ipynb")
    x_train, x_test, y_train, y_test, a_train, a_test, feature_names = load_data(ensure_dataset(args, dirs), args.seed)
    risk = load_risk(args.output_dir, feature_names)
    task_imp = task_importance(x_train, y_train, feature_names, args.seed)
    logging.info("Feature names: %s", feature_names)
    logging.info("Risk scores: %s", risk)
    logging.info("Task importance scores: %s", task_imp)
    tol = {"attack": args.attack_tolerance, "dp": args.dp_tolerance, "eo": args.eo_tolerance, "task": args.task_tolerance}
    logging.info("Tolerances: %s", tol)

    train_idx, val_idx = train_test_split(np.arange(len(x_train)), test_size=0.2, random_state=args.seed, stratify=y_train)
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(train_idx, size=min(args.search_train_sample, len(train_idx)), replace=False)
    val_idx = rng.choice(val_idx, size=min(args.search_val_sample, len(val_idx)), replace=False)
    x_inner, y_inner, a_inner = x_train.iloc[train_idx].reset_index(drop=True), y_train.iloc[train_idx].reset_index(drop=True), a_train.iloc[train_idx].reset_index(drop=True)
    x_val, y_val, a_val = x_train.iloc[val_idx].reset_index(drop=True), y_train.iloc[val_idx].reset_index(drop=True), a_train.iloc[val_idx].reset_index(drop=True)
    logging.info("Search uses inner_train=%s validation=%s", len(x_inner), len(x_val))

    configs = [("exhaustive_grid", c) for c in base_configs()]
    start = max(c["config_id"] for _, c in configs) + 1
    configs += [("random_search", c) for c in random_configs(start, args.random_trials, args.seed)]
    start = max(c["config_id"] for _, c in configs) + 1
    bayes, nsga = optuna_configs(start, args.optuna_trials, args.seed)
    configs += [("bayesian_tpe", c) for c in bayes]
    configs += [("multi_objective_nsga2", c) for c in nsga]
    logging.info("Number of candidates tested: %s", len(configs))

    reference = load_reference(args.output_dir)
    target_by_budget = {
        eps: reference[(reference["privacy_budget"] == eps) & (reference["method"] == "COREX-Targeted")].iloc[0].to_dict()
        for eps in PRIVACY_BUDGETS
    }
    logging.info("COREX-Targeted reference values: %s", target_by_budget)

    evaluated = []
    noise_cache = {}
    for index, (search_method, cfg) in enumerate(configs, start=1):
        if index == 1 or index % 25 == 0 or index == len(configs):
            logging.info("Evaluating adaptive candidate %s/%s", index, len(configs))
        metrics, scales, transformed, selected = evaluate_adaptive_config(cfg, x_inner, x_val, y_inner, y_val, a_inner, a_val, risk, task_imp, feature_names, args.seed)
        noise_cache[cfg["config_id"]] = (scales, transformed, selected, cfg)
        for eps in PRIVACY_BUDGETS:
            target = target_by_budget[eps]
            feasible, score = candidate_score(metrics, target, tol)
            evaluated.append({
                "privacy_budget": eps,
                "search_method": search_method,
                "adaptive_variant": cfg["adaptive_variant"],
                "config_id": cfg["config_id"],
                "task_auc_val": metrics["task_auc"],
                "attack_auc_val": metrics["attack_auc"],
                "dp_gap_val": metrics["dp_gap"],
                "eo_gap_val": metrics["eo_gap"],
                "feasible": feasible,
                "score": score,
                "config_json": json.dumps(cfg, sort_keys=True),
            })
    all_results = pd.DataFrame(evaluated)
    all_results.to_csv(dirs["adaptive"] / "adaptive_search_all_results.csv", index=False)
    feasible_count = int(all_results["feasible"].sum())
    logging.info("Number of feasible candidate-budget pairs: %s", feasible_count)

    best_rows = []
    noise_rows = []
    optimized_rows = []
    for eps in PRIVACY_BUDGETS:
        subset = all_results[all_results["privacy_budget"] == eps].copy()
        if subset["feasible"].any():
            candidates = subset[subset["feasible"]].sort_values(["task_auc_val", "attack_auc_val", "dp_gap_val", "eo_gap_val"], ascending=[False, True, True, True])
        else:
            candidates = subset.sort_values("score", ascending=False)
        best = candidates.iloc[0]
        scales, transformed, selected, cfg = noise_cache[int(best["config_id"])]
        final_metrics, final_scales, final_transformed, final_selected = evaluate_adaptive_config(cfg, x_train, x_test, y_train, y_test, a_train, a_test, risk, task_imp, feature_names, args.seed)
        target = target_by_budget[eps]
        final_feasible, _ = candidate_score(final_metrics, target, tol)
        best_row = {
            "privacy_budget": eps,
            "search_method": best["search_method"],
            "adaptive_variant": cfg["adaptive_variant"],
            "task_auc": final_metrics["task_auc"],
            "attack_auc": final_metrics["attack_auc"],
            "dp_gap": final_metrics["dp_gap"],
            "eo_gap": final_metrics["eo_gap"],
            "feasible": final_feasible,
            "target_task_auc": target["task_auc"],
            "target_attack_auc": target["attack_auc"],
            "target_dp_gap": target["dp_gap"],
            "target_eo_gap": target["eo_gap"],
            "task_auc_delta_vs_corex_targeted": final_metrics["task_auc"] - target["task_auc"],
            "attack_auc_delta_vs_corex_targeted": final_metrics["attack_auc"] - target["attack_auc"],
            "dp_gap_delta_vs_corex_targeted": final_metrics["dp_gap"] - target["dp_gap"],
            "eo_gap_delta_vs_corex_targeted": final_metrics["eo_gap"] - target["eo_gap"],
            "selected_features": ";".join(final_selected),
            "noise_scales_json": json.dumps(final_scales, sort_keys=True),
            "config_json": json.dumps(cfg, sort_keys=True),
        }
        best_rows.append(best_row)
        optimized_rows.append({
            "privacy_budget": eps,
            "method": "COREX-Adaptive Optimized",
            **final_metrics,
            "perturbed_features": ";".join(final_selected),
            "noise_config": json.dumps({"optimized": True, "config_id": int(cfg["config_id"]), "variant": cfg["adaptive_variant"], "noise_scales": final_scales}, sort_keys=True),
        })
        for f in feature_names:
            noise_rows.append({
                "privacy_budget": eps,
                "feature": f,
                "proxy_risk": risk[f],
                "task_importance": task_imp[f],
                "transformed_risk": final_transformed[f],
                "noise_scale": final_scales[f],
                "was_perturbed": final_scales[f] > 0,
                "adaptive_variant": cfg["adaptive_variant"],
                "config_id": cfg["config_id"],
            })
        logging.info("Best config epsilon=%s: %s", eps, best_row)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(dirs["adaptive"] / "adaptive_best_by_budget.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(dirs["adaptive"] / "optimized_adaptive_noise_summary.csv", index=False)

    ref_rows = reference[reference["method"].isin(["No Privacy", "Uniform Perturbation", "LIME-Targeted", "SHAP-Targeted", "COREX-Targeted", "COREX-Adaptive"])].copy()
    ref_rows["method"] = ref_rows["method"].replace({"COREX-Adaptive": "COREX-Adaptive Original"})
    updated = pd.concat([ref_rows, pd.DataFrame(optimized_rows)], ignore_index=True)
    updated["method"] = pd.Categorical(updated["method"], METHOD_ORDER, ordered=True)
    updated = updated.sort_values(["privacy_budget", "method"])
    updated.to_csv(dirs["tables"] / "table2_privacy_utility_optimized_adaptive.csv", index=False)

    eps_table = updated[updated["privacy_budget"].round(3) == 0.1].copy()
    lines = []
    for method_name in METHOD_ORDER:
        row = eps_table[eps_table["method"] == method_name].iloc[0]
        lines.append(f"{method_name} & {row['task_auc']:.3f} & {row['attack_auc']:.3f} & {row['dp_gap']:.3f} & {row['eo_gap']:.3f} \\\\")
    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Privacy--utility results on Give Me Some Credit at $\epsilon=0.1$. Higher task AUC is better; lower attack AUC, DP gap, and EO gap are better.}",
        r"\label{tab:privacy_results}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Task AUC & Attack AUC & DP Gap & EO Gap \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (dirs["tables"] / "table2_privacy_utility_epsilon_0p1_optimized_latex.txt").write_text(latex, encoding="utf-8")

    interp = make_interpretation(updated, best_df)
    (dirs["adaptive"] / "adaptive_optimization_interpretation.txt").write_text(interp, encoding="utf-8")
    plot_outputs(updated, dirs["figures"])
    logging.info("Optimization complete.")


def make_interpretation(updated: pd.DataFrame, best: pd.DataFrame) -> str:
    eps = 0.1
    sub = updated[updated["privacy_budget"].round(3) == eps]
    orig = sub[sub["method"] == "COREX-Adaptive Original"].iloc[0]
    opt = sub[sub["method"] == "COREX-Adaptive Optimized"].iloc[0]
    targeted = sub[sub["method"] == "COREX-Targeted"].iloc[0]
    improves_original = opt["task_auc"] > orig["task_auc"] and opt["attack_auc"] <= orig["attack_auc"] + 0.01
    feasible_any = bool(best["feasible"].any())
    best_variant = best["adaptive_variant"].mode().iloc[0]
    best_search = best["search_method"].mode().iloc[0]
    beats_targeted = (
        opt["task_auc"] > targeted["task_auc"]
        and opt["attack_auc"] <= targeted["attack_auc"] + 0.01
        and opt["dp_gap"] <= targeted["dp_gap"] + 0.005
        and opt["eo_gap"] <= targeted["eo_gap"] + 0.005
    )
    lines = [
        f"Optimized adaptive improves over original adaptive at epsilon=0.1: {'yes' if improves_original else 'no'}.",
        f"Optimized adaptive meets attack/DP/EO tolerances for at least one privacy budget: {'yes' if feasible_any else 'no'}.",
        f"Optimized adaptive preserves task AUC at epsilon=0.1: {'yes' if opt['task_auc'] >= max(0.70, targeted['task_auc'] - 0.03) else 'no'}.",
        f"Most frequent best adaptive variant: {best_variant}.",
        f"Most frequent best search method: {best_search}.",
        f"Optimized adaptive outperforms COREX-Targeted at epsilon=0.1: {'yes' if beats_targeted else 'no'}.",
    ]
    if beats_targeted:
        lines.append("Optimized COREX-Adaptive provides the best observed privacy--utility trade-off.")
    elif improves_original:
        lines.append("Adaptive calibration improves over the original adaptive mechanism.")
    else:
        lines.append("COREX-Targeted remains the strongest empirical method, while adaptive perturbation requires further calibration.")
    return "\n".join(lines) + "\n"


def plot_outputs(updated: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = updated[updated["method"].isin(["COREX-Targeted", "COREX-Adaptive Original", "COREX-Adaptive Optimized"])].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for method_name, group in plot_df.groupby("method", observed=False):
        ax.scatter(group["attack_auc"], group["task_auc"], label=method_name, s=45)
    ax.set_xlabel("Attack AUC")
    ax.set_ylabel("Task AUC")
    ax.set_title("Adaptive Optimization Trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "adaptive_optimization_tradeoff.png", dpi=220)
    fig.savefig(fig_dir / "adaptive_optimization_tradeoff.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for method_name, group in plot_df.groupby("method", observed=False):
        group = group.sort_values("privacy_budget")
        axes[0].plot(group["privacy_budget"], group["task_auc"], marker="o", label=method_name)
        axes[1].plot(group["privacy_budget"], group["attack_auc"], marker="o", label=method_name)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Task AUC")
    axes[1].set_ylabel("Attack AUC")
    for ax in axes:
        ax.set_xlabel("Privacy Budget")
        ax.grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "adaptive_optimization_by_budget.png", dpi=220)
    fig.savefig(fig_dir / "adaptive_optimization_by_budget.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
