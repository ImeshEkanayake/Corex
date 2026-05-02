#!/usr/bin/env python3
"""Standalone Give Me Some Credit corex proxy-audit experiment.

Reproducible Give Me Some Credit experiments for the COREX paper.

The pipeline starts from raw `cs-training.csv`, constructs the age-derived
sensitive attribute, trains predictive and proxy models, builds LIME/SHAP-like
and COREX proxy rankings, and writes the paper tables plus figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ImportError:  # pragma: no cover - exercised only in minimal envs
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover - exercised only in minimal envs
    LimeTabularExplainer = None


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import corex  # noqa: E402


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


@dataclass
class PreparedData:
    raw_path: Path
    feature_names: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    a_train: pd.Series
    a_test: pd.Series


class DataFrameModelWrapper:
    """Adapter so COREX masked numpy arrays keep sklearn feature names."""

    def __init__(self, model, columns: list[str]) -> None:
        self.model = model
        self.columns = list(columns)

    def _frame(self, x) -> pd.DataFrame:
        array = np.asarray(x, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return pd.DataFrame(array, columns=self.columns)

    def predict_proba(self, x) -> np.ndarray:
        return self.model.predict_proba(self._frame(x))

    def predict(self, x) -> np.ndarray:
        return self.model.predict(self._frame(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COREX Give Me Some Credit paper experiments.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional local cs-training.csv path.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "corex" / "outputs")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--explain-sample-size", type=int, default=48)
    parser.add_argument("--corex-local-sample-size", type=int, default=16)
    parser.add_argument("--baseline-samples", type=int, default=128)
    parser.add_argument("--proxy-game-sample-size", type=int, default=256)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "rankings": output_dir / "rankings",
        "splits": output_dir / "splits",
        "models": output_dir / "models",
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


def ensure_dataset(data_path: Path | None, data_dir: Path) -> Path:
    if data_path is not None:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        return data_path
    downloaded = data_dir / "cs-training.csv"
    if not downloaded.exists():
        logging.info("Downloading Give Me Some Credit data to %s", downloaded)
        urlretrieve(DATA_URL, downloaded)
    return downloaded


def load_and_prepare(raw_path: Path, output_dirs: dict[str, Path], test_size: float, seed: int) -> PreparedData:
    raw = pd.read_csv(raw_path)
    raw = raw.rename(columns={raw.columns[0]: "id"}) if raw.columns[0].startswith("Unnamed") else raw
    for column in ["id", "ID", "Id"]:
        if column in raw.columns:
            raw = raw.drop(columns=[column])

    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    raw = raw[EXPECTED_COLUMNS].copy()
    raw[TARGET] = raw[TARGET].astype(int)
    age_median = float(raw[AGE].median())
    sensitive = (raw[AGE] >= age_median).astype(int).rename("age_ge_median")
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
    x_train_imputed = imputer.fit_transform(x_train_raw)
    x_test_imputed = imputer.transform(x_test_raw)
    x_train = pd.DataFrame(scaler.fit_transform(x_train_imputed), columns=features.columns)
    x_test = pd.DataFrame(scaler.transform(x_test_imputed), columns=features.columns)

    pd.DataFrame({"feature": list(features.columns)}).to_csv(output_dirs["data"].parent / "feature_names.csv", index=False)
    pd.DataFrame({"train_index": train_idx}).to_csv(output_dirs["splits"] / "train_indices.csv", index=False)
    pd.DataFrame({"test_index": test_idx}).to_csv(output_dirs["splits"] / "test_indices.csv", index=False)
    joblib.dump({"imputer": imputer, "scaler": scaler}, output_dirs["models"] / "preprocessing.joblib")

    return PreparedData(
        raw_path=raw_path,
        feature_names=list(features.columns),
        train_indices=train_idx,
        test_indices=test_idx,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        a_train=a_train,
        a_test=a_test,
    )


def auc(y_true: pd.Series | np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def score_model(model, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def train_predictive_models(data: PreparedData, models_dir: Path, seed: int) -> tuple[object, pd.DataFrame]:
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(data.x_train, data.y_train)
        fitted[name] = model
        train_auc = auc(data.y_train, score_model(model, data.x_train))
        test_auc = auc(data.y_test, score_model(model, data.x_test))
        rows.append({"model": name, "train_auc": train_auc, "test_auc": test_auc})
        joblib.dump(model, models_dir / f"predictive_{name}.joblib")
        logging.info("Predictive model %s test AUC %.4f", name, test_auc)
    results = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    best_name = str(results.iloc[0]["model"])
    joblib.dump(fitted[best_name], models_dir / "primary_predictive_model.joblib")
    results.to_csv(models_dir / "predictive_model_results.csv", index=False)
    return fitted[best_name], results


def train_proxy_models(data: PreparedData, models_dir: Path, seed: int) -> tuple[object, pd.DataFrame]:
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(data.x_train, data.a_train)
        fitted[name] = model
        train_auc = auc(data.a_train, score_model(model, data.x_train))
        test_auc = auc(data.a_test, score_model(model, data.x_test))
        rows.append({"model": name, "train_auc": train_auc, "test_auc": test_auc})
        joblib.dump(model, models_dir / f"proxy_{name}.joblib")
        logging.info("Proxy model %s test AUC %.4f", name, test_auc)
    results = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    best_name = str(results.iloc[0]["model"])
    joblib.dump(fitted[best_name], models_dir / "primary_proxy_model.joblib")
    results.to_csv(models_dir / "proxy_model_results.csv", index=False)
    return fitted[best_name], results


def make_sample_indices(n_rows: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n_rows), size=min(size, n_rows), replace=False))


def local_predictor(model, columns: list[str]) -> Callable[[list[float]], float]:
    def predict(row: list[float]) -> float:
        flat = np.asarray(row, dtype=float).reshape(-1)
        return float(score_model(model, pd.DataFrame([flat], columns=columns))[0])

    return predict


def lime_attributions(model, x_sample: pd.DataFrame, x_train: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if LimeTabularExplainer is not None:
        logging.info("Using lime.lime_tabular.LimeTabularExplainer")
        explainer = LimeTabularExplainer(
            training_data=x_train.to_numpy(dtype=float),
            feature_names=list(x_train.columns),
            class_names=["non_default", "default"],
            mode="classification",
            discretize_continuous=False,
            random_state=seed,
        )
        weights = []
        feature_names = list(x_train.columns)
        feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
        for _, row in x_sample.iterrows():
            explanation = explainer.explain_instance(
                row.to_numpy(dtype=float),
                model.predict_proba,
                num_features=len(feature_names),
                num_samples=n_samples,
                labels=(1,),
            )
            row_weights = np.zeros(len(feature_names), dtype=float)
            for feature, value in explanation.as_list(label=1):
                if feature in feature_to_index:
                    row_weights[feature_to_index[feature]] = float(value)
            weights.append(row_weights)
        return ranking_from_values(feature_names, np.vstack(weights), "lime_weight")

    logging.info("lime package unavailable; using local weighted-ridge fallback")
    rng = np.random.default_rng(seed)
    columns = list(x_sample.columns)
    scale = x_train.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=float)
    rows = []
    for _, row in x_sample.iterrows():
        center = row.to_numpy(dtype=float)
        perturb = center + rng.normal(0.0, scale, size=(n_samples, len(columns)))
        perturb[0, :] = center
        distances = np.linalg.norm((perturb - center) / scale, axis=1)
        kernel_width = math.sqrt(len(columns)) * 0.75
        weights = np.exp(-(distances**2) / max(kernel_width**2, 1e-8))
        scores = score_model(model, pd.DataFrame(perturb, columns=columns))
        ridge = Ridge(alpha=1e-3)
        ridge.fit(perturb - center, scores, sample_weight=weights)
        rows.append(np.asarray(ridge.coef_, dtype=float))
    values = np.vstack(rows)
    return ranking_from_values(columns, values, "lime_weight")


def kernel_shap_values(model, x_sample: pd.DataFrame, background: np.ndarray, n_samples: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    columns = list(x_sample.columns)
    all_values = []
    for _, row in x_sample.iterrows():
        instance = row.to_numpy(dtype=float)
        masks = rng.binomial(1, 0.5, size=(n_samples, len(columns))).astype(float)
        masks[0, :] = 0.0
        masks[1, :] = 1.0
        y_local = []
        weights = []
        for mask in masks:
            sample = background + mask * (instance - background)
            y_local.append(float(score_model(model, pd.DataFrame([sample], columns=columns))[0]))
            size = int(mask.sum())
            if size in {0, len(columns)}:
                weights.append(1000.0)
            else:
                weights.append((len(columns) - 1) / (math.comb(len(columns), size) * size * (len(columns) - size)))
        ridge = Ridge(alpha=1e-6)
        ridge.fit(masks, np.asarray(y_local), sample_weight=np.asarray(weights))
        all_values.append(np.asarray(ridge.coef_, dtype=float))
    values = np.vstack(all_values)
    return values, ranking_from_values(columns, values, "shap_value")


def package_shap_values(model, x_sample: pd.DataFrame, background: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, object]:
    if shap is None:
        raise ImportError("shap is not installed.")
    logging.info("Using shap.TreeExplainer for SHAP values and beeswarm")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(x_sample)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
    shap_explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=x_sample.to_numpy(dtype=float),
        feature_names=list(x_sample.columns),
    )
    return values, ranking_from_values(list(x_sample.columns), values, "shap_value"), shap_explanation


def ranking_from_values(feature_names: list[str], values: np.ndarray, score_name: str) -> pd.DataFrame:
    scores = np.mean(np.abs(values), axis=0)
    ranking = pd.DataFrame({"feature": feature_names, score_name: scores})
    ranking["rank"] = ranking[score_name].rank(ascending=False, method="first").astype(int)
    return ranking.sort_values("rank").reset_index(drop=True)


def save_package_shap_beeswarm(explanation, out_png: Path, out_pdf: Path) -> None:
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names), show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def save_fallback_beeswarm(values: np.ndarray, x_sample: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    order = np.argsort(np.mean(np.abs(values), axis=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(SEED)
    for y_pos, feature_idx in enumerate(order):
        jitter = rng.normal(0, 0.08, size=values.shape[0])
        feature_values = x_sample.iloc[:, feature_idx].to_numpy(dtype=float)
        scatter = ax.scatter(
            values[:, feature_idx],
            y_pos + jitter,
            c=feature_values,
            cmap="viridis",
            s=16,
            alpha=0.75,
            edgecolors="none",
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([x_sample.columns[index] for index in order])
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Beeswarm")
    fig.colorbar(scatter, ax=ax, label="standardized feature value")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def corex_proxy_rankings(proxy_model, data: PreparedData, sample_indices: np.ndarray, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, float], dict[str, float], object]:
    x_proxy = data.x_test.iloc[sample_indices].reset_index(drop=True)
    a_proxy = data.a_test.iloc[sample_indices].reset_index(drop=True)
    scorer = corex.GameTheoreticScorer(concepts=["banzhaf", "least_core", "nucleolus"], sampling_budget=None)
    auditor = corex.ProxyAuditor(
        DataFrameModelWrapper(proxy_model, data.feature_names),
        background=data.x_train.mean(axis=0).to_numpy(dtype=float),
        feature_names=data.feature_names,
        prediction_method="predict_proba",
        target_class=1,
        scorer=scorer,
    )
    report = auditor.audit_dataset(
        dataset=x_proxy.to_numpy(dtype=float),
        feature_names=data.feature_names,
        sensitive_attribute_series={"age_ge_median": a_proxy.to_numpy(dtype=float)},
        evaluation_mode="proxy_model",
        coalition_size_range=(1, min(3, len(data.feature_names))),
        top_k=12,
        max_features=len(data.feature_names),
        random_seed=seed,
    )
    result = report.attribute_reports["age_ge_median"]
    rankings = {}
    eps = {}
    lfr = {}
    risk_scores = {}
    solution_names = {
        "corex_banzhaf": "banzhaf",
        "corex_least_core": "least_core",
        "corex_nucleolus": "nucleolus",
    }
    for out_name, concept in solution_names.items():
        solution = result.solution_concepts[concept]
        values = np.asarray(solution.allocations, dtype=float).reshape(1, -1)
        rankings[out_name] = ranking_from_values(data.feature_names, values, concept)
        eps[out_name] = float(abs(result.solution_concepts["least_core"].epsilon or 0.0))
        risk_scores[out_name] = dict(zip(data.feature_names, np.abs(solution.allocations).astype(float)))
        lfr[out_name] = mean_local_fairness_risk(proxy_model, data, sample_indices[: min(16, len(sample_indices))], solution.allocations)
    return rankings, eps, lfr, risk_scores["corex_banzhaf"], report


def corex_predictive_rankings(primary_model, data: PreparedData, sample_indices: np.ndarray) -> dict[str, pd.DataFrame]:
    x_pred = data.x_test.iloc[sample_indices[: min(8, len(sample_indices))]].reset_index(drop=True)
    scorer = corex.GameTheoreticScorer(concepts=["banzhaf", "least_core", "nucleolus"], sampling_budget=None)
    values = {"banzhaf": [], "least_core": [], "nucleolus": []}
    for _, row in x_pred.iterrows():
        explainer = corex.KernelExplainer(
            local_predictor(primary_model, data.feature_names),
            background=data.x_train.mean(axis=0).to_numpy(dtype=float),
            feature_names=data.feature_names,
            scorer=scorer,
        )
        result = explainer.explain(row.to_numpy(dtype=float).tolist(), feature_names=data.feature_names)
        for concept in values:
            values[concept].append(np.asarray(result.solution_concepts[concept].allocations, dtype=float))
    return {
        f"predictive_{concept}": ranking_from_values(data.feature_names, np.vstack(concept_values), concept)
        for concept, concept_values in values.items()
    }


def mean_local_fairness_risk(proxy_model, data: PreparedData, sample_indices: np.ndarray, allocation: np.ndarray) -> float:
    predictor = local_predictor(proxy_model, data.feature_names)
    baseline = data.x_train.mean(axis=0).to_numpy(dtype=float)
    allocation = np.abs(np.asarray(allocation, dtype=float))
    risks = []
    for idx in sample_indices:
        row = data.x_test.iloc[int(idx)].to_numpy(dtype=float)
        full = predictor(row.tolist())
        empty = predictor(baseline.tolist())
        total_shift = abs(full - empty) + 1e-12
        risks.append(float(np.sum(allocation) / total_shift))
    return float(np.mean(risks))


def save_corex_coalition_plot(report, output_png: Path, output_pdf: Path) -> None:
    """Save the native COREX proxy-overlap plot used for proxy selection."""
    fig = report.plot_proxy_overlaps(
        top_k=4,
        dataset_name="Give Me Some Credit",
        target_description="age_ge_median",
    )
    fig.write_html(output_png.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(output_png, scale=2)
    fig.write_image(output_pdf)


def train_proxy_auc_for_features(data: PreparedData, features: list[str], seed: int) -> float:
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(data.x_train[features], data.a_train)
    return auc(data.a_test, score_model(model, data.x_test[features]))


def latex_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = []
    for _, row in df.iterrows():
        values = [str(row["method"])]
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(value)
            else:
                values.append("N/A" if pd.isna(value) else f"{float(value):.3f}")
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines) + "\n"


def build_table1(rankings: dict[str, pd.DataFrame], eps: dict[str, float | str], lfr: dict[str, float | str], data: PreparedData, tables_dir: Path, seed: int) -> pd.DataFrame:
    method_map = {
        "LIME": "lime",
        "SHAP": "shap",
        "COREX-Banzhaf": "corex_banzhaf",
        "COREX-Least-Core": "corex_least_core",
        "COREX-Nucleolus": "corex_nucleolus",
    }
    rows = []
    for method, key in method_map.items():
        ranking = rankings[key]
        row = {"method": method, "epsilon_star": eps.get(key, "N/A"), "lfr": lfr.get(key, "N/A")}
        for k in [3, 5, 7]:
            features = ranking.head(k)["feature"].tolist()
            row[f"top{k}_proxy_auc"] = train_proxy_auc_for_features(data, features, seed + k)
            row[f"top{k}_features"] = ";".join(features)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "table1_proxy_detection.csv", index=False)
    latex = latex_table(table, ["top3_proxy_auc", "top5_proxy_auc", "top7_proxy_auc", "epsilon_star", "lfr"])
    (tables_dir / "table1_proxy_detection_latex.txt").write_text(latex, encoding="utf-8")
    return table


def perturb(x: pd.DataFrame, features: list[str], scale: float, rng: np.random.Generator, weights: dict[str, float] | None = None) -> pd.DataFrame:
    out = x.copy()
    if not features:
        return out
    std = x[features].std(axis=0).replace(0.0, 1.0)
    for feature in features:
        weight = 1.0 if weights is None else float(weights.get(feature, 1.0))
        out[feature] = out[feature] + rng.normal(0.0, scale * weight * float(std[feature]), size=len(out))
    return out


def fairness_gaps(y_true: pd.Series, sensitive: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    prediction = (scores >= 0.5).astype(int)
    a = sensitive.to_numpy(dtype=int)
    y = y_true.to_numpy(dtype=int)
    pos_rates = []
    tprs = []
    for group in [0, 1]:
        group_mask = a == group
        pos_rates.append(float(prediction[group_mask].mean()) if group_mask.any() else 0.0)
        positive_mask = group_mask & (y == 1)
        tprs.append(float(prediction[positive_mask].mean()) if positive_mask.any() else 0.0)
    return float(abs(pos_rates[1] - pos_rates[0])), float(abs(tprs[1] - tprs[0]))


def build_table2(rankings: dict[str, pd.DataFrame], corex_risk: dict[str, float], data: PreparedData, tables_dir: Path, seed: int, target_k: int, noise_scale: float) -> pd.DataFrame:
    top_lime = rankings["lime"].head(target_k)["feature"].tolist()
    top_shap = rankings["shap"].head(target_k)["feature"].tolist()
    top_corex = rankings["corex_banzhaf"].head(target_k)["feature"].tolist()
    max_risk = max(corex_risk.values()) if corex_risk else 1.0
    adaptive_weights = {feature: 0.5 + 1.5 * (corex_risk.get(feature, 0.0) / max(max_risk, 1e-12)) for feature in top_corex}
    interventions = {
        "No Privacy": {"features": [], "scale": 0.0, "weights": None},
        "Uniform Perturbation": {"features": data.feature_names, "scale": noise_scale, "weights": None},
        "LIME-Targeted": {"features": top_lime, "scale": noise_scale, "weights": None},
        "SHAP-Targeted": {"features": top_shap, "scale": noise_scale, "weights": None},
        "COREX-Targeted": {"features": top_corex, "scale": noise_scale, "weights": None},
        "COREX-Adaptive": {"features": top_corex, "scale": noise_scale, "weights": adaptive_weights},
    }
    rows = []
    for offset, (method, config) in enumerate(interventions.items()):
        rng = np.random.default_rng(seed + 10_000 + offset)
        x_train = perturb(data.x_train, config["features"], config["scale"], rng, config["weights"])
        x_test = perturb(data.x_test, config["features"], config["scale"], rng, config["weights"])
        pred_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed + offset)
        pred_model.fit(x_train, data.y_train)
        task_scores = score_model(pred_model, x_test)
        attack = LogisticRegression(max_iter=2000, random_state=seed + offset)
        attack.fit(x_train, data.a_train)
        attack_scores = score_model(attack, x_test)
        dp_gap, eo_gap = fairness_gaps(data.y_test, data.a_test, task_scores)
        rows.append(
            {
                "method": method,
                "task_auc": auc(data.y_test, task_scores),
                "attack_auc": auc(data.a_test, attack_scores),
                "dp_gap": dp_gap,
                "eo_gap": eo_gap,
                "perturbed_features": ";".join(config["features"]) if config["features"] else "none",
                "noise_config": json.dumps({"scale": config["scale"], "weights": config["weights"]}, sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "table2_privacy_utility.csv", index=False)
    latex = latex_table(table, ["task_auc", "attack_auc", "dp_gap", "eo_gap"])
    (tables_dir / "table2_privacy_utility_latex.txt").write_text(latex, encoding="utf-8")
    return table


def save_ranking(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# COREX Give Me Some Credit Experiment\n", "Runs the script version end-to-end and leaves outputs under `../outputs`.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%run ../scripts/run_corex_experiments.py\n"],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dirs = make_output_dirs(args.output_dir)
    setup_logging(output_dirs["logs"] / "experiment.log")
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(
        "%s:%s: %s: %s", filename, lineno, category.__name__, message
    )
    np.random.seed(args.seed)
    logging.info("Starting Give Me Some Credit COREX experiment")

    config = vars(args).copy()
    config["data_url"] = DATA_URL
    config["explainers"] = {
        "lime": "lime.lime_tabular when installed; otherwise local weighted ridge perturbation explainer",
        "shap": "shap.TreeExplainer when installed; otherwise KernelSHAP-style weighted coalition regression",
    }
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    write_notebook(EXPERIMENT_DIR / "notebooks" / "corex_experiment_give_me_some_credit.ipynb")

    raw_path = ensure_dataset(args.data_path, output_dirs["data"])
    data = load_and_prepare(raw_path, output_dirs, args.test_size, args.seed)
    primary_model, predictive_results = train_predictive_models(data, output_dirs["models"], args.seed)
    proxy_model, proxy_results = train_proxy_models(data, output_dirs["models"], args.seed)

    sample_indices = make_sample_indices(len(data.x_test), args.explain_sample_size, args.seed)
    x_sample = data.x_test.iloc[sample_indices].reset_index(drop=True)
    logging.info("Computing LIME rankings")
    lime_ranking = lime_attributions(primary_model, x_sample, data.x_train, args.baseline_samples, args.seed)
    logging.info("Computing SHAP rankings and beeswarm")
    if shap is not None:
        shap_values, shap_ranking, shap_explanation = package_shap_values(primary_model, x_sample, data.x_train)
        save_package_shap_beeswarm(shap_explanation, output_dirs["figures"] / "shap_beeswarm.png", output_dirs["figures"] / "shap_beeswarm.pdf")
    else:
        shap_values, shap_ranking = kernel_shap_values(primary_model, x_sample, data.x_train.mean(axis=0).to_numpy(dtype=float), args.baseline_samples, args.seed)
        save_fallback_beeswarm(shap_values, x_sample, output_dirs["figures"] / "shap_beeswarm.png", output_dirs["figures"] / "shap_beeswarm.pdf")

    logging.info("Computing COREX proxy-game rankings")
    proxy_sample_indices = make_sample_indices(len(data.x_test), args.proxy_game_sample_size, args.seed + 1)
    corex_rankings, eps, lfr, corex_risk, proxy_report = corex_proxy_rankings(proxy_model, data, proxy_sample_indices, args.seed)
    save_corex_coalition_plot(proxy_report, output_dirs["figures"] / "corex_proxy_coalition.png", output_dirs["figures"] / "corex_proxy_coalition.pdf")
    logging.info("Computing COREX predictive-game rankings")
    predictive_corex_rankings = corex_predictive_rankings(primary_model, data, sample_indices)

    rankings = {
        "lime": lime_ranking,
        "shap": shap_ranking,
        **corex_rankings,
    }
    save_ranking(lime_ranking, output_dirs["rankings"] / "lime_ranking.csv")
    save_ranking(shap_ranking, output_dirs["rankings"] / "shap_ranking.csv")
    save_ranking(corex_rankings["corex_banzhaf"], output_dirs["rankings"] / "corex_banzhaf_ranking.csv")
    save_ranking(corex_rankings["corex_least_core"], output_dirs["rankings"] / "corex_least_core_ranking.csv")
    save_ranking(corex_rankings["corex_nucleolus"], output_dirs["rankings"] / "corex_nucleolus_ranking.csv")
    for name, ranking in predictive_corex_rankings.items():
        save_ranking(ranking, output_dirs["rankings"] / f"{name}_ranking.csv")

    eps_with_baselines = {"lime": "N/A", "shap": "N/A", **eps}
    lfr_with_baselines = {"lime": "N/A", "shap": "N/A", **lfr}
    table1 = build_table1(rankings, eps_with_baselines, lfr_with_baselines, data, output_dirs["tables"], args.seed)
    table2 = build_table2(rankings, corex_risk, data, output_dirs["tables"], args.seed, args.target_k, args.noise_scale)
    logging.info("Table 1:\n%s", table1.to_string(index=False))
    logging.info("Table 2:\n%s", table2.to_string(index=False))
    logging.info("Finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
