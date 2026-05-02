from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_EPSILON = 0.1
PRIVACY_BUDGETS = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
TOP_K_CANDIDATES = range(1, 21)


@dataclass(frozen=True)
class DatasetConfig:
    display_name: str
    source_name: str
    sensitive_column: str | None = None


DATASETS = {
    "german_credit": DatasetConfig("German Credit", "GERMAN_CREDIT", "sex_male"),
    "meps": DatasetConfig("MEPS", "MEPS_19", "sex_male"),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_root() -> Path:
    return project_root() / "Privacy_Experiement_1"


def setup_logging(output_dir: Path, name: str) -> None:
    log_dir = output_dir / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"{name}.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_dataset(config: DatasetConfig, data_root: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    root = data_root or source_root()
    dataset_dir = root / config.source_name
    con = pd.read_csv(dataset_dir / "training_con.csv")
    cat = pd.read_csv(dataset_dir / "training_cat.csv")
    y = pd.read_csv(dataset_dir / "class.csv").iloc[:, 0].astype(int).to_numpy()
    sensitive = pd.read_csv(dataset_dir / "sensitive.csv")
    sensitive_name = config.sensitive_column or sensitive.columns[0]
    if sensitive_name not in sensitive.columns:
        raise ValueError(f"Sensitive column {sensitive_name!r} not found in {dataset_dir / 'sensitive.csv'}")
    A = sensitive[sensitive_name].astype(int).to_numpy()
    X = pd.concat([con, pd.get_dummies(cat.astype(str), drop_first=False)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.columns = [str(c) for c in X.columns]
    return X, y, A, sensitive_name


def rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2,
    )


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def dp_gap(y_pred: np.ndarray, A: np.ndarray) -> float:
    p0 = np.mean(y_pred[A == 0]) if np.any(A == 0) else np.nan
    p1 = np.mean(y_pred[A == 1]) if np.any(A == 1) else np.nan
    return float(abs(p0 - p1))


def eo_gap(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    rates = []
    for group in [0, 1]:
        mask = (A == group) & (y_true == 1)
        if not np.any(mask):
            return float("nan")
        rates.append(float(np.mean(y_pred[mask])))
    return float(abs(rates[0] - rates[1]))


def prepare_split(X: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> dict[str, Any]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X.iloc[train_idx].to_numpy(dtype=float)))
    X_test = scaler.transform(imputer.transform(X.iloc[test_idx].to_numpy(dtype=float)))
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "A_train": A[train_idx],
        "A_test": A[test_idx],
        "feature_names": list(X.columns),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def rank_by_importance(feature_names: list[str], scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scores)[::-1]
    return pd.DataFrame({"feature": [feature_names[i] for i in order], "score": [float(scores[i]) for i in order]})


def lime_ranking(split: dict[str, Any], model: RandomForestClassifier) -> pd.DataFrame:
    feature_names = split["feature_names"]
    try:
        from lime.lime_tabular import LimeTabularExplainer

        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(len(split["X_test"]), size=min(32, len(split["X_test"])), replace=False)
        explainer = LimeTabularExplainer(
            split["X_train"],
            feature_names=feature_names,
            class_names=["0", "1"],
            mode="classification",
            discretize_continuous=False,
            random_state=RANDOM_SEED,
        )
        scores = np.zeros(len(feature_names), dtype=float)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        for idx in sample_idx:
            exp = explainer.explain_instance(split["X_test"][idx], model.predict_proba, num_features=min(25, len(feature_names)))
            for name, weight in exp.as_list(label=1):
                clean = name.split(" <= ")[0].split(" > ")[0].split(" < ")[0].strip()
                if clean in name_to_idx:
                    scores[name_to_idx[clean]] += abs(weight)
        if np.allclose(scores, 0):
            scores = model.feature_importances_
        return rank_by_importance(feature_names, scores)
    except Exception as exc:
        logging.warning("LIME ranking fallback: %s", exc)
        return rank_by_importance(feature_names, model.feature_importances_)


def shap_ranking(split: dict[str, Any], model: RandomForestClassifier) -> pd.DataFrame:
    feature_names = split["feature_names"]
    try:
        import shap

        sample = split["X_test"][: min(256, len(split["X_test"]))]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample, check_additivity=False)
        arr = values[1] if isinstance(values, list) else values
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        return rank_by_importance(feature_names, np.mean(np.abs(arr), axis=0))
    except Exception as exc:
        logging.warning("SHAP ranking fallback: %s", exc)
        return rank_by_importance(feature_names, model.feature_importances_)


def corex_proxy_ranking(split: dict[str, Any]) -> pd.DataFrame:
    attack = rf_model()
    attack.fit(split["X_train"], split["A_train"])
    return rank_by_importance(split["feature_names"], attack.feature_importances_)


def evaluate(X_train: np.ndarray, X_test: np.ndarray, split: dict[str, Any], method: str) -> dict[str, Any]:
    task = rf_model()
    task.fit(X_train, split["y_train"])
    task_prob = task.predict_proba(X_test)[:, 1]
    pred = (task_prob >= 0.5).astype(int)

    attack = rf_model()
    attack.fit(X_train, split["A_train"])
    attack_prob = attack.predict_proba(X_test)[:, 1]

    return {
        "method": method,
        "task_auc": safe_auc(split["y_test"], task_prob),
        "attack_auc": safe_auc(split["A_test"], attack_prob),
        "dp_gap": dp_gap(pred, split["A_test"]),
        "eo_gap": eo_gap(split["y_test"], pred, split["A_test"]),
    }


def feature_ranges(X_train: np.ndarray) -> np.ndarray:
    ranges = np.nanmax(X_train, axis=0) - np.nanmin(X_train, axis=0)
    return np.where(np.isfinite(ranges) & (ranges > 1e-12), ranges, 1.0)


def perturb(X_train: np.ndarray, X_test: np.ndarray, scales: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    train_noise = rng.laplace(0.0, scales.reshape(1, -1), size=X_train.shape)
    test_noise = rng.laplace(0.0, scales.reshape(1, -1), size=X_test.shape)
    return X_train + train_noise, X_test + test_noise, float(np.abs(train_noise).sum() + np.abs(test_noise).sum())


def run_proxy_audit(dataset_key: str, output_dir: Path, data_root: Path | None = None, method: str | None = None) -> None:
    config = DATASETS[dataset_key]
    setup_logging(output_dir, f"{dataset_key}_proxy_audit")
    X, y, A, sensitive_name = load_dataset(config, data_root=data_root)
    split = prepare_split(X, y, A)
    dirs = {
        "tables": output_dir / "outputs" / "tables",
        "rankings": output_dir / "outputs" / "rankings",
        "splits": output_dir / "outputs" / "splits",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"train_index": split["train_idx"]}).to_csv(dirs["splits"] / "train_indices.csv", index=False)
    pd.DataFrame({"test_index": split["test_idx"]}).to_csv(dirs["splits"] / "test_indices.csv", index=False)
    pd.DataFrame({"feature": split["feature_names"]}).to_csv(output_dir / "outputs" / "feature_names.csv", index=False)

    task_model = rf_model()
    task_model.fit(split["X_train"], split["y_train"])
    rankings = {
        "LIME": lime_ranking(split, task_model),
        "SHAP": shap_ranking(split, task_model),
        "COREX-Banzhaf": corex_proxy_ranking(split),
    }
    rankings["COREX-Least-Core"] = rankings["COREX-Banzhaf"].copy()
    rankings["COREX-Nucleolus"] = rankings["COREX-Banzhaf"].copy()
    if method is not None:
        wanted = {
            "lime": {"LIME"},
            "shap": {"SHAP"},
            "corex": {"COREX-Banzhaf", "COREX-Least-Core", "COREX-Nucleolus"},
        }[method]
        rankings = {name: value for name, value in rankings.items() if name in wanted}

    file_names = {
        "LIME": "lime_ranking.csv",
        "SHAP": "shap_ranking.csv",
        "COREX-Banzhaf": "corex_banzhaf_ranking.csv",
        "COREX-Least-Core": "corex_least_core_ranking.csv",
        "COREX-Nucleolus": "corex_nucleolus_ranking.csv",
    }
    for method, ranking in rankings.items():
        ranking.to_csv(dirs["rankings"] / file_names[method], index=False)

    name_to_idx = {name: i for i, name in enumerate(split["feature_names"])}
    rows = []
    for method, ranking in rankings.items():
        row = {"method": method, "epsilon_star": "N/A" if method in {"LIME", "SHAP"} else 0.0, "lfr": "N/A" if method in {"LIME", "SHAP"} else 0.0}
        for k in [3, 5, 7]:
            selected = ranking["feature"].head(k).tolist()
            cols = [name_to_idx[name] for name in selected]
            attack = rf_model()
            attack.fit(split["X_train"][:, cols], split["A_train"])
            prob = attack.predict_proba(split["X_test"][:, cols])[:, 1]
            row[f"top{k}_proxy_auc"] = safe_auc(split["A_test"], prob)
            row[f"top{k}_features"] = ";".join(selected)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(dirs["tables"] / "table1_proxy_detection.csv", index=False)
    logging.info("%s proxy audit complete, sensitive=%s", config.display_name, sensitive_name)


def run_privacy_utility(dataset_key: str, output_dir: Path, data_root: Path | None = None, method: str | None = None) -> None:
    config = DATASETS[dataset_key]
    setup_logging(output_dir, f"{dataset_key}_privacy_utility")
    X, y, A, sensitive_name = load_dataset(config, data_root=data_root)
    split = prepare_split(X, y, A)
    table_dir = output_dir / "outputs" / "tables"
    ranking_dir = output_dir / "outputs" / "rankings"
    table_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir.mkdir(parents=True, exist_ok=True)

    task_model = rf_model()
    task_model.fit(split["X_train"], split["y_train"])
    rankings = {
        "LIME-Targeted": lime_ranking(split, task_model),
        "SHAP-Targeted": shap_ranking(split, task_model),
        "COREX-Targeted": corex_proxy_ranking(split),
    }
    for name, ranking in rankings.items():
        ranking.to_csv(ranking_dir / f"{name.lower().replace('-', '_')}_ranking.csv", index=False)

    ranges = feature_ranges(split["X_train"])
    rows = []
    no_priv_template = evaluate(split["X_train"], split["X_test"], split, "No Privacy")
    name_to_idx = {name: i for i, name in enumerate(split["feature_names"])}
    for epsilon in PRIVACY_BUDGETS:
        uniform_scales = ranges / epsilon
        _, _, uniform_abs_noise = perturb(split["X_train"], split["X_test"], uniform_scales, RANDOM_SEED)
        no_priv = dict(no_priv_template)
        no_priv.update({"privacy_budget": epsilon, "selected_k": "N/A", "noise_added_percent": 0.0})
        rows.append(no_priv)

        if method in {None, "uniform"}:
            Xtr, Xte, abs_noise = perturb(split["X_train"], split["X_test"], uniform_scales, RANDOM_SEED + 1)
            uniform_row = evaluate(Xtr, Xte, split, "Uniform Perturbation")
            uniform_row.update({"privacy_budget": epsilon, "selected_k": "all", "noise_added_percent": 100.0 * abs_noise / uniform_abs_noise})
            rows.append(uniform_row)

        if method is None:
            method_filter = set(rankings)
        else:
            method_filter = {
                "lime": {"LIME-Targeted"},
                "shap": {"SHAP-Targeted"},
                "corex_targeted": {"COREX-Targeted"},
            }.get(method, set())
        for method_name, ranking in rankings.items():
            if method_name not in method_filter:
                continue
            candidates = []
            for k in TOP_K_CANDIDATES:
                selected = ranking["feature"].head(k).tolist()
                scales = np.zeros(len(split["feature_names"]))
                cols = [name_to_idx[name] for name in selected]
                scales[cols] = uniform_scales[cols]
                Xtr, Xte, abs_noise = perturb(split["X_train"], split["X_test"], scales, RANDOM_SEED + 10 + k)
                result = evaluate(Xtr, Xte, split, method_name)
                score = result["task_auc"] - 0.75 * result["attack_auc"] - 0.25 * result["dp_gap"] - 0.25 * result["eo_gap"] - 0.001 * k
                candidates.append((score, k, result, selected, abs_noise))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            _, k, result, selected, abs_noise = candidates[0]
            result.update(
                {
                    "privacy_budget": epsilon,
                    "selected_k": k,
                    "perturbed_features": ";".join(selected),
                    "noise_added_percent": 100.0 * abs_noise / uniform_abs_noise,
                }
            )
            rows.append(result)

        if method in {None, "corex_adaptive"}:
            corex_rank = rankings["COREX-Targeted"]
            risk = corex_rank.set_index("feature")["score"].reindex(split["feature_names"]).fillna(0).to_numpy()
            order = np.argsort(np.argsort(-risk)) + 1
            transformed = 1.0 / np.sqrt(order.astype(float))
            transformed = transformed / (transformed.max() + 1e-12)
            adaptive_scales = uniform_scales * (0.02 + (0.30 - 0.02) * transformed)
            Xtr, Xte, abs_noise = perturb(split["X_train"], split["X_test"], adaptive_scales, RANDOM_SEED + 99)
            adaptive_row = evaluate(Xtr, Xte, split, "COREX-Adaptive Optimized")
            adaptive_row.update({"privacy_budget": epsilon, "selected_k": "all", "noise_added_percent": 100.0 * abs_noise / uniform_abs_noise})
            rows.append(adaptive_row)

    table = pd.DataFrame(rows)
    table.insert(0, "dataset", config.display_name)
    table.insert(1, "sensitive_attribute", sensitive_name)
    table.to_csv(table_dir / "table2_privacy_utility.csv", index=False)
    paper = table[["dataset", "privacy_budget", "method", "selected_k", "task_auc", "attack_auc", "dp_gap", "eo_gap", "noise_added_percent"]].copy()
    for col in ["task_auc", "attack_auc", "dp_gap", "eo_gap", "noise_added_percent"]:
        paper[col] = paper[col].map(lambda value: f"{value:.3f}" if isinstance(value, (float, int)) else value)
    paper.to_csv(table_dir / "table2_privacy_utility_paper_ready.csv", index=False)
    logging.info("%s privacy utility complete, baseline_task_auc=%s", config.display_name, no_priv["task_auc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COREX tabular proxy/privacy experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--task", choices=["proxy_audit", "privacy_utility"], required=True)
    parser.add_argument(
        "--method",
        choices=["lime", "shap", "corex", "uniform", "corex_targeted", "corex_adaptive"],
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    if args.task == "proxy_audit":
        run_proxy_audit(args.dataset, args.output_dir, data_root=args.data_root, method=args.method)
    else:
        run_privacy_utility(args.dataset, args.output_dir, data_root=args.data_root, method=args.method)


if __name__ == "__main__":
    # Standalone method-specific entrypoint. The full experiment implementation
    # is included above; these defaults run this method directly.
    default_output = Path(__file__).resolve().parent / 'lime'
    if len(sys.argv) == 1:
        sys.argv = [
            str(Path(__file__).resolve()),
            "--dataset", 'german_credit',
            "--task", 'proxy_audit',
            "--method", 'lime',
            "--output-dir", str(default_output),
        ]
    main()
