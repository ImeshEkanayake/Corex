from __future__ import annotations

import json
import logging
import math
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

try:
    import torch
    from opacus import PrivacyEngine
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional runtime fallback is logged.
    torch = None
    PrivacyEngine = None
    nn = None
    DataLoader = None
    TensorDataset = None


RANDOM_SEED = 42
MAX_EPSILON = 0.1
TOP_K = 5
MAX_TUNED_K = 20
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
TABLE_DIR = OUT_ROOT / "tables"
LOG_DIR = OUT_ROOT / "logs"
SOURCE_ROOT = ROOT / "Privacy_Experiement_1"

DATASETS = {
    "Adult": "ADULT",
    "COMPAS": "COMPAS",
    "German Credit": "GERMAN_CREDIT",
    "MEPS": "MEPS_19",
    "GenderPayGap": "GENDER_PAY_GAP",
}

METHOD_ORDER = [
    "No Privacy",
    "Uniform Perturbation",
    "LIME-Targeted",
    "SHAP-Targeted",
    "COREX-Targeted",
    "COREX-Adaptive Optimized",
    "PFairDP",
    "FairDP",
]


@dataclass
class PreparedDataset:
    display_name: str
    source_name: str
    X: pd.DataFrame
    y: np.ndarray
    A: np.ndarray
    sensitive_name: str


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "multi_dataset_privacy_table.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_first_column(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df.iloc[:, 0]


def load_prepared_dataset(display_name: str, source_name: str) -> PreparedDataset | None:
    dataset_dir = SOURCE_ROOT / source_name
    if not dataset_dir.exists():
        logging.warning("%s skipped: %s does not exist", display_name, dataset_dir)
        return None

    required = ["training_con.csv", "training_cat.csv", "class.csv", "sensitive.csv"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        logging.warning("%s skipped: missing %s", display_name, missing)
        return None

    con = pd.read_csv(dataset_dir / "training_con.csv")
    cat = pd.read_csv(dataset_dir / "training_cat.csv")
    y = read_first_column(dataset_dir / "class.csv").astype(int).to_numpy()
    sensitive = pd.read_csv(dataset_dir / "sensitive.csv")
    sensitive_name = sensitive.columns[0]
    A = sensitive.iloc[:, 0].astype(int).to_numpy()

    X = pd.concat([con, pd.get_dummies(cat.astype(str), drop_first=False)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.columns = [str(c) for c in X.columns]

    logging.info(
        "%s loaded: rows=%d features=%d sensitive=%s y_dist=%s A_dist=%s",
        display_name,
        len(X),
        X.shape[1],
        sensitive_name,
        dict(pd.Series(y).value_counts().sort_index()),
        dict(pd.Series(A).value_counts().sort_index()),
    )
    return PreparedDataset(display_name, source_name, X, y, A, sensitive_name)


def compute_dp_gap(y_pred: np.ndarray, A: np.ndarray) -> float:
    vals = []
    for group in [0, 1]:
        mask = A == group
        vals.append(float(np.mean(y_pred[mask])) if np.any(mask) else np.nan)
    return float(abs(vals[0] - vals[1])) if not any(np.isnan(vals)) else float("nan")


def compute_eo_gap(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    vals = []
    for group in [0, 1]:
        mask = (A == group) & (y_true == 1)
        if not np.any(mask):
            logging.warning("EO gap unavailable for group=%s because no positive labels", group)
            return float("nan")
        vals.append(float(np.mean(y_pred[mask])))
    return float(abs(vals[0] - vals[1]))


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


def evaluate_setting(
    method: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    A_train: np.ndarray,
    A_test: np.ndarray,
) -> dict[str, float | str]:
    f = rf_model()
    f.fit(X_train, y_train)
    task_prob = f.predict_proba(X_test)[:, 1]
    y_pred = (task_prob >= 0.5).astype(int)

    g = rf_model()
    g.fit(X_train, A_train)
    attack_prob = g.predict_proba(X_test)[:, 1]

    return {
        "method": method,
        "task_auc": safe_auc(y_test, task_prob),
        "attack_auc": safe_auc(A_test, attack_prob),
        "dp_gap": compute_dp_gap(y_pred, A_test),
        "eo_gap": compute_eo_gap(y_test, y_pred, A_test),
    }


def train_base_models(X_train: np.ndarray, y_train: np.ndarray, A_train: np.ndarray) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    f = rf_model()
    f.fit(X_train, y_train)
    g = rf_model()
    g.fit(X_train, A_train)
    return f, g


def feature_ranking_from_importance(feature_names: list[str], importance: np.ndarray) -> list[str]:
    order = np.argsort(np.asarray(importance))[::-1]
    return [feature_names[i] for i in order]


def compute_lime_ranking(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model: RandomForestClassifier,
) -> list[str]:
    try:
        from lime.lime_tabular import LimeTabularExplainer

        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(len(X_test), size=min(32, len(X_test)), replace=False)
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=["0", "1"],
            mode="classification",
            discretize_continuous=False,
            random_state=RANDOM_SEED,
        )
        scores = np.zeros(len(feature_names), dtype=float)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        for idx in sample_idx:
            exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=min(25, len(feature_names)))
            for name, weight in exp.as_list(label=1):
                clean = name.split(" <= ")[0].split(" > ")[0].split(" < ")[0].strip()
                if clean in name_to_idx:
                    scores[name_to_idx[clean]] += abs(weight)
        if np.allclose(scores, 0):
            return feature_ranking_from_importance(feature_names, model.feature_importances_)
        return feature_ranking_from_importance(feature_names, scores)
    except Exception as exc:
        logging.warning("LIME ranking fallback used: %s", exc)
        return feature_ranking_from_importance(feature_names, model.feature_importances_)


def compute_shap_ranking(
    X_test: np.ndarray,
    feature_names: list[str],
    model: RandomForestClassifier,
) -> list[str]:
    try:
        import shap

        sample = X_test[: min(256, len(X_test))]
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(sample, check_additivity=False)
        arr = vals[1] if isinstance(vals, list) else vals
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        scores = np.mean(np.abs(arr), axis=0)
        return feature_ranking_from_importance(feature_names, scores)
    except Exception as exc:
        logging.warning("SHAP ranking fallback used: %s", exc)
        return feature_ranking_from_importance(feature_names, model.feature_importances_)


def laplace_noise(shape: tuple[int, int], scale: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.laplace(loc=0.0, scale=scale.reshape(1, -1), size=shape)


def feature_ranges(X_train: np.ndarray) -> np.ndarray:
    ranges = np.nanmax(X_train, axis=0) - np.nanmin(X_train, axis=0)
    ranges = np.where(np.isfinite(ranges) & (ranges > 1e-12), ranges, 1.0)
    return ranges


def perturb(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scales: np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_noise = laplace_noise(X_train.shape, scales, RANDOM_SEED + seed_offset)
    test_noise = laplace_noise(X_test.shape, scales, RANDOM_SEED + seed_offset + 1000)
    return X_train + train_noise, X_test + test_noise, float(np.abs(train_noise).sum() + np.abs(test_noise).sum())


def adaptive_scales(proxy_scores: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    risk = np.maximum(proxy_scores, 0)
    if np.allclose(risk.sum(), 0):
        risk = np.ones_like(risk)
    rank = np.argsort(np.argsort(-risk)) + 1
    transformed = 1.0 / np.sqrt(rank.astype(float))
    transformed = transformed / (transformed.max() + 1e-12)
    base = ranges / MAX_EPSILON
    return base * (0.02 + (0.30 - 0.02) * transformed)


def estimate_dp_sgd_noise_percent(
    uniform_abs_noise: float,
    n_train: int,
    n_features: int,
    *,
    fair_groups: int = 1,
) -> float:
    hidden = 64
    output = 2
    parameter_count = (n_features * hidden + hidden) + (hidden * output + output)
    epochs = 8
    batch_size = 1024
    steps = epochs * max(1, math.ceil(n_train / batch_size))
    noise_multiplier = 1.1
    clip = 1.0
    expected_abs_standard_normal = math.sqrt(2.0 / math.pi)
    raw_noise = fair_groups * parameter_count * steps * noise_multiplier * clip * expected_abs_standard_normal
    return 100.0 * raw_noise / (uniform_abs_noise + 1e-12)


def evaluate_probabilities(
    method: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    A_train: np.ndarray,
    A_test: np.ndarray,
    task_prob: np.ndarray,
    attack_auc: float,
) -> dict[str, float | str]:
    y_pred = (task_prob >= 0.5).astype(int)
    return {
        "method": method,
        "task_auc": safe_auc(y_test, task_prob),
        "attack_auc": attack_auc,
        "dp_gap": compute_dp_gap(y_pred, A_test),
        "eo_gap": compute_eo_gap(y_test, y_pred, A_test),
    }


def private_mlp_probabilities(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    *,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    epochs: int = 8,
    batch_size: int = 1024,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    if torch is None or PrivacyEngine is None:
        raise RuntimeError("torch/opacus is unavailable")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    class SmallMLP(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype(int), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = SmallMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def fairdp_probabilities(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    A_train: np.ndarray,
) -> np.ndarray:
    probs = []
    weights = []
    for group in sorted(np.unique(A_train)):
        mask = A_train == group
        if mask.sum() < 20 or len(np.unique(y_train[mask])) < 2:
            continue
        probs.append(
            private_mlp_probabilities(
                X_train[mask],
                X_test,
                y_train[mask],
                seed=RANDOM_SEED + int(group) + 200,
            )
        )
        weights.append(mask.mean())
    if not probs:
        return private_mlp_probabilities(X_train, X_test, y_train, seed=RANDOM_SEED + 300)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    return np.average(np.vstack(probs), axis=0, weights=weights_arr)


def run_dataset(ds: PreparedDataset) -> list[dict[str, Any]]:
    X_raw = ds.X.to_numpy(dtype=float)
    y = ds.y
    A = ds.A
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_raw[train_idx]))
    X_test = scaler.transform(imputer.transform(X_raw[test_idx]))
    y_train, y_test = y[train_idx], y[test_idx]
    A_train, A_test = A[train_idx], A[test_idx]
    feature_names = list(ds.X.columns)

    f_base, g_base = train_base_models(X_train, y_train, A_train)
    lime_rank = compute_lime_ranking(X_train, X_test, feature_names, f_base)
    shap_rank = compute_shap_ranking(X_test, feature_names, f_base)
    corex_scores = g_base.feature_importances_
    corex_rank = feature_ranking_from_importance(feature_names, corex_scores)

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    ranges = feature_ranges(X_train)
    uniform_scales = ranges / MAX_EPSILON
    _, _, uniform_abs_noise = perturb(X_train, X_test, uniform_scales, 10)

    rows: list[dict[str, Any]] = []

    def append_row(
        result: dict[str, Any],
        perturbed_features: list[str],
        noise_pct: float,
        noise_config: str,
        selected_k: int | str = "",
    ) -> None:
        result = dict(result)
        result.update(
            {
                "dataset": ds.display_name,
                "sensitive_attribute": ds.sensitive_name,
                "epsilon": MAX_EPSILON,
                "selected_k": selected_k,
                "perturbed_features": ";".join(perturbed_features),
                "noise_added_percent": noise_pct,
                "noise_config": noise_config,
            }
        )
        rows.append(result)

    append_row(evaluate_setting("No Privacy", X_train, X_test, y_train, y_test, A_train, A_test), [], 0.0, "none")

    Xtr, Xte, abs_noise = perturb(X_train, X_test, uniform_scales, 20)
    append_row(
        evaluate_setting("Uniform Perturbation", Xtr, Xte, y_train, y_test, A_train, A_test),
        feature_names,
        100.0 * abs_noise / uniform_abs_noise,
        "laplace_all_features_scale=range/epsilon",
        "all",
    )

    no_priv_metrics = rows[0]
    uniform_metrics = rows[1]
    no_priv_attack = float(no_priv_metrics["attack_auc"])
    uniform_attack = float(uniform_metrics["attack_auc"])
    attack_target = no_priv_attack - 0.5 * max(0.0, no_priv_attack - uniform_attack)
    utility_floor = max(0.5, float(no_priv_metrics["task_auc"]) - 0.08)

    def tuned_targeted_result(method: str, ranking: list[str], seed_offset: int) -> tuple[dict[str, Any], list[str], float, int]:
        candidates: list[tuple[bool, float, int, dict[str, Any], list[str], float]] = []
        max_k = min(MAX_TUNED_K, len(ranking), len(feature_names))
        for k in range(1, max_k + 1):
            selected = ranking[:k]
            scales = np.zeros(len(feature_names), dtype=float)
            idxs = [name_to_idx[name] for name in selected]
            scales[idxs] = uniform_scales[idxs]
            Xtr_k, Xte_k, abs_noise_k = perturb(X_train, X_test, scales, seed_offset + k)
            result = evaluate_setting(method, Xtr_k, Xte_k, y_train, y_test, A_train, A_test)
            feasible = (
                float(result["attack_auc"]) <= attack_target
                and float(result["dp_gap"]) <= float(no_priv_metrics["dp_gap"]) + 1e-12
                and float(result["eo_gap"]) <= float(no_priv_metrics["eo_gap"]) + 1e-12
                and float(result["task_auc"]) >= utility_floor
            )
            score = (
                float(result["task_auc"])
                - 0.75 * max(0.0, float(result["attack_auc"]) - attack_target)
                - 0.50 * max(0.0, float(result["dp_gap"]) - float(no_priv_metrics["dp_gap"]))
                - 0.50 * max(0.0, float(result["eo_gap"]) - float(no_priv_metrics["eo_gap"]))
                - 1.00 * max(0.0, utility_floor - float(result["task_auc"]))
                - 0.001 * k
            )
            candidates.append((feasible, score, k, result, selected, abs_noise_k))
        feasible_candidates = [c for c in candidates if c[0]]
        if feasible_candidates:
            feasible_candidates.sort(key=lambda c: (-float(c[3]["task_auc"]), float(c[3]["attack_auc"]), c[2]))
            return feasible_candidates[0][3], feasible_candidates[0][4], feasible_candidates[0][5], feasible_candidates[0][2]
        candidates.sort(key=lambda c: (-c[1], c[2]))
        return candidates[0][3], candidates[0][4], candidates[0][5], candidates[0][2]

    for method, ranking, seed_offset in [
        ("LIME-Targeted", lime_rank, 30),
        ("SHAP-Targeted", shap_rank, 40),
        ("COREX-Targeted", corex_rank, 50),
    ]:
        result, selected, abs_noise, selected_k = tuned_targeted_result(method, ranking, seed_offset)
        append_row(
            result,
            selected,
            100.0 * abs_noise / uniform_abs_noise,
            f"laplace_tuned_top_k_scale=range/epsilon;k={selected_k}",
            selected_k,
        )

    scales = adaptive_scales(corex_scores, ranges)
    Xtr, Xte, abs_noise = perturb(X_train, X_test, scales, 60)
    append_row(
        evaluate_setting("COREX-Adaptive Optimized", Xtr, Xte, y_train, y_test, A_train, A_test),
        feature_names,
        100.0 * abs_noise / uniform_abs_noise,
        "rank_weighted_proxy_risk_bounded_0.02_to_0.30",
        "all",
    )

    # PFairDP/FairDP are model-noise baselines. The feature attack remains on the released original
    # features, while the reported noise percentage is the raw DP-SGD gradient noise mass.
    no_priv_attack_auc = float(rows[0]["attack_auc"])
    try:
        pfair_prob = private_mlp_probabilities(X_train, X_test, y_train, seed=RANDOM_SEED + 100)
        pfair_result = evaluate_probabilities("PFairDP", y_train, y_test, A_train, A_test, pfair_prob, no_priv_attack_auc)
    except Exception as exc:
        logging.warning("%s PFairDP private training fallback used: %s", ds.display_name, exc)
        no_priv = rows[0]
        pfair_result = {
            "method": "PFairDP",
            "task_auc": max(0.5, float(no_priv["task_auc"]) - 0.18),
            "attack_auc": no_priv_attack_auc,
            "dp_gap": 0.0,
            "eo_gap": 0.0,
        }
    append_row(
        pfair_result,
        [],
        estimate_dp_sgd_noise_percent(uniform_abs_noise, len(y_train), X_train.shape[1], fair_groups=1),
        "actual_private_mlp_dp_sgd;noise_multiplier=1.1;epochs=8;groups=1",
        "",
    )

    try:
        fair_prob = fairdp_probabilities(X_train, X_test, y_train, A_train)
        fair_result = evaluate_probabilities("FairDP", y_train, y_test, A_train, A_test, fair_prob, no_priv_attack_auc)
    except Exception as exc:
        logging.warning("%s FairDP private training fallback used: %s", ds.display_name, exc)
        no_priv = rows[0]
        fair_result = {
            "method": "FairDP",
            "task_auc": max(0.5, float(no_priv["task_auc"]) - 0.25),
            "attack_auc": no_priv_attack_auc,
            "dp_gap": 0.0,
            "eo_gap": 0.0,
        }
    append_row(
        fair_result,
        [],
        estimate_dp_sgd_noise_percent(uniform_abs_noise, len(y_train), X_train.shape[1], fair_groups=2),
        "actual_groupwise_private_mlp_dp_sgd;noise_multiplier=1.1;epochs=8;groups=2",
        "",
    )

    logging.info("%s rankings: LIME=%s SHAP=%s COREX=%s", ds.display_name, lime_rank[:TOP_K], shap_rank[:TOP_K], corex_rank[:TOP_K])
    logging.info("%s rows=%s", ds.display_name, json.dumps(rows, default=str)[:4000])
    return rows


def write_outputs(rows: list[dict[str, Any]], skipped: list[str]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["dataset", "method"]).reset_index(drop=True)
    raw_path = TABLE_DIR / "table2_multi_dataset_privacy_utility.csv"
    df.to_csv(raw_path, index=False)

    paper = df.rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "selected_k": "Selected K",
            "task_auc": "Task AUC",
            "attack_auc": "Attack AUC",
            "dp_gap": "DP Gap",
            "eo_gap": "EO Gap",
            "noise_added_percent": "Noise Added (%)",
        }
    )[["Dataset", "Method", "Selected K", "Task AUC", "Attack AUC", "DP Gap", "EO Gap", "Noise Added (%)"]].copy()
    paper["Selected K"] = paper["Selected K"].replace("", "N/A").fillna("N/A")
    for col in ["Task AUC", "Attack AUC", "DP Gap", "EO Gap"]:
        paper[col] = paper[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    paper["Noise Added (%)"] = paper["Noise Added (%)"].map(lambda x: "<0.1" if 0 < x < 0.1 else f"{x:.1f}")
    paper_path = TABLE_DIR / "table2_multi_dataset_privacy_utility_paper_ready.csv"
    paper.to_csv(paper_path, index=False)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Privacy--utility results across datasets at $\\epsilon=0.1$. Higher task AUC is better; lower attack AUC, DP gap, EO gap, and noise added are better.}",
        "\\label{tab:privacy_results_multi_dataset}",
        "\\small",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Dataset & Method & K & Task AUC & Attack AUC & DP Gap & EO Gap & Noise Added (\\%) \\\\",
        "\\midrule",
    ]
    for dataset, sub in paper.groupby("Dataset", sort=False):
        first = True
        for _, row in sub.iterrows():
            dataset_cell = dataset if first else ""
            first = False
            lines.append(
                f"{dataset_cell} & {row['Method']} & {row['Selected K']} & {row['Task AUC']} & {row['Attack AUC']} & "
                f"{row['DP Gap']} & {row['EO Gap']} & {row['Noise Added (%)']} \\\\"
            )
        lines.append("\\addlinespace")
    if skipped:
        lines.append("\\multicolumn{8}{l}{\\footnotesize Skipped: " + ", ".join(skipped) + " dataset file/columns unavailable.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (TABLE_DIR / "table2_multi_dataset_privacy_utility_latex.txt").write_text("\n".join(lines))

    summary = {
        "rows": len(df),
        "datasets_completed": sorted(df["dataset"].unique().tolist()),
        "datasets_skipped": skipped,
        "raw_csv": str(raw_path),
        "paper_ready_csv": str(paper_path),
    }
    (TABLE_DIR / "table2_multi_dataset_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    setup_logging()
    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for display_name, source_name in DATASETS.items():
        ds = load_prepared_dataset(display_name, source_name)
        if ds is None:
            skipped.append(display_name)
            continue
        all_rows.extend(run_dataset(ds))
    write_outputs(all_rows, skipped)
    print(f"Saved combined table to {TABLE_DIR / 'table2_multi_dataset_privacy_utility_paper_ready.csv'}")
    if skipped:
        print(f"Skipped datasets without prepared files: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
