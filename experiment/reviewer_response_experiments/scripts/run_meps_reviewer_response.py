from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corex.games.game_theory import BanzhafIndex, CooperativeGame

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
DATA_DIR = ROOT / "Privacy_Experiement_1" / "MEPS_19"

SEEDS = list(range(42, 52))
MC_BUDGETS = [50, 100, 250, 500, 1000, 2000, 5000]
REFERENCE_MC_BUDGET = 10000
MC_VALIDATION_SAMPLE_SIZE = 1000
TOP_K = 7
BASE_EPSILON = 0.1
MAX_FEATURES_FOR_MC = 24


def setup() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "meps_reviewer_response.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )


def load_meps() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    con = pd.read_csv(DATA_DIR / "training_con.csv")
    cat = pd.read_csv(DATA_DIR / "training_cat.csv")
    y = pd.read_csv(DATA_DIR / "class.csv").iloc[:, 0].astype(int).to_numpy()
    A = pd.read_csv(DATA_DIR / "sensitive.csv")["sex_male"].astype(int).to_numpy()
    X = pd.concat([con, pd.get_dummies(cat.astype(str), drop_first=False)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.columns = [str(c) for c in X.columns]
    return X, y, A


def split_data(X: pd.DataFrame, y: np.ndarray, A: np.ndarray, seed: int) -> dict:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(X.iloc[train_idx].to_numpy(float)))
    x_test = scaler.transform(imputer.transform(X.iloc[test_idx].to_numpy(float)))
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "a_train": A[train_idx],
        "a_test": A[test_idx],
        "feature_names": list(X.columns),
    }


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def dp_gap(y_pred: np.ndarray, A: np.ndarray) -> float:
    return float(abs(np.mean(y_pred[A == 0]) - np.mean(y_pred[A == 1])))


def eo_gap(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    rates = []
    for group in [0, 1]:
        mask = (A == group) & (y_true == 1)
        if not np.any(mask):
            return float("nan")
        rates.append(float(np.mean(y_pred[mask])))
    return float(abs(rates[0] - rates[1]))


def evaluate(x_train: np.ndarray, x_test: np.ndarray, split: dict, seed: int, method: str) -> dict:
    task = rf(seed)
    task.fit(x_train, split["y_train"])
    task_prob = task.predict_proba(x_test)[:, 1]
    pred = (task_prob >= 0.5).astype(int)

    attack = rf(seed + 1000)
    attack.fit(x_train, split["a_train"])
    attack_prob = attack.predict_proba(x_test)[:, 1]
    return {
        "method": method,
        "task_auc": safe_auc(split["y_test"], task_prob),
        "attack_auc": safe_auc(split["a_test"], attack_prob),
        "dp_gap": dp_gap(pred, split["a_test"]),
        "eo_gap": eo_gap(split["y_test"], pred, split["a_test"]),
    }


def full_proxy_scores(split: dict, seed: int) -> tuple[np.ndarray, list[str]]:
    proxy = rf(seed)
    proxy.fit(split["x_train"], split["a_train"])
    scores = proxy.feature_importances_.astype(float)
    return scores, split["feature_names"]


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(a)[::-1][:k])
    top_b = set(np.argsort(b)[::-1][:k])
    return len(top_a & top_b) / len(top_a | top_b)


def build_proxy_coalition_game(split: dict, selected_idx: np.ndarray, seed: int) -> CooperativeGame:
    """Create the actual COREX proxy game used for Monte Carlo convergence.

    The proxy model is fitted only on an inner-training split. Coalition values are
    validation AUCs after masking absent coalition members to the standardized
    baseline value 0. This exercises COREX's sampled Banzhaf solver path directly.
    """
    inner_idx, val_idx = train_test_split(
        np.arange(len(split["a_train"])),
        test_size=0.2,
        random_state=seed,
        stratify=split["a_train"] if len(np.unique(split["a_train"])) > 1 else None,
    )
    rng = np.random.default_rng(seed)
    if len(val_idx) > MC_VALIDATION_SAMPLE_SIZE:
        val_idx = rng.choice(val_idx, MC_VALIDATION_SAMPLE_SIZE, replace=False)

    x_inner = split["x_train"][inner_idx][:, selected_idx]
    a_inner = split["a_train"][inner_idx]
    x_val = split["x_train"][val_idx][:, selected_idx]
    a_val = split["a_train"][val_idx]

    proxy = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    proxy.fit(x_inner, a_inner)

    cache: dict[tuple[int, ...], float] = {}

    def value_function(coalition: frozenset[int]) -> float:
        key = tuple(sorted(coalition))
        cached = cache.get(key)
        if cached is not None:
            return cached
        masked = np.zeros_like(x_val)
        if key:
            cols = np.asarray(key, dtype=int)
            masked[:, cols] = x_val[:, cols]
        score = proxy.predict_proba(masked)[:, 1]
        value = safe_auc(a_val, score)
        cache[key] = value
        return value

    selected_names = [split["feature_names"][i] for i in selected_idx]
    return CooperativeGame(
        players=selected_names,
        value_function=value_function,
        value_function_name="validation_proxy_auc_with_masking",
        game_metadata={
            "dataset": "MEPS_19",
            "selected_feature_count": int(len(selected_idx)),
            "inner_train_size": int(len(inner_idx)),
            "validation_size": int(len(val_idx)),
            "reference": "actual_corex_banzhaf_monte_carlo",
        },
        cache_mode="sampled",
        sample_budget=REFERENCE_MC_BUDGET,
        coalition_representation="indices",
        max_cache_size=max(128, REFERENCE_MC_BUDGET * 30),
    )


def solve_corex_banzhaf_mc(game: CooperativeGame, budget: int, seed: int) -> tuple[np.ndarray, dict]:
    solver = BanzhafIndex(sample_size=budget, random_seed=seed)
    solution = solver.solve(game)
    diagnostics = dict(solution.solver_diagnostics)
    diagnostics["coalition_cache_size"] = game.coalition_cache_size()
    return np.asarray(solution.allocations, dtype=float), diagnostics


def run_mc_convergence(X: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        split = split_data(X, y, A, seed)
        proxy_scores, feature_names = full_proxy_scores(split, seed)
        top_idx = np.argsort(proxy_scores)[::-1][:MAX_FEATURES_FOR_MC]
        game = build_proxy_coalition_game(split, top_idx, seed)
        start = time.perf_counter()
        reference, ref_diag = solve_corex_banzhaf_mc(game, REFERENCE_MC_BUDGET, seed + REFERENCE_MC_BUDGET)
        reference_runtime = time.perf_counter() - start
        logging.info(
            "COREX MC reference seed=%s budget=%s method=%s cache=%s runtime=%.3f",
            seed,
            REFERENCE_MC_BUDGET,
            ref_diag.get("method"),
            ref_diag.get("coalition_cache_size"),
            reference_runtime,
        )
        for budget in MC_BUDGETS:
            start = time.perf_counter()
            approx, diagnostics = solve_corex_banzhaf_mc(game, budget, seed + budget)
            runtime = time.perf_counter() - start
            corr = spearmanr(reference, approx).correlation
            if np.isnan(corr):
                corr = 0.0
            rows.append(
                {
                    "seed": seed,
                    "mc_budget": budget,
                    "reference_mc_budget": REFERENCE_MC_BUDGET,
                    "solver_method": diagnostics.get("method"),
                    "exact_or_approximate": diagnostics.get("exact_or_approximate"),
                    "solver_samples": diagnostics.get("samples"),
                    "coalition_cache_size": diagnostics.get("coalition_cache_size"),
                    "selected_feature_count": int(len(top_idx)),
                    "spearman_vs_reference": float(corr),
                    "jaccard_top7": topk_jaccard(reference, approx, min(TOP_K, len(reference))),
                    "l2_error": float(np.linalg.norm(approx - reference) / (np.linalg.norm(reference) + 1e-12)),
                    "runtime_seconds": runtime,
                    "reference_runtime_seconds": reference_runtime,
                    "candidate_features": ";".join([feature_names[i] for i in top_idx]),
                    "reference_top7_features": ";".join([feature_names[top_idx[i]] for i in np.argsort(reference)[::-1][:TOP_K]]),
                    "approx_top7_features": ";".join([feature_names[top_idx[i]] for i in np.argsort(approx)[::-1][:TOP_K]]),
                }
            )
        logging.info("Finished actual COREX MC convergence seed=%s", seed)
    return pd.DataFrame(rows)


def feature_ranges(x_train: np.ndarray) -> np.ndarray:
    ranges = np.nanmax(x_train, axis=0) - np.nanmin(x_train, axis=0)
    return np.where(np.isfinite(ranges) & (ranges > 1e-12), ranges, 1.0)


def laplace_perturb(x_train: np.ndarray, x_test: np.ndarray, scales: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        x_train + rng.laplace(0, scales.reshape(1, -1), size=x_train.shape),
        x_test + rng.laplace(0, scales.reshape(1, -1), size=x_test.shape),
    )


def knn_manifold_perturb(x_train: np.ndarray, x_test: np.ndarray, cols: np.ndarray, seed: int, k_neighbors: int = 25) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train_new = x_train.copy()
    x_test_new = x_test.copy()
    context_cols = np.setdiff1d(np.arange(x_train.shape[1]), cols)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(x_train)), metric="euclidean")
    nn.fit(x_train[:, context_cols])
    for matrix, out in [(x_train, x_train_new), (x_test, x_test_new)]:
        neigh = nn.kneighbors(matrix[:, context_cols], return_distance=False)
        picks = rng.integers(0, neigh.shape[1], size=len(matrix))
        donor_rows = neigh[np.arange(len(matrix)), picks]
        out[:, cols] = x_train[donor_rows][:, cols]
    return x_train_new, x_test_new


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_samples: int = 800, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x_s = x[rng.choice(len(x), min(max_samples, len(x)), replace=False)]
    y_s = y[rng.choice(len(y), min(max_samples, len(y)), replace=False)]
    z = np.vstack([x_s, y_s])
    d = cdist(z[: min(300, len(z))], z[: min(300, len(z))], "sqeuclidean")
    gamma = 1.0 / (np.median(d[d > 0]) + 1e-12)
    kxx = np.exp(-gamma * cdist(x_s, x_s, "sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(y_s, y_s, "sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(x_s, y_s, "sqeuclidean")).mean()
    return float(kxx + kyy - 2 * kxy)


def corr_error(x_ref: np.ndarray, x_new: np.ndarray) -> float:
    corr_ref = np.corrcoef(x_ref, rowvar=False)
    corr_new = np.corrcoef(x_new, rowvar=False)
    corr_ref = np.nan_to_num(corr_ref)
    corr_new = np.nan_to_num(corr_new)
    return float(np.mean(np.abs(corr_ref - corr_new)))


def run_manifold_study(X: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        split = split_data(X, y, A, seed)
        scores, feature_names = full_proxy_scores(split, seed)
        top_cols = np.argsort(scores)[::-1][:TOP_K]
        ranges = feature_ranges(split["x_train"])
        scales = np.zeros_like(ranges)
        scales[top_cols] = ranges[top_cols] / BASE_EPSILON

        settings = [("No Privacy", split["x_train"], split["x_test"])]
        lap_train, lap_test = laplace_perturb(split["x_train"], split["x_test"], scales, seed + 10)
        settings.append(("COREX-Targeted Laplace", lap_train, lap_test))
        knn_train, knn_test = knn_manifold_perturb(split["x_train"], split["x_test"], top_cols, seed + 20)
        settings.append(("COREX-Targeted kNN-Manifold", knn_train, knn_test))

        for method, xtr, xte in settings:
            result = evaluate(xtr, xte, split, seed, method)
            result.update(
                {
                    "seed": seed,
                    "selected_features": ";".join([feature_names[i] for i in top_cols]),
                    "mmd_train": 0.0 if method == "No Privacy" else mmd_rbf(split["x_train"], xtr, seed=seed),
                    "corr_error_train": 0.0 if method == "No Privacy" else corr_error(split["x_train"], xtr),
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg = {}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    return df.groupby(group_cols, as_index=False).agg(**agg)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[df["method"] == "COREX-Targeted Laplace"].set_index("seed")
    for method in ["No Privacy", "COREX-Targeted kNN-Manifold"]:
        comp = df[df["method"] == method].set_index("seed")
        common = sorted(set(base.index) & set(comp.index))
        for metric in ["task_auc", "attack_auc", "dp_gap", "eo_gap", "mmd_train", "corr_error_train"]:
            if len(common) < 2:
                p_value = np.nan
            else:
                try:
                    p_value = wilcoxon(base.loc[common, metric], comp.loc[common, metric]).pvalue
                except ValueError:
                    p_value = 1.0
            rows.append(
                {
                    "baseline": "COREX-Targeted Laplace",
                    "comparison": method,
                    "metric": metric,
                    "mean_difference_comparison_minus_baseline": float(comp.loc[common, metric].mean() - base.loc[common, metric].mean()),
                    "wilcoxon_p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def plot_mc(mc: pd.DataFrame) -> None:
    summary = summarize(mc, ["mc_budget"], ["spearman_vs_reference", "jaccard_top7", "l2_error", "runtime_seconds"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].errorbar(summary["mc_budget"], summary["spearman_vs_reference_mean"], yerr=summary["spearman_vs_reference_std"], marker="o")
    axes[0].set_ylabel("Spearman vs reference")
    axes[1].errorbar(summary["mc_budget"], summary["jaccard_top7_mean"], yerr=summary["jaccard_top7_std"], marker="o")
    axes[1].set_ylabel("Jaccard@7")
    axes[2].errorbar(summary["mc_budget"], summary["l2_error_mean"], yerr=summary["l2_error_std"], marker="o")
    axes[2].set_ylabel("Relative L2 error")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Monte Carlo budget")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meps_mc_convergence.png", dpi=220)
    fig.savefig(FIG_DIR / "meps_mc_convergence.pdf")
    plt.close(fig)


def plot_manifold(df: pd.DataFrame) -> None:
    summary = summarize(df, ["method"], ["task_auc", "attack_auc", "mmd_train", "corr_error_train"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = ["task_auc", "attack_auc", "mmd_train", "corr_error_train"]
    labels = ["Task AUC", "Attack AUC", "MMD", "Correlation Error"]
    for ax, metric, label in zip(axes, metrics, labels):
        ax.bar(summary["method"], summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], color="#4b5563")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "meps_manifold_perturbation_comparison.png", dpi=220)
    fig.savefig(FIG_DIR / "meps_manifold_perturbation_comparison.pdf")
    plt.close(fig)


def main() -> None:
    setup()
    X, y, A = load_meps()
    logging.info("Loaded MEPS rows=%s features=%s", len(X), X.shape[1])

    mc = run_mc_convergence(X, y, A)
    mc.to_csv(TABLE_DIR / "meps_mc_convergence_raw.csv", index=False)
    summarize(mc, ["mc_budget"], ["spearman_vs_reference", "jaccard_top7", "l2_error", "runtime_seconds"]).to_csv(
        TABLE_DIR / "meps_mc_convergence_summary.csv", index=False
    )
    plot_mc(mc)

    manifold = run_manifold_study(X, y, A)
    manifold.to_csv(TABLE_DIR / "meps_manifold_perturbation_raw.csv", index=False)
    summarize(manifold, ["method"], ["task_auc", "attack_auc", "dp_gap", "eo_gap", "mmd_train", "corr_error_train"]).to_csv(
        TABLE_DIR / "meps_manifold_perturbation_summary.csv", index=False
    )
    paired_tests(manifold).to_csv(TABLE_DIR / "meps_statistical_tests.csv", index=False)
    plot_manifold(manifold)

    config = {
        "dataset": "MEPS_19",
        "seeds": SEEDS,
        "mc_budgets": MC_BUDGETS,
        "reference_mc_budget": REFERENCE_MC_BUDGET,
        "mc_validation_sample_size": MC_VALIDATION_SAMPLE_SIZE,
        "top_k": TOP_K,
        "base_epsilon": BASE_EPSILON,
        "max_features_for_mc": MAX_FEATURES_FOR_MC,
        "mc_convergence_method": "actual_corex_BanzhafIndex_sample_size_on_proxy_coalition_game",
        "mc_value_function": "validation_proxy_auc_with_absent_features_masked_to_standardized_zero",
    }
    (OUTPUT_DIR / "meps_reviewer_response_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(TABLE_DIR / "meps_mc_convergence_summary.csv")
    print(TABLE_DIR / "meps_manifold_perturbation_summary.csv")
    print(TABLE_DIR / "meps_statistical_tests.csv")


if __name__ == "__main__":
    main()
