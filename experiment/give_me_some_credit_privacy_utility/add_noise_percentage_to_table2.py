from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TOP_K = 5
EPSILON_FOR_PAPER = 0.1
METHOD_INDEX = {
    "No Privacy": 0,
    "Uniform Perturbation": 1,
    "LIME-Targeted": 2,
    "SHAP-Targeted": 3,
    "COREX-Targeted": 4,
    "COREX-Adaptive Original": 5,
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def total_absolute_noise(table2_mod, x_train: pd.DataFrame, x_test: pd.DataFrame, scales: dict[str, float], train_seed: int, test_seed: int) -> float:
    x_train_noisy = table2_mod.add_laplace_noise(x_train, scales, train_seed)
    x_test_noisy = table2_mod.add_laplace_noise(x_test, scales, test_seed)
    train_noise = np.abs(x_train_noisy.to_numpy(dtype=float) - x_train.to_numpy(dtype=float)).sum()
    test_noise = np.abs(x_test_noisy.to_numpy(dtype=float) - x_test.to_numpy(dtype=float)).sum()
    return float(train_noise + test_noise)


def scales_for_row(table2_mod, row: pd.Series, x_train: pd.DataFrame, feature_names: list[str], rankings: dict[str, pd.DataFrame]) -> tuple[dict[str, float], int, int]:
    method = str(row["method"])
    epsilon = float(row["privacy_budget"])

    if method == "COREX-Adaptive Optimized":
        config = json.loads(row["noise_config"])
        config_id = int(config["config_id"])
        scales = {feature: float(value) for feature, value in config["noise_scales"].items()}
        return scales, SEED + config_id * 2 + 1, SEED + config_id * 2 + 2

    method_for_scales = "COREX-Adaptive" if method == "COREX-Adaptive Original" else method
    scales, _, _ = table2_mod.method_dp_scales(method_for_scales, x_train, feature_names, rankings, TOP_K, epsilon)
    method_index = METHOD_INDEX[method]
    train_seed = SEED + 10_000 + int(epsilon * 1_000_000) + method_index
    test_seed = SEED + 20_000 + int(epsilon * 1_000_000) + method_index
    return scales, train_seed, test_seed


def latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Privacy--utility results on Give Me Some Credit at $\epsilon=0.1$. Higher task AUC is better; lower attack AUC, DP gap, EO gap, and noise added are better. Noise added is normalized so Uniform Perturbation is 100\%.}",
        r"\label{tab:privacy_results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Task AUC & Attack AUC & DP Gap & EO Gap & Noise Added (\%) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        noise_value = float(row["noise_added_percent"])
        noise_display = "<0.1" if 0.0 < noise_value < 0.1 else f"{noise_value:.1f}"
        lines.append(
            f"{row['method']} & {row['task_auc']:.3f} & {row['attack_auc']:.3f} & "
            f"{row['dp_gap']:.3f} & {row['eo_gap']:.3f} & {noise_display} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    experiment_dir = Path(__file__).resolve().parents[1]
    outputs = experiment_dir / "outputs"
    table2_mod = import_module(experiment_dir / "scripts" / "run_table2_privacy_utility.py", "table2_privacy_module")

    dirs = table2_mod.make_dirs(outputs)
    raw_path = table2_mod.ensure_dataset(None, dirs["data"])
    x_train, x_test, y_train, y_test, a_train, a_test, feature_names = table2_mod.load_data(raw_path, dirs, 0.2, SEED)
    rankings = table2_mod.require_rankings(dirs["rankings"])

    table_path = outputs / "tables" / "table2_privacy_utility_optimized_adaptive.csv"
    table = pd.read_csv(table_path)

    uniform_totals: dict[float, float] = {}
    for epsilon in sorted(table["privacy_budget"].astype(float).unique()):
        uniform_scales, _, _ = table2_mod.method_dp_scales("Uniform Perturbation", x_train, feature_names, rankings, TOP_K, epsilon)
        uniform_index = METHOD_INDEX["Uniform Perturbation"]
        uniform_totals[epsilon] = total_absolute_noise(
            table2_mod,
            x_train,
            x_test,
            uniform_scales,
            SEED + 10_000 + int(epsilon * 1_000_000) + uniform_index,
            SEED + 20_000 + int(epsilon * 1_000_000) + uniform_index,
        )

    totals = []
    percentages = []
    for _, row in table.iterrows():
        epsilon = float(row["privacy_budget"])
        scales, train_seed, test_seed = scales_for_row(table2_mod, row, x_train, feature_names, rankings)
        total = total_absolute_noise(table2_mod, x_train, x_test, scales, train_seed, test_seed)
        percent = 0.0 if uniform_totals[epsilon] <= 0 else 100.0 * total / uniform_totals[epsilon]
        if row["method"] == "Uniform Perturbation":
            percent = 100.0
        totals.append(total)
        percentages.append(percent)

    table["total_absolute_noise"] = totals
    table["noise_added_percent"] = percentages
    out_csv = outputs / "tables" / "table2_privacy_utility_optimized_adaptive_with_noise.csv"
    table.to_csv(out_csv, index=False)

    paper = table[np.isclose(table["privacy_budget"].astype(float), EPSILON_FOR_PAPER)].copy()
    paper_ready = paper[
        ["method", "task_auc", "attack_auc", "dp_gap", "eo_gap", "noise_added_percent"]
    ].copy()
    paper_ready.columns = ["Method", "Task AUC", "Attack AUC", "DP Gap", "EO Gap", "Noise Added (%)"]
    paper_ready[["Task AUC", "Attack AUC", "DP Gap", "EO Gap"]] = paper_ready[
        ["Task AUC", "Attack AUC", "DP Gap", "EO Gap"]
    ].round(3)
    paper_ready["Noise Added (%)"] = paper_ready["Noise Added (%)"].map(
        lambda value: "<0.1" if 0.0 < float(value) < 0.1 else f"{float(value):.1f}"
    )
    paper_ready.to_csv(outputs / "tables" / "table2_privacy_utility_epsilon_0p1_optimized_with_noise_paper_ready.csv", index=False)

    latex = latex_table(paper)
    (outputs / "tables" / "table2_privacy_utility_epsilon_0p1_optimized_with_noise_latex.txt").write_text(latex, encoding="utf-8")

    print(out_csv)
    print(outputs / "tables" / "table2_privacy_utility_epsilon_0p1_optimized_with_noise_paper_ready.csv")
    print(outputs / "tables" / "table2_privacy_utility_epsilon_0p1_optimized_with_noise_latex.txt")


if __name__ == "__main__":
    main()
