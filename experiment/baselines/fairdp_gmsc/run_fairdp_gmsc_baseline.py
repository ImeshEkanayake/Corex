from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
DATA_URL = "https://raw.githubusercontent.com/dsrscientist/dataset5/main/cs-training.csv"
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
PRIVACY_BUDGETS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


@dataclass
class PreparedData:
    feature_names: list[str]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    a_train: pd.Series
    a_test: pd.Series


class FairDPNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "data": root / "outputs" / "data",
        "tables": root / "outputs" / "tables",
        "models": root / "outputs" / "models",
        "logs": root / "outputs" / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler()],
    )


def ensure_dataset(data_dir: Path) -> Path:
    path = data_dir / "cs-training.csv"
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parents[4]
    candidates = [
        project_root / "Paper Experiemnts" / "give_me_some_credit_corex" / "outputs" / "data" / "cs-training.csv",
        project_root / "Paper Experiemnts" / "give_me_some_credit_table2_privacy_utility" / "outputs" / "data" / "cs-training.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, path)
            return path
    urlretrieve(DATA_URL, path)
    return path


def load_data(path: Path) -> PreparedData:
    raw = pd.read_csv(path)
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
    sensitive = (raw[AGE] >= float(raw[AGE].median())).astype(int).rename("A_age_ge_median")
    x_raw = raw.drop(columns=[TARGET, AGE])
    y = raw[TARGET]
    train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=SEED, stratify=y)
    x_train_raw = x_raw.iloc[train_idx].reset_index(drop=True)
    x_test_raw = x_raw.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    a_train = sensitive.iloc[train_idx].reset_index(drop=True)
    a_test = sensitive.iloc[test_idx].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(x_train_raw)), columns=x_raw.columns)
    x_test = pd.DataFrame(scaler.transform(imputer.transform(x_test_raw)), columns=x_raw.columns)
    return PreparedData(list(x_raw.columns), x_train, x_test, y_train, y_test, a_train, a_test)


def compute_dp_gap(y_pred: np.ndarray, a: pd.Series) -> float:
    groups = []
    values = a.to_numpy(dtype=int)
    for group in [0, 1]:
        mask = values == group
        groups.append(float(np.mean(y_pred[mask])) if mask.any() else np.nan)
    return float(abs(groups[1] - groups[0]))


def compute_eo_gap(y_true: pd.Series, y_pred: np.ndarray, a: pd.Series) -> float:
    y_values = y_true.to_numpy(dtype=int)
    a_values = a.to_numpy(dtype=int)
    tprs = []
    for group in [0, 1]:
        mask = (a_values == group) & (y_values == 1)
        if not mask.any():
            return float("nan")
        tprs.append(float(np.mean(y_pred[mask])))
    return float(abs(tprs[1] - tprs[0]))


def feature_attack_auc(data: PreparedData) -> float:
    attack = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
    attack.fit(data.x_train, data.a_train)
    return float(roc_auc_score(data.a_test, attack.predict_proba(data.x_test)[:, 1]))


def uniform_input_noise_reference(root: Path) -> dict[float, float]:
    table_path = (
        root.parents[1]
        / "give_me_some_credit_table2_privacy_utility"
        / "outputs"
        / "tables"
        / "table2_privacy_utility_optimized_adaptive_with_noise.csv"
    )
    table = pd.read_csv(table_path)
    uniform = table[table["method"] == "Uniform Perturbation"]
    return {float(row["privacy_budget"]): float(row["total_absolute_noise"]) for _, row in uniform.iterrows()}


def make_loader(x: pd.DataFrame, y: pd.Series, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x.to_numpy(dtype=np.float32)),
        torch.tensor(y.to_numpy(dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train_group_model(x: pd.DataFrame, y: pd.Series, epsilon: float, epochs: int, batch_size: int, clip: float) -> tuple[nn.Module, float, float, int]:
    model = FairDPNet(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = make_loader(x, y, batch_size)
    privacy_engine = PrivacyEngine()
    fallback_noise_multiplier = None
    try:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=float(epsilon),
            target_delta=1e-5,
            max_grad_norm=clip,
        )
    except ValueError as exc:
        fallback_noise_multiplier = max(25.0, 5.0 / float(epsilon))
        logging.warning("FairDP Opacus target epsilon=%s failed (%s); fallback noise_multiplier=%s", epsilon, exc, fallback_noise_multiplier)
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=fallback_noise_multiplier,
            max_grad_norm=clip,
        )
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        losses = []
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        logging.info("group size=%s epsilon=%s epoch=%s loss=%.6f", len(x), epsilon, epoch + 1, float(np.mean(losses)))
    spent = float(privacy_engine.get_epsilon(1e-5))
    noise_multiplier = float(getattr(optimizer, "noise_multiplier", fallback_noise_multiplier or 0.0))
    return model, noise_multiplier, spent, len(loader)


def average_state_dict(model_a: nn.Module, model_b: nn.Module) -> dict[str, torch.Tensor]:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    state_a = {key.removeprefix("_module."): value for key, value in state_a.items()}
    state_b = {key.removeprefix("_module."): value for key, value in state_b.items()}
    return {key: (state_a[key].detach().cpu() + state_b[key].detach().cpu()) / 2.0 for key in state_a}


def expected_abs_gradient_noise(parameter_count: int, steps_a: int, steps_b: int, ns_a: float, ns_b: float, clip: float) -> float:
    return float(parameter_count * (steps_a * ns_a + steps_b * ns_b) * clip * np.sqrt(2.0 / np.pi))


def run_fairdp(data: PreparedData, epsilon: float, uniform_reference: dict[float, float], dirs: dict[str, Path]) -> dict[str, object]:
    seed_everything(SEED)
    epochs = 8
    batch_size = 1024
    clip = 1.0
    mask0 = data.a_train.to_numpy(dtype=int) == 0
    mask1 = data.a_train.to_numpy(dtype=int) == 1

    model0, ns0, spent0, batches0 = train_group_model(data.x_train.loc[mask0].reset_index(drop=True), data.y_train.loc[mask0].reset_index(drop=True), epsilon, epochs, batch_size, clip)
    model1, ns1, spent1, batches1 = train_group_model(data.x_train.loc[mask1].reset_index(drop=True), data.y_train.loc[mask1].reset_index(drop=True), epsilon, epochs, batch_size, clip)

    global_model = FairDPNet(len(data.feature_names))
    global_model.load_state_dict(average_state_dict(model0, model1))
    global_model.eval()
    with torch.no_grad():
        logits = global_model(torch.tensor(data.x_test.to_numpy(dtype=np.float32)))
        task_proba = torch.sigmoid(logits).numpy()
    y_pred = (task_proba >= 0.5).astype(int)
    parameter_count = sum(parameter.numel() for parameter in global_model.parameters() if parameter.requires_grad)
    expected_noise = expected_abs_gradient_noise(parameter_count, epochs * batches0, epochs * batches1, ns0, ns1, clip)
    noise_percent = 100.0 * expected_noise / float(uniform_reference[float(epsilon)])
    torch.save(global_model.state_dict(), dirs["models"] / f"fairdp_gmsc_epsilon_{str(epsilon).replace('.', 'p')}.pt")
    return {
        "privacy_budget": epsilon,
        "method": "FairDP",
        "task_auc": float(roc_auc_score(data.y_test, task_proba)),
        "attack_auc": feature_attack_auc(data),
        "model_score_attack_auc": float(roc_auc_score(data.a_test, task_proba)),
        "dp_gap": compute_dp_gap(y_pred, data.a_test),
        "eo_gap": compute_eo_gap(data.y_test, y_pred, data.a_test),
        "noise_added_percent": noise_percent,
        "expected_fairdp_absolute_noise": expected_noise,
        "noise_multiplier_group0": ns0,
        "noise_multiplier_group1": ns1,
        "spent_epsilon_group0": spent0,
        "spent_epsilon_group1": spent1,
        "perturbed_features": "model gradients by sensitive group",
        "noise_config": json.dumps(
            {
                "type": "FairDP group-wise DP-SGD gradient Gaussian noise",
                "target_epsilon": epsilon,
                "epochs": epochs,
                "batch_size": batch_size,
                "clip": clip,
                "noise_multiplier_group0": ns0,
                "noise_multiplier_group1": ns1,
                "spent_epsilon_group0": spent0,
                "spent_epsilon_group1": spent1,
                "expected_fairdp_absolute_noise": expected_noise,
                "noise_added_percent_reference": "Uniform Perturbation total absolute input noise",
            },
            sort_keys=True,
        ),
    }


def write_tables(results: pd.DataFrame, dirs: dict[str, Path]) -> None:
    results.to_csv(dirs["tables"] / "fairdp_gmsc_results.csv", index=False)
    paper = results[
        [
            "privacy_budget",
            "method",
            "task_auc",
            "attack_auc",
            "dp_gap",
            "eo_gap",
            "noise_added_percent",
            "model_score_attack_auc",
            "noise_multiplier_group0",
            "noise_multiplier_group1",
            "expected_fairdp_absolute_noise",
        ]
    ].copy()
    paper.to_csv(dirs["tables"] / "fairdp_gmsc_results_paper_ready.csv", index=False)
    row = results[np.isclose(results["privacy_budget"], 0.1)].iloc[0]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{FairDP baseline results on Give Me Some Credit at $\epsilon=0.1$. FairDP noise is raw group-wise DP-SGD gradient noise normalized to Uniform Perturbation input-noise mass.}",
        r"\label{tab:fairdp_baseline}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Task AUC & Attack AUC & DP Gap & EO Gap & Noise Added (\%) \\",
        r"\midrule",
        f"FairDP & {row['task_auc']:.3f} & {row['attack_auc']:.3f} & {row['dp_gap']:.3f} & {row['eo_gap']:.3f} & {row['noise_added_percent']:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (dirs["tables"] / "fairdp_gmsc_epsilon_0p1_latex.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dirs = make_dirs(root)
    setup_logging(dirs["logs"] / "fairdp_gmsc_baseline.log")
    seed_everything(SEED)
    data = load_data(ensure_dataset(dirs["data"]))
    uniform_reference = uniform_input_noise_reference(root)
    rows = []
    for epsilon in PRIVACY_BUDGETS:
        logging.info("Running FairDP baseline epsilon=%s", epsilon)
        rows.append(run_fairdp(data, epsilon, uniform_reference, dirs))
    write_tables(pd.DataFrame(rows), dirs)


if __name__ == "__main__":
    main()
