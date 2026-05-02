from __future__ import annotations

import json
import logging
import shutil
import random
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


class PFairDPNet(nn.Module):
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


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler()],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


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


def ensure_dataset(data_dir: Path) -> Path:
    path = data_dir / "cs-training.csv"
    if not path.exists():
        project_root = Path(__file__).resolve().parents[4]
        candidates = [
            project_root / "Paper Experiemnts" / "give_me_some_credit_corex" / "outputs" / "data" / "cs-training.csv",
            project_root / "Paper Experiemnts" / "give_me_some_credit_table2_privacy_utility" / "outputs" / "data" / "cs-training.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                shutil.copy2(candidate, path)
                logging.info("Copied Give Me Some Credit dataset from %s", candidate)
                return path
        logging.info("Downloading Give Me Some Credit dataset to %s", path)
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
    a_values = a.to_numpy(dtype=int)
    rates = []
    for group in [0, 1]:
        mask = a_values == group
        rates.append(float(np.mean(y_pred[mask])) if mask.any() else np.nan)
    return float(abs(rates[1] - rates[0]))


def compute_eo_gap(y_true: pd.Series, y_pred: np.ndarray, a: pd.Series) -> float:
    y_values = y_true.to_numpy(dtype=int)
    a_values = a.to_numpy(dtype=int)
    rates = []
    for group in [0, 1]:
        mask = (a_values == group) & (y_values == 1)
        if not mask.any():
            return float("nan")
        rates.append(float(np.mean(y_pred[mask])))
    return float(abs(rates[1] - rates[0]))


def feature_attack_auc(data: PreparedData) -> float:
    attack = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
    attack.fit(data.x_train, data.a_train)
    return float(roc_auc_score(data.a_test, attack.predict_proba(data.x_test)[:, 1]))


def uniform_input_noise_reference(root: Path) -> dict[float, float]:
    """Load the Uniform Perturbation absolute-noise denominator from Table 2."""
    table_path = (
        root.parents[1]
        / "give_me_some_credit_table2_privacy_utility"
        / "outputs"
        / "tables"
        / "table2_privacy_utility_optimized_adaptive_with_noise.csv"
    )
    if not table_path.exists():
        logging.warning("Uniform noise reference missing: %s", table_path)
        return {}
    table = pd.read_csv(table_path)
    uniform = table[table["method"] == "Uniform Perturbation"]
    return {
        float(row["privacy_budget"]): float(row["total_absolute_noise"])
        for _, row in uniform.iterrows()
    }


def expected_dp_sgd_absolute_noise(model: nn.Module, noise_multiplier: float, max_grad_norm: float, batch_size: int, steps: int) -> float:
    """Expected absolute Gaussian noise injected into DP-SGD gradients.

    PFairDP uses Opacus DP-SGD. Opacus clips per-sample gradients and adds
    Gaussian noise with std = noise_multiplier * max_grad_norm to the summed
    gradients before the optimizer update is normalized. For comparison with
    feature perturbation, report the raw in-process noise mass, not the
    batch-averaged optimizer-update noise.
    """
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    raw_gradient_std = float(noise_multiplier) * float(max_grad_norm)
    expected_abs_per_parameter = raw_gradient_std * float(np.sqrt(2.0 / np.pi))
    return float(parameter_count * steps * expected_abs_per_parameter)


def train_pfairdp(data: PreparedData, epsilon: float, dirs: dict[str, Path], uniform_reference: dict[float, float]) -> dict[str, object]:
    seed_everything(SEED)
    device = torch.device("cpu")
    epochs = 8
    batch_size = 1024
    delta = 1e-5
    max_grad_norm = 1.0
    fairness_lambda = 0.20

    x_tensor = torch.tensor(data.x_train.to_numpy(dtype=np.float32))
    y_tensor = torch.tensor(data.y_train.to_numpy(dtype=np.float32))
    a_tensor = torch.tensor(data.a_train.to_numpy(dtype=np.float32))
    dataset = TensorDataset(x_tensor, y_tensor, a_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = PFairDPNet(len(data.feature_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    privacy_engine = PrivacyEngine()
    fallback_noise_multiplier = None
    try:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=epochs,
            target_epsilon=float(epsilon),
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
    except ValueError as exc:
        fallback_noise_multiplier = max(25.0, 5.0 / float(epsilon))
        logging.warning(
            "Opacus could not target epsilon=%s exactly (%s). Falling back to explicit noise_multiplier=%s.",
            epsilon,
            exc,
            fallback_noise_multiplier,
        )
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=fallback_noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    criterion = nn.BCEWithLogitsLoss()
    dp_sgd_noise_multiplier = float(getattr(optimizer, "noise_multiplier", fallback_noise_multiplier or 0.0))
    steps = epochs * len(loader)

    model.train()
    for epoch in range(epochs):
        losses = []
        for xb, yb, ab in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ab = ab.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            pred = torch.sigmoid(logits)
            bce = criterion(logits, yb)
            mask0 = ab < 0.5
            mask1 = ab >= 0.5
            if bool(mask0.any()) and bool(mask1.any()):
                fairness_penalty = torch.abs(pred[mask0].mean() - pred[mask1].mean())
            else:
                fairness_penalty = torch.tensor(0.0, device=device)
            loss = bce + fairness_lambda * fairness_penalty
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logging.info("epsilon=%s epoch=%s mean_loss=%.6f", epsilon, epoch + 1, float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(data.x_test.to_numpy(dtype=np.float32), device=device)
        task_proba = torch.sigmoid(model(x_test)).cpu().numpy()
    y_pred = (task_proba >= 0.5).astype(int)
    task_auc = float(roc_auc_score(data.y_test, task_proba))
    model_score_attack_auc = float(roc_auc_score(data.a_test, task_proba))
    spent_epsilon = float(privacy_engine.get_epsilon(delta))
    expected_training_noise = expected_dp_sgd_absolute_noise(
        model,
        dp_sgd_noise_multiplier,
        max_grad_norm,
        batch_size,
        steps,
    )
    denominator = float(uniform_reference.get(float(epsilon), 0.0))
    noise_added_percent = 0.0 if denominator <= 0 else 100.0 * expected_training_noise / denominator
    torch.save(model.state_dict(), dirs["models"] / f"pfairdp_gmsc_epsilon_{str(epsilon).replace('.', 'p')}.pt")
    return {
        "privacy_budget": epsilon,
        "method": "PFairDP",
        "task_auc": task_auc,
        "attack_auc": feature_attack_auc(data),
        "model_score_attack_auc": model_score_attack_auc,
        "dp_gap": compute_dp_gap(y_pred, data.a_test),
        "eo_gap": compute_eo_gap(data.y_test, y_pred, data.a_test),
        "noise_added_percent": noise_added_percent,
        "expected_dp_sgd_absolute_noise": expected_training_noise,
        "dp_sgd_noise_multiplier": dp_sgd_noise_multiplier,
        "spent_epsilon": spent_epsilon,
        "perturbed_features": "model gradients",
        "noise_config": json.dumps(
            {
                "type": "DP-SGD model-gradient Gaussian noise",
                "target_epsilon": epsilon,
                "spent_epsilon": spent_epsilon,
                "delta": delta,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_grad_norm": max_grad_norm,
                "noise_multiplier": dp_sgd_noise_multiplier,
                "training_steps": steps,
                "expected_dp_sgd_absolute_noise": expected_training_noise,
                "noise_added_percent_reference": "Uniform Perturbation total absolute input noise; PFairDP value uses raw in-process DP-SGD gradient Gaussian noise",
                "fairness_lambda": fairness_lambda,
                "input_feature_noise": False,
                "fallback_noise_multiplier": fallback_noise_multiplier,
            },
            sort_keys=True,
        ),
    }


def latex_for_epsilon(df: pd.DataFrame, epsilon: float) -> str:
    row = df[np.isclose(df["privacy_budget"], epsilon)].iloc[0]
    noise_value = float(row["noise_added_percent"])
    noise_display = "<0.1" if 0.0 < noise_value < 0.1 else f"{noise_value:.1f}"
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{PFairDP baseline results on Give Me Some Credit at $\epsilon={epsilon}$. Higher task AUC is better; lower attack AUC, DP gap, EO gap, and noise added are better.}}",
            r"\label{tab:pfairdp_baseline}",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & Task AUC & Attack AUC & DP Gap & EO Gap & Noise Added (\%) \\",
            r"\midrule",
            f"PFairDP & {row['task_auc']:.3f} & {row['attack_auc']:.3f} & {row['dp_gap']:.3f} & {row['eo_gap']:.3f} & {noise_display} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dirs = make_dirs(root)
    setup_logging(dirs["logs"] / "pfairdp_gmsc_baseline.log")
    seed_everything(SEED)

    data = load_data(ensure_dataset(dirs["data"]))
    logging.info("Prepared Give Me Some Credit data: train=%s test=%s features=%s", len(data.x_train), len(data.x_test), data.feature_names)

    uniform_reference = uniform_input_noise_reference(root)
    rows = []
    for epsilon in PRIVACY_BUDGETS:
        logging.info("Running PFairDP baseline epsilon=%s", epsilon)
        rows.append(train_pfairdp(data, epsilon, dirs, uniform_reference))

    results = pd.DataFrame(rows)
    results.to_csv(dirs["tables"] / "pfairdp_gmsc_results.csv", index=False)
    paper = results[
        [
            "privacy_budget",
            "method",
            "task_auc",
            "attack_auc",
            "dp_gap",
            "eo_gap",
            "noise_added_percent",
            "spent_epsilon",
            "model_score_attack_auc",
            "dp_sgd_noise_multiplier",
            "expected_dp_sgd_absolute_noise",
        ]
    ].copy()
    paper.to_csv(dirs["tables"] / "pfairdp_gmsc_results_paper_ready.csv", index=False)
    (dirs["tables"] / "pfairdp_gmsc_epsilon_0p1_latex.txt").write_text(latex_for_epsilon(results, 0.1), encoding="utf-8")
    logging.info("Wrote PFairDP baseline outputs to %s", dirs["tables"])


if __name__ == "__main__":
    main()
