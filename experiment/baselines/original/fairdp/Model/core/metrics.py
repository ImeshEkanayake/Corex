import math
import torch
import numpy as np
from typing import Sequence
from torch.nn import Module
from torch.utils.data import DataLoader
from scipy.special import erf
from Utils.console import console
from Utils.utils import get_index_by_value

Device = torch.device


def fair_metric(
    loader: DataLoader, model: Module, device: Device, threshold: float = 0.5
):
    pred_fn = torch.nn.Sigmoid().to(device)
    model.to(device)
    model.eval()
    mal_pred = torch.Tensor([]).to(device)
    mal_targ = torch.Tensor([]).to(device)
    fem_pred = torch.Tensor([]).to(device)
    fem_targ = torch.Tensor([]).to(device)

    with torch.no_grad():

        for bi, batch in enumerate(loader):

            feat, target, z = batch
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            score = torch.squeeze(score)
            pred = (pred_fn(score) > threshold).int()
            idx_mal = get_index_by_value(a=z, val=1)
            idx_fem = get_index_by_value(a=z, val=0)
            if len(idx_mal) > 0:
                mal_pred = torch.cat([mal_pred, pred[idx_mal]], dim=0)
                mal_targ = torch.cat((mal_targ, target[idx_mal]), dim=0)
            if len(idx_fem) > 0:
                fem_pred = torch.cat([fem_pred, pred[idx_fem]], dim=0)
                fem_targ = torch.cat((fem_targ, target[idx_fem]), dim=0)

    prob_mal = mal_pred.mean().item()
    prob_fem = fem_pred.mean().item()
    demo_p = np.abs(prob_mal - prob_fem)

    confusion_vector = mal_pred / mal_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    mal_tpr = tp / (tp + fn + 1e-12)
    mal_fpr = fp / (fp + tn + 1e-12)

    confusion_vector = fem_pred / fem_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    fem_tpr = tp / (tp + fn + 1e-12)
    fem_fpr = fp / (fp + tn + 1e-12)

    all_targ = torch.cat((mal_targ, fem_targ), dim=0)
    label, counts = all_targ.unique(sorted=True, return_counts=True)
    rate_neg = counts[0].item() / all_targ.size(dim=0)
    rate_pos = 1 - rate_neg
    console.log(
        f"Labels: {label}, with counts: {counts} -> neg rate {rate_neg} and pos rate {rate_pos}"
    )
    eq_opp = np.abs(mal_tpr - fem_tpr)
    eq_odd = np.abs(mal_tpr - fem_tpr) * rate_pos + rate_neg * np.abs(mal_fpr - fem_fpr)
    return demo_p, eq_opp, eq_odd


def fair_metric_sm(
    loader: DataLoader,
    model: Module,
    device: Device,
    num_draw: int,
    noise_scale: float,
    threshold: float = 0.5,
):
    pred_fn = torch.nn.Sigmoid().to(device)
    model.to(device)
    model.eval()
    mal_pred = torch.Tensor([]).to(device)
    mal_targ = torch.Tensor([]).to(device)
    fem_pred = torch.Tensor([]).to(device)
    fem_targ = torch.Tensor([]).to(device)

    with torch.no_grad():
        org_state = model.state_dict()
        for bi, batch in enumerate(loader):
            feat, target, z = batch
            feat = feat.to(device)
            target = target.to(device)
            pred = 0.0
            for i in range(num_draw):
                model.load_state_dict(org_state)
                state_dict = model.state_dict()
                for key in state_dict.keys():
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_scale,
                        size=state_dict[key].size(),
                        requires_grad=False,
                    ).to(device)
                    state_dict[key] = state_dict[key] + noise
                model.load_state_dict(state_dict)
                pred = pred + pred_fn(torch.squeeze(model(feat)))
            pred = pred / num_draw
            pred = (pred > threshold).int()
            idx_mal = get_index_by_value(a=z, val=1)
            idx_fem = get_index_by_value(a=z, val=0)
            if len(idx_mal) > 0:
                mal_pred = torch.cat([mal_pred, pred[idx_mal]], dim=0)
                mal_targ = torch.cat((mal_targ, target[idx_mal]), dim=0)
            if len(idx_fem) > 0:
                fem_pred = torch.cat([fem_pred, pred[idx_fem]], dim=0)
                fem_targ = torch.cat((fem_targ, target[idx_fem]), dim=0)

    prob_mal = mal_pred.mean().item()
    prob_fem = fem_pred.mean().item()
    demo_p = np.abs(prob_mal - prob_fem)

    confusion_vector = mal_pred / mal_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    mal_tpr = tp / (tp + fn + 1e-12)
    mal_fpr = fp / (fp + tn + 1e-12)

    confusion_vector = fem_pred / fem_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    fem_tpr = tp / (tp + fn + 1e-12)
    fem_fpr = fp / (fp + tn + 1e-12)

    all_targ = torch.cat((mal_targ, fem_targ), dim=0)
    label, counts = all_targ.unique(sorted=True, return_counts=True)
    rate_neg = counts[0].item() / all_targ.size(dim=0)
    rate_pos = 1 - rate_neg
    console.log(
        f"Labels: {label}, with counts: {counts} -> neg rate {rate_neg} and pos rate {rate_pos}"
    )
    eq_opp = np.abs(mal_tpr - fem_tpr)
    eq_odd = np.abs(mal_tpr - fem_tpr) * rate_pos + rate_neg * np.abs(mal_fpr - fem_fpr)
    return demo_p, eq_opp, eq_odd


def fair_metric_multi(
    loader: DataLoader,
    models: Sequence[Module],
    device: Device,
    threshold: float = 0.5,
):
    pred_fn = torch.nn.Sigmoid().to(device)
    for model in models:
        model.to(device)
        model.eval()
    mal_pred = torch.Tensor([]).to(device)
    mal_targ = torch.Tensor([]).to(device)
    fem_pred = torch.Tensor([]).to(device)
    fem_targ = torch.Tensor([]).to(device)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            feat, target, z = batch
            feat = feat.to(device)
            target = target.to(device)
            pred = 0.0
            for model in models:
                pred = pred + pred_fn(torch.squeeze(model(feat)))
            pred = pred / len(models)
            pred = (pred > threshold).int()
            idx_mal = get_index_by_value(a=z, val=1)
            idx_fem = get_index_by_value(a=z, val=0)
            if len(idx_mal) > 0:
                mal_pred = torch.cat([mal_pred, pred[idx_mal]], dim=0)
                mal_targ = torch.cat((mal_targ, target[idx_mal]), dim=0)
            if len(idx_fem) > 0:
                fem_pred = torch.cat([fem_pred, pred[idx_fem]], dim=0)
                fem_targ = torch.cat((fem_targ, target[idx_fem]), dim=0)

    prob_mal = mal_pred.mean().item()
    prob_fem = fem_pred.mean().item()
    demo_p = np.abs(prob_mal - prob_fem)

    confusion_vector = mal_pred / mal_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    mal_tpr = tp / (tp + fn + 1e-12)
    mal_fpr = fp / (fp + tn + 1e-12)

    confusion_vector = fem_pred / fem_targ
    tp = torch.sum(confusion_vector == 1).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    fn = torch.sum(confusion_vector == 0).item()
    fem_tpr = tp / (tp + fn + 1e-12)
    fem_fpr = fp / (fp + tn + 1e-12)

    all_targ = torch.cat((mal_targ, fem_targ), dim=0)
    label, counts = all_targ.unique(sorted=True, return_counts=True)
    rate_neg = counts[0].item() / all_targ.size(dim=0)
    rate_pos = 1 - rate_neg
    console.log(
        f"Labels: {label}, with counts: {counts} -> neg rate {rate_neg} and pos rate {rate_pos}"
    )
    eq_opp = np.abs(mal_tpr - fem_tpr)
    eq_odd = np.abs(mal_tpr - fem_tpr) * rate_pos + rate_neg * np.abs(mal_fpr - fem_fpr)
    return demo_p, eq_opp, eq_odd


def emp_bound(mal_loader, fem_loader, model, device, std, alpha=0.05, mode="test"):
    model.to(device)
    mal_prop = 0.0
    fem_prop = 0.0
    mal_prop_pos = 0.0
    fem_prop_pos = 0.0
    mal_prop_neg = 0.0
    fem_prop_neg = 0.0
    num_mal = 1e-12
    num_fem = 1e-12
    num_mal_pos = 1e-12
    num_fem_pos = 1e-12
    num_mal_neg = 1e-12
    num_fem_neg = 1e-12
    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(mal_loader):
            features, target, _ = d
            features = features.to(device, dtype=torch.float)
            x_in, x_out = model(features, mode="eval")
            x_out = torch.squeeze(x_out, dim=-1)
            for i in range(x_in.size(dim=0)):
                x = x_in[i]
                z = x_out[i].cpu().detach().numpy()
                y = target[i].item()
                p = 1 / 2 + 1 / 2 * erf(
                    z / (std * np.sqrt(x.norm(p=2).item() ** 2) * np.sqrt(2) + 1e-10)
                )
                num_mal += 1
                mal_prop += p
                if y == 1:
                    mal_prop_pos += p
                    num_mal_pos += 1
                else:
                    mal_prop_neg += p
                    num_mal_neg += 1

        for bi, d in enumerate(fem_loader):
            features, target, _ = d
            features = features.to(device, dtype=torch.float)
            x_in, x_out = model(features, mode="eval")
            x_out = torch.squeeze(x_out, dim=-1)
            for i in range(x_in.size(dim=0)):
                x = x_in[i]
                z = x_out[i].cpu().detach().numpy()
                y = target[i].item()
                p = 1 / 2 + 1 / 2 * erf(
                    z
                    / (std * np.sqrt(x.norm(p=2).item() ** 2 + 1) * np.sqrt(2) + 1e-10)
                )
                num_fem += 1
                fem_prop += p
                if y == 1:
                    fem_prop_pos += p
                    num_fem_pos += 1
                else:
                    fem_prop_neg += p
                    num_fem_neg += 1

    # add laplace noise
    mal_prop = (mal_prop + np.random.laplace(loc=0.0, scale=10.0)) / num_mal
    fem_prop = (fem_prop + np.random.laplace(loc=0.0, scale=10.0)) / num_fem

    mal_prop_pos = (mal_prop_pos + np.random.laplace(loc=0.0, scale=10.0)) / num_mal_pos
    fem_prop_pos = (fem_prop_pos + np.random.laplace(loc=0.0, scale=10.0)) / num_fem_pos

    mal_prop_neg = (mal_prop_neg + np.random.laplace(loc=0.0, scale=10.0)) / num_mal_neg
    fem_prop_neg = (fem_prop_neg + np.random.laplace(loc=0.0, scale=10.0)) / num_fem_neg

    mal_prop = np.clip(mal_prop, 0, 1)
    fem_prop = np.clip(fem_prop, 0, 1)

    mal_prop_pos = np.clip(mal_prop_pos, 0, 1)
    fem_prop_pos = np.clip(fem_prop_pos, 0, 1)

    mal_prop_neg = np.clip(mal_prop_neg, 0, 1)
    fem_prop_neg = np.clip(fem_prop_neg, 0, 1)

    emp_demo = (
        np.abs(mal_prop - fem_prop)
        + hoeffding_bound(nobs=num_mal, alpha=alpha / 2)
        + hoeffding_bound(nobs=num_fem, alpha=alpha / 2)
    )

    emp_eq = (
        np.abs(mal_prop_pos - fem_prop_pos)
        + hoeffding_bound(nobs=num_mal_pos, alpha=alpha / 2)
        + hoeffding_bound(nobs=num_fem_pos, alpha=alpha / 2)
    )

    emp_odd = (
        1 / 2 * emp_eq
        + 1 / 2 * np.abs(mal_prop_neg - fem_prop_neg)
        + 1 / 2 * hoeffding_bound(nobs=num_mal_neg, alpha=alpha / 2)
        + 1 / 2 * hoeffding_bound(nobs=num_fem_neg, alpha=alpha / 2)
    )

    print("=" * 100)
    print(f"# mal {num_mal}, # fem {num_fem}")
    print(f"# mal pos {num_mal_pos}, # fem pos {num_fem_pos}")
    print(f"# mal neg {num_mal_neg}, # fem neg {num_fem_neg}")
    print("=" * 100)
    return emp_demo, emp_eq, emp_odd


def hoeffding_bound(nobs, alpha, bonferroni_hyp_n=1):
    return math.sqrt(math.log(bonferroni_hyp_n / alpha) / (2 * nobs))
