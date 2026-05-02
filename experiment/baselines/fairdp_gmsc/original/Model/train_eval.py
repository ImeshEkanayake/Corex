import torch
import numpy as np
from copy import deepcopy
from typing import Union, Sequence
from sklearn.metrics import log_loss, average_precision_score, confusion_matrix
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from collections import defaultdict
from typing import Sequence, Dict
from Utils.console import console
from Model.core.helper import get_grad_vec
from opacus.utils.batch_memory_manager import BatchMemoryManager
from Model.core.forward import forward_fairdp

Device = torch.device

# train


def tr_clean(
    loader: DataLoader,
    model: Module,
    obj: Module,
    opt: Optimizer,
    metrics: Dict,
    pred_fn: Module,
    device: torch.device,
):

    model.to(device)
    model.train()

    tr_loss = 0
    num_data = 0

    for bi, batch in enumerate(loader):
        model.zero_grad()
        opt.zero_grad()
        feat, target, _ = batch
        feat = feat.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        score = model(feat)
        score = torch.squeeze(score)
        loss = obj(score, target)
        loss.backward()
        opt.step()
        pred = pred_fn(score.detach())
        for key in metrics.keys():
            metrics[key].update(pred, target.int())
        tr_loss += loss.item() * feat.size(dim=0)
        num_data += feat.size(dim=0)

    tr_loss = tr_loss / num_data
    tr_perf = {}
    for key in metrics.keys():
        tr_perf[key] = metrics[key].compute().item()
        metrics[key].reset()

    return tr_loss, tr_perf


def tr_smooth(
    loader: DataLoader,
    models: Sequence[Module],
    obj: Module,
    opt: Optimizer,
    alpha: float,
    device: torch.device,
):

    model_tr, model_ref = models
    model_tr.to(device)
    model_ref.to(device)
    model_tr.train()
    for n, p in model_tr.named_parameters():
        p.requires_grad = True
    for n, p in model_ref.named_parameters():
        p.requires_grad = False
    tr_loss = 0
    num_pt = 0
    param = model_ref.state_dict()
    for bi, batch in enumerate(loader):
        model_tr.zero_grad()
        feat, target, _ = batch
        feat = feat.to(device)
        target = target.to(device)
        score = model_tr(feat)
        score = torch.squeeze(score)
        l2_norm = 0.0
        for n, p in model_tr.named_parameters():
            l2_norm += torch.norm(p - param[n], p=2)

        obj_loss = obj(score, target)
        loss = obj_loss + alpha * l2_norm / 2
        loss.backward()
        opt.step()
        tr_loss += obj_loss.item()
        num_pt += feat.size(dim=0)
    return loss / num_pt


def tr_dpsgd(
    loader: DataLoader,
    model: Module,
    obj: Module,
    opt: Optimizer,
    metrics: Dict,
    pred_fn: Module,
    device: torch.device,
    batch_size: int,
):

    model.to(device)
    model.train()
    tr_loss = 0
    num_pt = 0
    with BatchMemoryManager(
        data_loader=loader, max_physical_batch_size=batch_size, optimizer=opt
    ) as memory_safe_data_loader:

        for bi, batch in enumerate(memory_safe_data_loader):
            opt.zero_grad()
            feat, target, _ = batch
            feat = feat.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            score = model(feat)
            score = torch.squeeze(score)
            loss = obj(score, target)
            loss.backward()
            opt.step()
            tr_loss += loss.mean().item() * feat.size(dim=0)
            num_pt += feat.size(dim=0)

            pred = pred_fn(score.detach())
            for key in metrics.keys():
                metrics[key].update(pred, target.int())
    tr_perf = {}
    for key in metrics.keys():
        tr_perf[key] = metrics[key].compute().item()
        metrics[key].reset()
    tr_loss = tr_loss / num_pt
    return tr_loss, tr_perf


def tr_remove(
    loader: DataLoader,
    model: Module,
    obj: Module,
    opt: Optimizer,
    metrics: Dict,
    pred_fn: Module,
    device: Device,
    ns: float,
    clip: float,
):
    model.train()
    running_loss = 0.0
    ssum = 0
    num_pt = 0
    info = dict()
    num_classes = 2
    sigma = ns
    num_microbatches = len(loader)
    # print(len(dataloader))
    info["cosine"] = np.zeros((num_classes, num_microbatches))
    info["distance"] = np.zeros((num_classes, num_microbatches))
    info["norm"] = np.zeros((num_classes, num_microbatches))
    info["scale"] = np.zeros((num_classes, num_microbatches))
    info["loss"] = np.zeros((num_classes, num_microbatches))
    info["count"] = np.zeros((num_classes, num_microbatches))
    for i, data in enumerate(loader, 0):
        # DPSGD-F or Naive. To compare with DPSGD under the same privacy budget, the algorithm runs a few less iterations in the last epoch
        # if epoch == args.epochs-1 and i > 213 - 213*args.epochs*0.01/6.56:
        #     print('early termination')
        #     print(i)
        #     break

        inputs, labels, ismale = data
        # print(labels.size())
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights_tensor = labels.int().bincount()
        weights = weights_tensor.cpu()
        opt.zero_grad()

        scores = model(inputs)
        scores = torch.squeeze(scores)
        losses = obj(scores, labels)
        pred = pred_fn(scores)
        for key in metrics.keys():
            metrics[key].update(pred, labels.int())
        running_loss += torch.mean(losses).item() * inputs.size(dim=0)
        num_pt += inputs.size(dim=0)

        # losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)
        # losses = deepcopy(loss)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)
        grad_vecs = dict()
        count_vecs = np.zeros(num_classes)
        batch_loss = defaultdict(int)
        for pos, j in enumerate(losses):
            j.backward(retain_graph=True)
            grad_vec = get_grad_vec(model, device)
            # console.log(f"processed iter {pos}: {grad_vec.size()}")
            label = labels[pos].item()
            if torch.norm(grad_vec).item() > clip:
                count_vecs[int(label)] += 1
            batch_loss[int(label)] += j
            if grad_vecs.get(label, False) is not False:
                grad_vecs[int(label)].add_(grad_vec)
            else:
                grad_vecs[int(label)] = grad_vec
            model.zero_grad()

        sigma_k = np.zeros(num_classes)
        S_k = np.zeros(num_classes)
        theta_k = np.zeros(num_classes)

        # DPSGD-F
        above_threshold = count_vecs  # + np.random.normal(0, 10*sigma, num_classes)
        above_threshold = np.where(
            above_threshold < 0, 0, above_threshold.astype("int")
        )
        below_threshold = (
            np.array(weights) - count_vecs
        )  # + np.random.normal(0, 10*sigma, num_classes)
        below_threshold = np.where(
            below_threshold < 0, 0, below_threshold.astype("int")
        )
        noisy_weights = above_threshold + below_threshold
        noisy_count_vecs = np.where(
            noisy_weights == 0, 1, above_threshold / (noisy_weights + 1e-12)
        )

        # # Naive
        # noisy_weights = np.array(weights) + np.random.normal(0, 10 * sigma, num_classes)
        # noisy_weights = np.where(noisy_weights < 1, 1, noisy_weights.astype('int'))
        # print(noisy_count_vecs)

        for k, vec in sorted(grad_vecs.items(), key=lambda t: t[0]):
            # DPSGD-F
            S_k[int(k)] = clip * (
                1 + (noisy_count_vecs[int(k)]) / (np.mean(noisy_count_vecs) + 1e-12)
            )
            theta_k[int(k)] = 1

            ## Naive
            # S_k[k] = S
            # theta_k[k] = batch_size / num_classes / noisy_weights[k]

            sigma_k[int(k)] = S_k[int(k)] * theta_k[int(k)]

        console.log(S_k)
        mean_grad = np.mean(S_k)
        sigma_new = sigma * (max(sigma_k)) / (mean_grad + 1e-12)
        for pos, j in enumerate(losses):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), S_k[int(labels[pos])])
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    saved_var[tensor_name].add_(
                        new_grad * sigma_k[int(labels[pos])] / mean_grad
                    )
            model.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            # if tensor.grad is not None:
            if device.type == "cuda":
                saved_var[tensor_name].add_(
                    torch.cuda.FloatTensor(tensor.shape).normal_(0, sigma_new)
                )
            else:
                saved_var[tensor_name].add_(
                    torch.FloatTensor(tensor.shape).normal_(0, sigma_new)
                )
            tensor.grad = saved_var[tensor_name] / num_microbatches

        total_grad_vec = get_grad_vec(model, device)
        # print(sorted(grad_vecs.items(), key=lambda t: t[0]))
        for k, vec in sorted(grad_vecs.items(), key=lambda t: t[0]):
            # print(int(k), i)
            vec = vec / weights[int(k)]
            cosine = torch.cosine_similarity(total_grad_vec, vec, dim=-1)
            distance = torch.norm(total_grad_vec - vec)

            info["cosine"][int(k), i] = cosine.item()
            info["distance"][int(k), i] = distance.item()
            info["norm"][int(k), i] = S_k[int(k)]
            info["scale"][int(k), i] = sigma_k[int(k)]
            info["loss"][int(k), i] = batch_loss[int(k)].item() / weights[int(k)].item()
            info["count"][int(k), i] = weights[int(k)].item()
        opt.step()

    tr_perf = {}
    for key in metrics.keys():
        tr_perf[key] = metrics[key].compute().item()
        metrics[key].reset()

    return running_loss / num_pt, tr_perf


def tr_dpsgd_sm(
    loader: DataLoader,
    models: Sequence[Module],
    obj: Module,
    opt: Optimizer,
    device: Device,
    alpha: float,
    batch_size: int,
):

    model_tr, model_ref = models
    model_tr.to(device)
    model_ref.to(device)
    for n, p in model_tr.named_parameters():
        p.requires_grad = True
    for n, p in model_ref.named_parameters():
        p.requires_grad = False
    tr_loss = 0
    num_pt = 0
    param = model_ref.state_dict()

    with BatchMemoryManager(
        data_loader=loader, max_physical_batch_size=batch_size, optimizer=opt
    ) as memory_safe_data_loader:

        for bi, batch in enumerate(memory_safe_data_loader):
            opt.zero_grad()
            feat, target, _ = batch
            feat = feat.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            score = model_tr(feat)
            score = torch.squeeze(score)
            obj_loss = obj(score, target)
            l2_norm = 0.0
            for p in model_tr.named_parameters():
                l2_norm += torch.norm(p[1] - param[p[0]], p=2)
            l2_norm = alpha / 2 * l2_norm
            loss = obj_loss + l2_norm
            loss.backward()
            opt.step()
            num_pt += feat.size(dim=0)
            tr_loss += obj_loss.mean().item() * feat.size(dim=0)

    return tr_loss / num_pt


def tr_fairdp(
    loaders: Sequence[DataLoader],
    models: Sequence[Module],
    obj: Module,
    opts: Sequence[Optimizer],
    device: Device,
    clip: float,
    clay: float,
    ns: float,
    get: bool = False,
):

    mal_loader, fem_loader = loaders
    mal_model, fem_model, glo_model = models
    mal_model.to(device)
    fem_model.to(device)
    opt_mal, opt_fem = opts
    # model.train()

    if get == False:
        num_step = min(len(mal_loader), len(fem_loader))
        mal_iter = iter(mal_loader)
        fem_iter = iter(fem_loader)
        mal_loss = 0
        mal_numpt = 0
        fem_loss = 0
        fem_numpt = 0
        for step in range(num_step):
            with torch.no_grad():
                norm = 0
                for p in glo_model.named_parameters():
                    if "out_layer" in p[0]:
                        norm = norm + p[1].data.norm(p=2)
                if norm.item() > clay:
                    weight = clay / (norm + 1e-12)
                    for p in glo_model.named_parameters():
                        if "out_layer" in p[0]:
                            p[1].data = p[1].data * weight
                    del norm
                    del weight
                global_dict = glo_model.state_dict()
                mal_model.load_state_dict(global_dict)
                fem_model.load_state_dict(global_dict)

            # mal update
            opt_mal.zero_grad()
            mal_batch = next(mal_iter)
            loss_m, npt_m = forward_fairdp(
                batch=mal_batch,
                model=mal_model,
                device=device,
                opt=opt_mal,
                obj=obj,
                clip=clip,
                ns=ns,
                get=get,
            )
            mal_loss += loss_m
            mal_numpt += npt_m

            # fem update
            opt_fem.zero_grad()
            fem_batch = next(fem_iter)
            loss_f, npt_f = forward_fairdp(
                batch=fem_batch,
                model=fem_model,
                device=device,
                opt=opt_fem,
                obj=obj,
                clip=clip,
                ns=ns,
                get=get,
            )
            fem_loss += loss_f
            fem_numpt += npt_f

            with torch.no_grad():
                mal_dict = mal_model.state_dict()
                fem_dict = fem_model.state_dict()
                for key in global_dict.keys():
                    global_dict[key] = torch.div(
                        mal_dict[key].clone() + fem_dict[key].clone(), 2
                    )
                glo_model.load_state_dict(global_dict)
        return mal_loss / mal_numpt, fem_loss / fem_numpt
    else:
        num_step = min(len(mal_loader), len(fem_loader))
        mal_iter = iter(mal_loader)
        fem_iter = iter(fem_loader)
        mal_loss = 0
        mal_numpt = 0
        fem_loss = 0
        fem_numpt = 0
        for step in range(num_step):
            with torch.no_grad():
                norm = 0
                for p in glo_model.named_parameters():
                    if "out_layer" in p[0]:
                        norm = norm + p[1].data.norm(p=2)
                if norm.item() > clay:
                    weight = clay / (norm + 1e-12)
                    for p in glo_model.named_parameters():
                        if "out_layer" in p[0]:
                            p[1].data = p[1].data * weight
                    del norm
                    del weight
                global_dict = glo_model.state_dict()
                mal_model.load_state_dict(global_dict)
                fem_model.load_state_dict(global_dict)
            if step < num_step - 1:
                # mal update
                opt_mal.zero_grad()
                mal_batch = next(mal_iter)
                loss_m, npt_m = forward_fairdp(
                    batch=mal_batch,
                    model=mal_model,
                    device=device,
                    opt=opt_mal,
                    obj=obj,
                    clip=clip,
                    ns=ns,
                    get=False,
                )
                mal_loss += loss_m
                mal_numpt += npt_m

                # fem update
                opt_fem.zero_grad()
                fem_batch = next(fem_iter)
                loss_f, npt_f = forward_fairdp(
                    batch=fem_batch,
                    model=fem_model,
                    device=device,
                    opt=opt_fem,
                    obj=obj,
                    clip=clip,
                    ns=ns,
                    get=False,
                )
                fem_loss += loss_f
                fem_numpt += npt_f

                with torch.no_grad():
                    mal_dict = mal_model.state_dict()
                    fem_dict = fem_model.state_dict()
                    for key in global_dict.keys():
                        global_dict[key] = torch.div(
                            mal_dict[key].clone() + fem_dict[key].clone(), 2
                        )
                    glo_model.load_state_dict(global_dict)
            else:
                opt_mal.zero_grad()
                mal_batch = next(mal_iter)
                last_lay_mal, loss_m, npt_m = forward_fairdp(
                    batch=mal_batch,
                    model=mal_model,
                    device=device,
                    opt=opt_mal,
                    obj=obj,
                    clip=clip,
                    ns=ns,
                    get=True,
                )
                mal_loss += loss_m
                mal_numpt += npt_m

                # fem update
                opt_fem.zero_grad()
                fem_batch = next(fem_iter)
                last_lay_fem, loss_f, npt_f = forward_fairdp(
                    batch=fem_batch,
                    model=fem_model,
                    device=device,
                    opt=opt_fem,
                    obj=obj,
                    clip=clip,
                    ns=ns,
                    get=True,
                )
                fem_loss += loss_f
                fem_numpt += npt_f
                last_lay = (last_lay_mal, last_lay_fem)
                num_pt = (npt_m, npt_f)
                loss_gen = (mal_loss / mal_numpt, fem_loss / fem_numpt)
        return last_lay, num_pt, loss_gen


# eval
def eval_fn(
    loader: DataLoader,
    model: Module,
    obj: Module,
    metrics: Dict,
    device: torch.device,
    pred_fn: Module,
):
    model.to(device)
    avg_loss = 0
    num_pt = 0
    model.eval()
    with torch.no_grad():

        for bi, batch in enumerate(loader):
            feat, target, _ = batch
            feat = feat.to(device)
            target = target.to(device)
            score = model(feat)
            score = torch.squeeze(score, dim=-1)
            loss = obj(score, target).mean()
            avg_loss += loss.item() * feat.size(dim=0)
            num_pt += feat.size(dim=0)
            pred = pred_fn(score)
            for key in metrics.keys():
                metrics[key].update(pred, target.int())

    avg_loss = avg_loss / num_pt
    perf = {}
    for key in metrics.keys():
        perf[key] = metrics[key].compute().item()
        metrics[key].reset()

    return avg_loss, perf


def eval_smooth(
    loader: DataLoader,
    model: Module,
    obj: Module,
    metrics: Dict,
    device: torch.device,
    pred_fn: Module,
    ns_: float,
    ndraw: int,
):
    model.to(device)
    avg_loss = 0
    num_data = 0
    model.eval()
    org_state = model.state_dict()

    with torch.no_grad():
        for bi, batch in enumerate(loader):

            feat, target, _ = batch
            feat = feat.to(device)
            target = target.to(device)
            score = 0.0
            for i in range(ndraw):
                model.load_state_dict(org_state)
                state_dict = model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key] + torch.normal(
                        mean=0.0,
                        std=ns_,
                        size=state_dict[key].size(),
                        requires_grad=False,
                    ).to(device)
                model.load_state_dict(state_dict)
                score = score + model(feat)
            score = score / ndraw
            score = torch.squeeze(score, dim=-1)
            loss = obj(score, target)
            avg_loss += loss.mean().item() * feat.size(dim=0)
            num_data += feat.size(dim=0)
            pred = pred_fn(score)
            for key in metrics.keys():
                metrics[key].update(pred, target.int())

    avg_loss = avg_loss / num_data
    perf = {}
    for key in metrics.keys():
        perf[key] = metrics[key].compute().item()
        metrics[key].reset()
    return avg_loss, perf


def eval_multi_fn(
    loader: DataLoader,
    models: Sequence[Module],
    obj: Module,
    metrics: Dict,
    device: Device,
    pred_fn: Module,
):

    for model in models:
        model.to(device)
        model.eval()

    avg_loss = 0
    num_data = 0

    with torch.no_grad():

        for bi, batch in enumerate(loader):
            feat, target, _ = batch
            feat = feat.to(device)
            target = target.to(device)
            for j, model in enumerate(models):
                if j == 0:
                    score = model(feat)
                else:
                    score = score + model(feat)
            score = torch.squeeze(score, dim=-1) / len(models)
            loss = obj(score, target).mean()
            avg_loss += loss.item() * feat.size(dim=0)
            num_data += feat.size(dim=0)
            pred = pred_fn(score)
            for key in metrics.keys():
                metrics[key].update(pred, target.int())

        perf = {}
        for key in metrics.keys():
            perf[key] = metrics[key].compute().item()
            metrics[key].reset()
    return avg_loss / num_data, perf


# def eval_func(
#     args,
#     model: torch.Tensor,
#     noise: torch.Tensor,
#     X: np.ndarray,
#     y: np.ndarray,
#     fair: bool = False,
# ):

#     if args.gsmode == "func" or args.gsmode == "funcog":
#         feat = torch.from_numpy(X.astype(np.float32))
#         pred = torch.sigmoid(torch.mm(feat, model))
#         loss = log_loss(y_true=y, y_pred=pred.detach().numpy())  # y=y, pred=
#         perf = average_precision_score(
#             y_true=y, y_score=pred.detach().numpy()
#         )  # args=args, y_true=y, y_pred=pred.detach().tolist()
#         if fair:
#             tn, fp, fn, tp = confusion_matrix(
#                 y, np.round(pred.detach().numpy())
#             ).ravel()
#             tpr = tp / (tp + fn)
#             fpr = fp / (fp + tn)
#             prob = np.sum(np.round(pred.detach().numpy())) / pred.shape[0]
#             return tpr, fpr, prob
#         else:
#             return perf, loss
#     else:
#         noise_m, noise_f = noise
#         X_train, X_valid, X_test, X_mal, X_fem = X
#         y_train, y_valid, y_test, y_mal, y_fem = y
#         pred_tr, pred_va, pred_te, pred_m, pred_f, pred_mg, pred_fg = (
#             0.0 for i in range(7)
#         )
#         model_m, model_f, model_g = model
#         for i in range(args.ndraw):
#             temp_g = model_g + (1 / 2) * (noise_m[i] + noise_f[i])
#             temp_m = model_m + noise_m[i]
#             temp_f = model_f + noise_f[i]
#             pred_tr = pred_tr + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_train.astype(np.float32)), temp_g)
#             )
#             pred_va = pred_va + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_valid.astype(np.float32)), temp_g)
#             )
#             pred_te = pred_te + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_test.astype(np.float32)), temp_g)
#             )
#             pred_mg = pred_mg + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_g)
#             )
#             pred_fg = pred_fg + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_g)
#             )
#             pred_m = pred_m + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_m)
#             )
#             pred_f = pred_f + torch.sigmoid(
#                 torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_f)
#             )
#         pred_tr = (1 / args.ndraw) * pred_tr
#         pred_va = (1 / args.ndraw) * pred_va
#         pred_te = (1 / args.ndraw) * pred_te
#         pred_m = (1 / args.ndraw) * pred_m
#         pred_f = (1 / args.ndraw) * pred_f
#         pred_mg = (1 / args.ndraw) * pred_mg
#         pred_fg = (1 / args.ndraw) * pred_fg

#         perf_tr = average_precision_score(
#             y_true=y_train, y_score=pred_tr.detach().tolist()
#         )
#         perf_va = average_precision_score(
#             y_true=y_valid, y_score=pred_va.detach().tolist()
#         )
#         perf_te = average_precision_score(
#             y_true=y_test, y_score=pred_te.detach().tolist()
#         )

#         mal_perf = average_precision_score(
#             y_true=y_mal, y_score=pred_mg.detach().tolist()
#         )
#         fem_perf = average_precision_score(
#             y_true=y_fem, y_score=pred_fg.detach().tolist()
#         )

#         loss_tr = log_loss(y_true=y_train, y_pred=pred_tr.detach().numpy())
#         loss_va = log_loss(y_true=y_valid, y_pred=pred_va.detach().numpy())
#         loss_te = log_loss(y_true=y_test, y_pred=pred_te.detach().numpy())

#         tn, fp, fn, tp = confusion_matrix(
#             y_mal, np.round(pred_mg.detach().numpy())
#         ).ravel()
#         male_tpr = tp / (tp + fn)
#         male_fpr = fp / (fp + tn)
#         male_prob = np.sum(np.round(pred_mg.detach().numpy())) / pred_mg.shape[0]

#         tn, fp, fn, tp = confusion_matrix(
#             y_fem, np.round(pred_fg.detach().numpy())
#         ).ravel()
#         female_tpr = tp / (tp + fn)
#         female_fpr = fp / (fp + tn)
#         female_prob = np.sum(np.round(pred_fg.detach().numpy())) / pred_fg.shape[0]
#         return (
#             (perf_tr, perf_va, perf_te),
#             (loss_tr, loss_va, loss_te),
#             (male_tpr, female_tpr),
#             (male_fpr, female_fpr),
#             (male_prob, female_prob),
#         )


# def tr_clean_grad(loaders:Sequence[DataLoader], model:Module, obj:Module, opt:Optimizer,
#                   metric:Metric, pred_fn:Module, device:torch.device):

#     model.to(device)
#     model.train()
#     m_loader, f_loader, loader = loaders

#     tr_loss = 0
#     num_data = 0


#     for bi, batch in enumerate(loader):

#         # get male & female grad
#         grad_m = get_grad_full(loader=m_loader, model=model, obj=obj, device=device)
#         grad_f = get_grad_full(loader=f_loader, model=model, obj=obj, device=device)

#         grad_m, norm_m = get_grad_vnorm(grad_dict=grad_m, device=device)
#         grad_f, norm_f = get_grad_vnorm(grad_dict=grad_f, device=device)

#         cos = (grad_m * grad_f).sum() / (grad_m.norm(p=2) * grad_f.norm(p=2) + 1e-12)

#         model.zero_grad()
#         opt.zero_grad()
#         loss, n, grad_dict = forward_clean_grad(model=model, batch=batch, device=device, metric=metric,
#                                 obj=obj, opt=opt, pred_fn=pred_fn)

#         cos_m = (grad_m * grad_dict['grad']).sum() / (grad_m.norm(p=2) * grad_dict['grad'].norm(p=2) + 1e-12)
#         cos_f = (grad_f * grad_dict['grad']).sum() / (grad_f.norm(p=2) * grad_dict['grad'].norm(p=2) + 1e-12)

#         results = {
#             "Grad diff": abs(norm_m - norm_f),
#             "Cosine": cos.item(),
#             "Cosine male": cos_m.item(),
#             "Cosine female": cos_f.item(),
#         }
#         tracker_log(dct=results)
#         tr_loss += loss * n
#         num_data += n

#     tr_loss = tr_loss / num_data
#     tr_perf = metric.compute()
#     metric.reset()

#     return tr_loss, tr_perf.item()
