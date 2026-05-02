import torch
import numpy as np
from copy import deepcopy
from typing import Dict
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.nn import Module
from Model.core.helper import get_grad_vnorm

Device = torch.device


def forward_fairdp(
    batch: tuple,
    model: Module,
    device: Device,
    opt: Optimizer,
    obj: Module,
    clip: float,
    ns: float,
    get: bool = False,
):

    feat, target, _ = batch
    feat = feat.to(device)
    target = target.to(device)
    opt.zero_grad()
    score = model(feat)
    score = torch.squeeze(score, dim=-1)
    loss = obj(score, target)
    num_pt = feat.size(dim=0)

    if get == False:

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name].add_(
                torch.FloatTensor(saved_var[tensor_name].shape)
                .normal_(0, clip * ns)
                .to(device)
            )
            tensor.grad = saved_var[tensor_name] / num_pt

        opt.step()

        return loss.sum().item(), feat.size(dim=0)
    else:
        last_lay = None
        saved_var = {}

        for name, tensor in model.named_parameters():
            saved_var[name] = torch.zeros_like(tensor).to(device)

        for pos, j in enumerate(loss):
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    if "out_layer" not in tensor_name:
                        new_grad = tensor.grad.clone()
                        saved_var[tensor_name].add_(new_grad)
                    else:
                        if pos == 0:
                            last_lay = tensor.grad.detach().clone().unsqueeze(dim=0)
                        else:
                            last_lay = torch.cat(
                                (
                                    last_lay,
                                    tensor.grad.detach().clone().unsqueeze(dim=0),
                                ),
                                dim=0,
                            )
            model.zero_grad()

        for name, tensor in model.named_parameters():
            if "out_layer" not in tensor_name:
                saved_var[name].add_(
                    torch.FloatTensor(saved_var[tensor_name].shape)
                    .normal_(0, clip * ns)
                    .to(device)
                )
                tensor.grad = saved_var[name] / num_pt
            else:
                tensor.grad = torch.zeros_like(tensor).to(device)

        opt.step()
        return last_lay, loss.sum().item(), num_pt


# def forward_dpsgd(
#     model: Module,
#     batch: tuple,
#     device: Device,
#     metric: Metric,
#     opt: Optimizer,
#     obj: Module,
#     pred_fn: Module,
#     clip: float,
#     ns: float,
#     l2: torch.Tensor = None,
# ):

#     feat, target, _ = batch
#     feat = feat.to(device, dtype=torch.float)
#     target = target.to(device, dtype=torch.float)
#     score = model(feat)
#     score = torch.squeeze(score)
#     loss = obj(score, target)
#     num_pt = feat.size(dim=0)

#     saved_var = dict()
#     for tensor_name, tensor in model.named_parameters():
#         saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

#     for pos, j in enumerate(loss):
#         j.backward(retain_graph=True)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         for tensor_name, tensor in model.named_parameters():
#             if tensor.grad is not None:
#                 new_grad = tensor.grad.clone()
#                 saved_var[tensor_name].add_(new_grad)
#         model.zero_grad()

#     for tensor_name, tensor in model.named_parameters():
#         saved_var[tensor_name].add_(
#             torch.FloatTensor(saved_var[tensor_name].shape)
#             .normal_(0, ns * clip)
#             .to(device)
#         )
#         tensor.grad = saved_var[tensor_name] / num_pt

#     if l2 is not None:
#         l2.backward()

#     opt.step()
#     pred = pred_fn(score.detach())
#     metric.update(pred, target.int())

#     return loss.mean().item(), feat.size(dim=0)


# def forward_func(
#     args,
#     model: Module,
#     model_: Module,
#     coff: tuple,
#     Q: np.ndarray,
#     Q_: np.ndarray,
#     noise: torch.Tensor,
# ):

#     if args.gsmode == "func":
#         coff_0, coff_1, coff_2 = coff
#         Q = torch.from_numpy(Q)
#         loss = (
#             coff_0
#             + torch.mm(coff_1.T, torch.mm(Q.T, model))
#             + torch.mm(torch.mm(torch.mm(Q.T, model).T, coff_2), torch.mm(Q.T, model))
#         )
#         model.retain_grad()
#         loss.backward()
#     elif args.gsmode == "funcog":
#         coff_0, coff_1, coff_2 = coff
#         Q = torch.from_numpy(Q)
#         # print(coff_2.size(), coff_1.size(), Q.size(), model.size())
#         loss = (
#             coff_0
#             + torch.mm(coff_1.T, torch.mm(Q.T, model))
#             + torch.mm(torch.mm(torch.mm(Q.T, model).T, coff_2), torch.mm(Q.T, model))
#         )
#         model.retain_grad()
#         loss.backward()
#     elif args.gsmode == "funcsm":
#         coff_0, coff_1, coff_2 = coff
#         Q = torch.from_numpy(Q)
#         # torch.mm(torch.mm(torch.mm(Q.T, model).T, coff_2), torch.mm(Q.T, model))
#         loss = (1 / args.ndraw) * (
#             coff_0
#             + torch.mm(coff_1.T, torch.mm(Q.T, model + noise))
#             + torch.mm(
#                 torch.mm(torch.mm(Q.T, model + noise).T, coff_2),
#                 torch.mm(Q.T, model + noise),
#             )
#         ) + (1 / args.ndraw) * 0.01 * torch.norm(model - model_, p=2)
#         model.retain_grad()
#         loss.backward()
#     return loss.item()


# def forward_clean_grad(
#     model: Module,
#     batch: tuple,
#     device: Device,
#     metric: Metric,
#     opt: Optimizer,
#     obj: Module,
#     pred_fn: Module,
# ):

#     feat, target, protect = batch
#     feat = feat.to(device, dtype=torch.float)
#     num_pt = feat.size(dim=0)
#     target = target.to(device, dtype=torch.float)
#     score = model(feat)
#     score = torch.squeeze(score)
#     loss = obj(score, target) / num_pt
#     loss.backward()

#     upd_grad = dict()

#     for named, tensor in model.named_parameters():
#         if tensor.grad is not None:
#             new_grad = tensor.grad.clone().detach()
#             upd_grad[named] = new_grad

#     opt.step()
#     pred = pred_fn(score.detach())
#     metric.update(pred, target.int())

#     grad, norm = get_grad_vnorm(grad_dict=upd_grad, device=device)
#     grad_dict = {
#         "grad": grad,
#         "norm": norm,
#     }

#     return loss.mean().item(), feat.size(dim=0), grad_dict


# def forward_fairdp_grad(
#     batch: tuple,
#     model: Module,
#     device: Device,
#     opt: Optimizer,
#     obj: Module,
#     clip: float,
#     ns: float,
#     pred_fn: Module,
#     get: bool = False,
# ):

#     feat, target, _ = batch
#     feat = feat.to(device)
#     target = target.to(device)

#     opt.zero_grad()
#     model.zero_grad()

#     score = model(feat)
#     score = torch.squeeze(score, dim=-1)
#     loss = obj(score, target)
#     num_pt = feat.size(dim=0)

#     if get == False:

#         grad_dict = dict()
#         for tensor_name, tensor in model.named_parameters():
#             grad_dict[tensor_name] = torch.zeros_like(tensor).to(device)

#         for pos, j in enumerate(loss):
#             j.backward(retain_graph=True)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             for tensor_name, tensor in model.named_parameters():
#                 if tensor.grad is not None:
#                     new_grad = tensor.grad
#                     grad_dict[tensor_name].add_(new_grad)
#             model.zero_grad()

#         for tensor_name, tensor in model.named_parameters():
#             noise = (
#                 torch.FloatTensor(grad_dict[tensor_name].shape)
#                 .normal_(0, clip * ns)
#                 .to(device)
#             )
#             grad_dict[tensor_name] = grad_dict[tensor_name] + noise
#             tensor.grad = grad_dict[tensor_name]

#         opt.step()

#         grad_v, grad_n = get_grad_vnorm(grad_dict=grad_dict, device=device)

#         grad = {
#             "grad": grad_v,
#             "norm": grad_n,
#         }
#         return loss.mean().item(), num_pt, grad
#     else:
#         pass


# def forward_fairdp_getcmodel(
#     batch: tuple,
#     model: Module,
#     cmodel: Module,
#     device: Device,
#     opt: Optimizer,
#     obj: Module,
#     clip: float,
#     lr: float,
#     ns: float,
#     pred_fn: Module,
#     get: bool = False,
# ):

#     feat, target, _ = batch
#     feat = feat.to(device)
#     target = target.to(device)

#     opt.zero_grad()
#     model.zero_grad()
#     cmodel.zero_grad()

#     score = model(feat)
#     score = torch.squeeze(score, dim=-1)
#     loss = obj(score, target)
#     num_pt = feat.size(dim=0)

#     if get == False:

#         grad_clip = dict()
#         grad_noise = dict()
#         for tensor_name, tensor in model.named_parameters():
#             grad_clip[tensor_name] = torch.zeros_like(tensor).to(device)
#             grad_noise[tensor_name] = torch.zeros_like(tensor).to(device)

#         for pos, j in enumerate(loss):
#             j.backward(retain_graph=True)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             for tensor_name, tensor in model.named_parameters():
#                 if tensor.grad is not None:
#                     new_grad = tensor.grad
#                     grad_clip[tensor_name].add_(new_grad)
#             model.zero_grad()

#         for tensor_name, tensor in model.named_parameters():
#             noise = (
#                 torch.FloatTensor(grad_clip[tensor_name].shape)
#                 .normal_(0, clip * ns)
#                 .to(device)
#             )
#             grad_noise[tensor_name] = grad_clip[tensor_name] + noise
#             tensor.data = tensor.data - lr * grad_noise[tensor_name] / num_pt

#         for tensor_name, tensor in cmodel.named_parameters():
#             tensor.data = tensor.data - lr * grad_clip[tensor_name] / num_pt

#         return loss.mean().item(), num_pt, cmodel
#     else:

#         last_lay = []
#         saved_var = {}

#         for name, tensor in model.named_parameters():
#             saved_var[name] = torch.zeros_like(tensor).to(device)

#         for pos, j in enumerate(loss):
#             j.backward(retain_graph=True)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             for tensor_name, tensor in model.named_parameters():
#                 if tensor.grad is not None:
#                     if "out_layer" not in tensor_name:
#                         new_grad = tensor.grad.clone()
#                         saved_var[tensor_name].add_(new_grad)
#                     else:
#                         last_lay.append(tensor.grad)
#             model.zero_grad()

#         for name, tensor in model.named_parameters():
#             if "out_layer" not in tensor_name:
#                 saved_var[name].add_(
#                     torch.FloatTensor(saved_var[name].shape)
#                     .normal_(0, clip * ns)
#                     .to(device)
#                 )
#                 tensor.grad = saved_var[name] / num_pt
#             else:
#                 tensor.grad = torch.zeros_like(tensor).to(device)

#         opt.step()
#         return last_lay, num_pt
