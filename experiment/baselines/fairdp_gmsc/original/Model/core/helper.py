import torch
import numpy as np
from typing import Dict
from copy import deepcopy
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score, \
                            precision_score, average_precision_score, confusion_matrix


Device = torch.device

def get_grad_vec(model:Module, device:Device):
    size = 0
    for name, layer in model.named_parameters():
        if name == 'decoder.weight':
            continue
        size += layer.view(-1).shape[0]
    if device.type == 'cpu':
        sum_var = torch.FloatTensor(size).fill_(0)
    else:
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        if name == 'decoder.weight':
            continue
        sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)
        size += layer.view(-1).shape[0]

    return sum_var

def get_dist(args, batch, grad, model, model_, criterion, device):
    if args.submode == 'w_loss':
        loss, loss_ = get_loss(batch=batch, model=model_, global_model=model, criterion=criterion, device=device)
        return np.abs(loss - loss_)
    elif args.submode == 'w_grad':
        grad_ = get_gradient_from_batch(batch=batch, model=model, criterion=criterion, device=device)
        return gradient_dist(model=model, grad=grad, grad_=grad_)
    elif args.submode == 'w_grad_dp':
        grad_ = get_gradient_from_batch_dp(batch=batch, model=model, criterion=criterion, device=device)
        return gradient_dist(model=model, grad=grad, grad_=grad_)

def get_loss(batch, model, global_model, criterion, device):
    global_model.to(device)
    model.to(device)
    group_loss = 0
    global_loss = 0
    num_data_point = 0
    model.eval()
    global_model.eval()
    with torch.no_grad():
        features, target, _ = batch
        num_pt = features.size(dim=0)
        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        outputs = model(features)
        outputs = torch.squeeze(outputs, dim=-1)
        loss_eval = criterion(outputs, target)
        group_loss += loss_eval.item() * num_pt
        outputs = global_model(features)
        outputs = torch.squeeze(outputs, dim=-1)
        loss_eval = criterion(outputs, target)
        global_loss += loss_eval.item() * num_pt
        num_data_point += num_pt
    return group_loss / num_pt, global_loss / num_pt

def get_gradient_from_batch(batch, model, criterion, device):
    model.to(device)
    model.train()
    features, target, _ = batch
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    model.zero_grad()
    for p in model.named_parameters():
        p[1].grad = torch.zeros_like(p[1])
    temp_par = {}
    output = model(features)
    output = torch.squeeze(output)
    loss = criterion(output, target)
    loss.backward()
    for p in model.named_parameters():
        temp_par[p[0]] = p[1].grad.detach().clone()
    return temp_par

def get_gradient_from_batch_dp(batch, model, criterion, device, clipping, noise_scale):
    model.to(device)
    model.train()
    noise_std = clipping * noise_scale
    features, target, _ = batch
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    temp_par = {}
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
    bz = features.size(dim=0)
    for i in range(bz):
        for p in model.named_parameters():
            p[1].grad = torch.zeros_like(p[1])
        feat = torch.unsqueeze(features[i], 0)
        targ = target[i]
        output = model(feat)
        output = torch.squeeze(output)
        loss = criterion(output, targ)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping, norm_type=2)
        for p in model.named_parameters():
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
    for p in model.named_parameters():
        temp_par[p[0]] = (temp_par[p[0]] + torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(device)) / bz
    return temp_par

def gradient_dist(model, grad, grad_):
    total_grad_norm = 0.0
    for p in model.named_parameters():
        total_grad_norm += (grad[p[0]] - grad_[p[0]]).norm(p=1)
    return total_grad_norm

def performace_eval(met:str, y_true:np.ndarray, y_pred:np.ndarray):
    if met == 'acc':
        return accuracy_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif met == 'f1':
        return f1_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif met == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif met == 'pre':
        return precision_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif met == 'ap':
        return average_precision_score(y_true=y_true, y_score=y_pred)

def fair_evaluate(args, model, noise, X, y, fair=False):
    
    if args.submode == 'func' or args.submode == 'torch' or args.submode == 'func_org':
        feat = torch.from_numpy(X.astype(np.float32))
        pred = torch.sigmoid(torch.mm(feat, model))
        loss = log_loss(y_true=y, y_pred=pred.detach().numpy())
        acc = performace_eval(args=args, y_true=y, y_pred=pred.detach().tolist())
        if fair:
            tn, fp, fn, tp = confusion_matrix(y, np.round(pred.detach().numpy())).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            prob = np.sum(np.round(pred.detach().numpy())) / pred.shape[0]
            return acc, loss, pred, tpr, fpr, prob
        else:
            return acc, loss, pred
    else:
        noise_m, noise_f = noise
        X_train, X_valid, X_test, X_mal, X_fem = X
        y_train, y_valid, y_test, y_mal, y_fem = y
        pred_tr, pred_va, pred_te, pred_m, pred_f, pred_mg, pred_fg = (0.0 for i in range(7))
        model_m, model_f, model_g = model
        for i in range(args.num_draws):
            temp_g = model_g + (1 / 2) * (noise_m[i] + noise_f[i])
            temp_m = model_m + noise_m[i]
            temp_f = model_f + noise_f[i]
            pred_tr = pred_tr + torch.sigmoid(torch.mm(torch.from_numpy(X_train.astype(np.float32)), temp_g))
            pred_va = pred_va + torch.sigmoid(torch.mm(torch.from_numpy(X_valid.astype(np.float32)), temp_g))
            pred_te = pred_te + torch.sigmoid(torch.mm(torch.from_numpy(X_test.astype(np.float32)), temp_g))
            pred_mg = pred_mg + torch.sigmoid(torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_g))
            pred_fg = pred_fg + torch.sigmoid(torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_g))
            pred_m = pred_m + torch.sigmoid(torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_m))
            pred_f = pred_f + torch.sigmoid(torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_f))
        pred_tr = (1 / args.num_draws) * pred_tr
        pred_va = (1 / args.num_draws) * pred_va
        pred_te = (1 / args.num_draws) * pred_te
        pred_m = (1 / args.num_draws) * pred_m
        pred_f = (1 / args.num_draws) * pred_f
        pred_mg = (1 / args.num_draws) * pred_mg
        pred_fg = (1 / args.num_draws) * pred_fg

        acc_tr = performace_eval(args=args, y_true=y_train, y_pred=pred_tr.detach().tolist())
        acc_va = performace_eval(args=args, y_true=y_valid, y_pred=pred_va.detach().tolist())
        acc_te = performace_eval(args=args, y_true=y_test, y_pred=pred_te.detach().tolist())

        male_acc = performace_eval(args=args, y_true=y_mal, y_pred=pred_mg.detach().tolist())
        female_acc = performace_eval(args=args, y_true=y_fem, y_pred=pred_fg.detach().tolist())

        loss_tr = log_loss(y=y_train, pred=pred_tr.detach().numpy())
        loss_va = log_loss(y=y_valid, pred=pred_va.detach().numpy())
        loss_te = log_loss(y=y_test, pred=pred_te.detach().numpy())

        tn, fp, fn, tp = confusion_matrix(y_mal, np.round(pred_mg.detach().numpy())).ravel()
        male_tpr = tp / (tp + fn)
        male_fpr = fp / (fp + tn)
        male_prob = np.sum(np.round(pred_mg.detach().numpy())) / pred_mg.shape[0]

        tn, fp, fn, tp = confusion_matrix(y_fem, np.round(pred_fg.detach().numpy())).ravel()
        female_tpr = tp / (tp + fn)
        female_fpr = fp / (fp + tn)
        female_prob = np.sum(np.round(pred_fg.detach().numpy())) / pred_fg.shape[0]

        male_norm = torch.norm(pred_mg - pred_m, p=2).item() / pred_m.size(0)
        female_norm = torch.norm(pred_fg - pred_f, p=2).item() / pred_f.size(0)
        norm = (male_norm, female_norm)
        return (acc_tr, acc_va, acc_te, male_acc, female_acc), (loss_tr, loss_va, loss_te), \
            (pred_tr, pred_va, pred_te, pred_m, pred_f), (male_tpr, female_tpr), \
            (male_fpr, female_fpr), (male_prob, female_prob), norm

def get_coefficient(X:np.ndarray, y:np.ndarray, epsilon:float=None, lbda:float=None):
    num_data_point = X.shape[0]
    num_feat = X.shape[1]
    sensitivity = num_feat ** 2 / 4 + num_feat
    coff_0 = 1.0
    coff_1 = np.sum(X / 2 - X * y, axis=0).astype(np.float32)
    coff_2 = (1 / 8) * np.dot(X.T, X).astype(np.float32)
    noise_1 = np.random.laplace(0.0, sensitivity / epsilon, coff_1.shape).astype(np.float32)
    noise_2 = np.random.laplace(0.0, sensitivity / epsilon, coff_2.shape).astype(np.float32)
    coff_1 = coff_1 + noise_1
    coff_2 = coff_2 + noise_2
    coff_2 = 1 / 2 + (coff_2 + coff_2.T)
    coff_2 = coff_2 + 5 * np.sqrt(2) * sensitivity * (1 / epsilon) * np.eye(num_feat)
    w, V = np.linalg.eig(coff_2)
    indx = np.where(w > 1e-8)[0]
    w = w[indx].astype(np.float32)
    V = V[:, indx].astype(np.float32)
    coff_2 = np.diag(w)
    coff_1 = np.dot(V.T, coff_1).astype(np.float32)
    return coff_0, (1 / num_data_point) * torch.from_numpy(coff_1.reshape(-1, 1)), (
            1 / num_data_point) * torch.from_numpy(coff_2), V,

def get_grad_vnorm(grad_dict:Dict, device:Device):
    norm = 0.0
    grad_vec = torch.Tensor([]).to(device)
    for key in grad_dict.keys():
        norm = norm + grad_dict[key].norm(p=2)**2
        grad_vec = torch.cat((grad_vec, grad_dict[key].flatten()), dim=0)
    return grad_vec, norm.sqrt().item()

def get_grad_full(loader:DataLoader, model:Module, obj:Module, device:torch.device):

    model.zero_grad()
    total_loss = 0.0
    num_pt = 0

    for bi, batch in enumerate(loader):
        feat, target, _ = batch
        feat = feat.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        score = model(feat)
        score = torch.squeeze(score)
        loss = obj(score, target)
        total_loss = total_loss + loss
        num_pt += feat.size(dim=0)

    total_loss = total_loss.sum() / num_pt
    total_loss.backward()

    upd_grad = dict()
    for named, tensor in model.named_parameters():
        if tensor.grad is not None:
            new_grad = tensor.grad.clone().detach()
            upd_grad[named] = new_grad

    model.zero_grad()
    return upd_grad
