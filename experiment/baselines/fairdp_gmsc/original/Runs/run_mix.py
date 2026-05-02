# import math
# import torch
# import numpy as np
# from copy import deepcopy
# from typing import Dict
# from rich.progress import Progress
# from torch.linalg import matrix_norm
# from torchmetrics.classification import BinaryAveragePrecision
# from Data.utils import generate_distshift, init_subloader
# from Model.utils import init_model, init_optim
# from Model.train_eval import tr_fairdp, eval_multi_fn, eval_fn, tr_dpsgd
# from Model.core.metrics import demo_parity_multi, eq_opp_odd_multi, emp_bound, demo_parity, eq_opp_odd
# from Utils.console import console

# Device = torch.device

# def run(args, tr_info:tuple, va_info:tuple, te_info:tuple, name:str, history:Dict, device:Device):

#     tr_loader, trmal_loader, trfem_loader = tr_info
#     va_loader = va_info
#     te_loader, temal_loader, tefem_loader, df_te = te_info

#     with console.status("Initializing Model") as status:
#         model_name = '{}.pt'.format(name)
#         model_glo = init_model(args=args, clip=False).to(device)
#         model_mal = init_model(args=args, clip=False).to(device)
#         model_fem = init_model(args=args, clip=False).to(device)
#         obj = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

#         # lr = 1 / (args.cgrad * args.ns * math.sqrt(2))
#         # lr = lr if (lr >= 1e-3) & (lr <= 0.02) else args.lr
#         opt_mal = init_optim(opt=args.opt, lr=args.lr, model=model_mal, wd=args.wd)
#         opt_fem = init_optim(opt=args.opt, lr=args.lr, model=model_fem, wd=args.wd)
#         opt_glo = init_optim(opt=args.opt, lr=args.lr, model=model_glo, wd=args.wd)

#         if args.lr_decay:
#             lr_lmbda = lambda ep: 1 / math.sqrt(ep + 1e-12)
#             m_sche = torch.optim.lr_scheduler.LambdaLR(opt_mal, lr_lambda=lr_lmbda)
#             f_sche = torch.optim.lr_scheduler.LambdaLR(opt_fem, lr_lambda=lr_lmbda)

#         pred_fn = torch.nn.Sigmoid().to(device)
#         metric = BinaryAveragePrecision().to(device)

#         console.log(f"Target model's configuration: {model_glo}")
#         console.log(f"[green]Train / Optimizing model for MALE with optimizer[/green]: {opt_mal}")
#         console.log(f"[green]Train / Optimizing model for FEMALE with optimizer[/green]: {opt_fem}")
#         console.log(f"[green]Train / Objective of the training process[/green]: {obj}")
#         console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
#         console.log(f"[green]Train / Evaluating with metrics[/green]: {metric}")
#         console.log(f"Done Initializing Model: :white_check_mark:")

#     with Progress(console=console) as progress:

#         tk_tr = progress.add_task("[red]Training...", total=args.epochs)
#         global_dict = model_glo.state_dict()

#         for epoch in range(args.epochs):

#             with torch.no_grad():
#                 norm = 0
#                 for p in model_glo.named_parameters():
#                     if 'out_layer' in p[0]:
#                         norm = norm + p[1].data.norm(p=2)
#                 if norm.item() > args.clay:
#                     weight = args.clay / (norm + 1e-12)
#                     for p in model_glo.named_parameters():
#                         if 'out_layer' in p[0]:
#                             p[1].data = p[1].data * weight
#                     del norm
#                     del weight

#                 global_dict = model_glo.state_dict()
#                 model_mal.load_state_dict(global_dict)
#                 model_fem.load_state_dict(global_dict)

#             if epoch < args.epochs - 1:

#                 choice = np.random.choice([0, 1], 1, p=[1 - args.swprop, args.swprop])
#                 if choice[0] == 1:
#                     _, _ = tr_fairdp(loader=trmal_loader, model=model_mal, obj=obj, opt=opt_mal, device=device,
#                                     clip=args.cgrad, ns=args.ns, pred_fn=pred_fn, get=False)
#                     _, _ = tr_fairdp(loader=trfem_loader, model=model_fem, obj=obj, opt=opt_fem, device=device,
#                                     clip=args.cgrad, ns=args.ns, pred_fn=pred_fn, get=False)

#                     with torch.no_grad():
#                         mal_dict = model_mal.state_dict()
#                         fem_dict = model_fem.state_dict()
#                         for key in global_dict.keys():
#                             global_dict[key] = torch.div(mal_dict[key].clone() + fem_dict[key].clone(), 2)
#                         model_glo.load_state_dict(global_dict)

#                     tr_loss, tr_perf = eval_fn(loader=tr_loader, model=model_glo, obj=obj, metric=metric, pred_fn=pred_fn, device=device)
#                 else:
#                     tr_loss, tr_perf = tr_dpsgd(loader=tr_loader, model=model_glo, obj=obj, metric=metric, pred_fn=pred_fn, device=device, clip=args.cgrad, ns=args.ns, opt=opt_glo)

#                 va_loss, va_perf = eval_fn(loader=va_loader, model=model_glo, obj=obj, metric=metric, pred_fn=pred_fn, device=device)
#                 te_loss, te_perf = eval_fn(loader=te_loader, model=model_glo, obj=obj, metric=metric, pred_fn=pred_fn, device=device)
#                 demo_p = demo_parity(mloader=temal_loader, floader=tefem_loader, pred_fn=pred_fn, model=model_glo, device=device)
#                 eq_opp, eq_odd = eq_opp_odd(mloader=temal_loader, floader=tefem_loader, pred_fn=pred_fn, model=model_glo, device=device)
#                 torch.save(model_glo.state_dict(), args.model_path + model_name)
#             else:
#                 with torch.no_grad():
#                     norm = 0
#                     for p in model_mal.named_parameters():
#                         if 'out_layer' in p[0]:
#                             norm = norm + p[1].data.norm(p=2)

#                     if norm.item() > args.clay:
#                         weight = args.clay / (norm + 1e-12)
#                         for p in model_mal.named_parameters():
#                             if 'out_layer' in p[0]:
#                                 p[1].data = p[1].data * weight
#                         del norm
#                         del weight

#                     norm = 0
#                     for p in model_fem.named_parameters():
#                         if 'out_layer' in p[0]:
#                             norm = norm + p[1].data.norm(p=2)
#                     if norm.item() > args.clay:
#                         weight = args.clay / (norm + 1e-12)
#                         for p in model_fem.named_parameters():
#                             if 'out_layer' in p[0]:
#                                 p[1].data = p[1].data * weight
#                         del norm
#                         del weight

#                 grad_mal, bz_mal = tr_fairdp(loader=trmal_loader, model=model_mal, obj=obj, opt=opt_mal, device=device,
#                                             clip=args.cgrad, ns=args.ns, pred_fn=pred_fn, get=True)
#                 grad_fem, bz_fem = tr_fairdp(loader=trfem_loader, model=model_fem, obj=obj, opt=opt_fem, device=device,
#                                             clip=args.cgrad, ns=args.ns, pred_fn=pred_fn, get=True)

#                 if args.debug_tr:
#                     console.log(f"Size of grad_mal: {grad_mal.size()}, Size of grad_fem {grad_fem.size()}")

#                 with torch.no_grad():

#                     glo_models = []

#                     model_mal_ = deepcopy(model_mal)
#                     model_fem_ = deepcopy(model_fem)

#                     model_glo_ = deepcopy(model_glo)
#                     mal_dict = model_mal_.state_dict()
#                     fem_dict = model_fem_.state_dict()
#                     for key in global_dict.keys():
#                         global_dict[key] = torch.div(deepcopy(mal_dict[key]) + deepcopy(fem_dict[key]), 2)
#                     model_glo_.load_state_dict(global_dict)

#                     std = args.cgrad * args.ns * args.lr / math.sqrt(2)

#                     if args.bdset == 'train':
#                         emp_demo, emp_eq, emp_odd = emp_bound(mal_loader=temal_loader, fem_loader=tefem_loader,
#                                                             model=model_glo_, device=device, std=std, mode='train')
#                     else:
#                         emp_demo, emp_eq, emp_odd = emp_bound(mal_loader=temal_loader, fem_loader=tefem_loader,
#                                                             model=model_glo_, device=device, std=std, mode='test')

#                     history['emp_demo'] = emp_demo
#                     history['emp_eq'] = emp_eq
#                     history['emp_odd'] = emp_odd
#                     console.log(f"Empirical bound: demo = {emp_demo}, eq = {emp_eq}, odd = {emp_odd}")
#                     del (model_glo_)
#                     del (model_mal_)
#                     del (model_fem_)

#                     bz_m = int(bz_mal / args.n_mo)
#                     bz_f = int(bz_fem / args.n_mo)
#                     if args.debug_tr:
#                         console.log(f"Last step: {bz_m} male, {bz_f} female")

#                     for i in range(args.n_mo):
#                         model_m = deepcopy(model_mal)
#                         model_f = deepcopy(model_fem)

#                         saved_var_mal = {}
#                         for tensor_name, tensor in model_m.named_parameters():
#                             if 'out_layer' in tensor_name:
#                                 saved_var_mal[tensor_name] = torch.zeros_like(tensor).to(device)

#                         for pos, j in enumerate(grad_mal[i * bz_m:(i + 1) * bz_m]):
#                             for tensor_name, tensor in model_m.named_parameters():
#                                 if 'out_layer' in tensor_name:
#                                     saved_var_mal[tensor_name].add_(j)

#                         for tensor_name, tensor in model_m.named_parameters():
#                             if 'out_layer' in tensor_name:
#                                 saved_var_mal[tensor_name].add_(
#                                     torch.FloatTensor(saved_var_mal[tensor_name].shape).normal_(0, args.cgrad*args.ns).to(device))
#                                 tensor.grad = saved_var_mal[tensor_name]
#                                 tensor.data = tensor.data - args.lr * tensor.grad

#                         saved_var_fem = {}
#                         for tensor_name, tensor in model_f.named_parameters():
#                             if 'out_layer' in tensor_name:
#                                 saved_var_fem[tensor_name] = torch.zeros_like(tensor).to(device)

#                         for pos, j in enumerate(grad_fem[int(i * bz_f):int((i + 1) * bz_f)]):
#                             for tensor_name, tensor in model_f.named_parameters():
#                                 if 'out_layer' in tensor_name:
#                                     saved_var_fem[tensor_name].add_(j)

#                         for tensor_name, tensor in model_f.named_parameters():
#                             if 'out_layer' in tensor_name:
#                                 saved_var_fem[tensor_name].add_(
#                                     torch.FloatTensor(saved_var_fem[tensor_name].shape).normal_(0, args.cgrad*args.ns).to(device))
#                                 tensor.grad = saved_var_fem[tensor_name]
#                                 tensor.data = tensor.data - args.lr * tensor.grad

#                         glo_m = deepcopy(model_glo)
#                         mal_dict = model_m.state_dict()
#                         fem_dict = model_f.state_dict()
#                         for key in global_dict.keys():
#                             global_dict[key] = torch.div(deepcopy(mal_dict[key]) + deepcopy(fem_dict[key]), 2)
#                         glo_m.load_state_dict(global_dict)
#                         glo_models.append(glo_m)

#                     tr_loss, tr_perf = eval_multi_fn(loader=tr_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#                     va_loss, va_perf = eval_multi_fn(loader=va_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#                     te_loss, te_perf = eval_multi_fn(loader=te_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#                     demo_p = demo_parity_multi(mloader=temal_loader, floader=tefem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#                     eq_opp, eq_odd = eq_opp_odd_multi(mloader=temal_loader, floader=tefem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#                     for i, m in enumerate(glo_models):
#                         torch.save(m.state_dict(), args.model_path + f'model_{i}_{name}.pt')

#             if args.lr_decay:
#                 console.log("Stepping with schedule LR")
#                 m_sche.step()
#                 f_sche.step()

#             progress.console.log(f"Epoch {epoch}: tr_loss {tr_loss}, tr_perf {tr_perf}, va_loss {va_loss}, va_perf {va_perf}, te_loss {te_loss}, te_perf {te_perf}, dp {demo_p}, eqopp {eq_opp}, eqodd {eq_odd}")

#             history['tr_loss'].append(tr_loss)
#             history['tr_perf'].append(tr_perf)
#             history['va_loss'].append(va_loss)
#             history['va_perf'].append(va_perf)
#             history['te_loss'].append(te_loss)
#             history['te_perf'].append(te_perf)
#             history['dp'].append(demo_p)
#             history['eqopp'].append(eq_opp)
#             history['eqodd'].append(eq_odd)
#             progress.advance(tk_tr)

#     tr_loss, tr_perf = eval_multi_fn(loader=tr_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#     va_loss, va_perf = eval_multi_fn(loader=va_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#     te_loss, te_perf = eval_multi_fn(loader=te_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#     demo_p = demo_parity_multi(mloader=temal_loader, floader=tefem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#     eq_opp, eq_odd = eq_opp_odd_multi(mloader=temal_loader, floader=tefem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#     history['best_test'] = te_perf
#     history['best_dp'] = demo_p
#     history['best_eqopp'] = eq_opp
#     history['best_eqodd'] = eq_odd
#     console.log("="*100, '\n', f"Best result: te_perf {te_perf}, demo {demo_p}, opp {eq_opp}, odd {eq_odd}", "\n", "=" * 100)

#     # eval distribution shift
#     gamma, dfs = generate_distshift(df=df_te, att=[args.att_lb, args.att_pr])
#     perf, demo, eqopp, eqodd = [], [], [], []
#     for i, df in enumerate(dfs):
#         if args.data == 'utk':
#             te_loader, mal_loader, fem_loader = init_subloader(df=df, target=args.att_lb, protect=args.att_pr, feat=args.att_fe, bste=args.bs, type='image')
#         else:
#             te_loader, mal_loader, fem_loader = init_subloader(df=df, target=args.att_lb, protect=args.att_pr, feat=args.att_fe, bste=args.bs, type='tabular')
#         te_loss, te_perf = eval_multi_fn(loader=te_loader, models=glo_models, obj=obj, metric=metric, device=device, pred_fn=pred_fn)
#         demo_p = demo_parity_multi(mloader=mal_loader, floader=fem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#         eq_opp, eq_odd = eq_opp_odd_multi(mloader=mal_loader, floader=fem_loader, models=glo_models, pred_fn=pred_fn, device=device)
#         perf.append(te_perf)
#         demo.append(demo_p)
#         eqopp.append(eq_opp)
#         eqodd.append(eq_odd)

#     history['gamma'] = gamma
#     history['perf_shift'] = perf
#     history['dp_shift'] = demo
#     history['op_shift'] = eqopp
#     history['od_shift'] = eqodd

#     return history
