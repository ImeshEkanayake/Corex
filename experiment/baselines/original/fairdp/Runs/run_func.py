# import torch
# import numpy as np
# from typing import Dict
# from rich.progress import Progress
# from Model.core.helper import get_coefficient
# from Model.core.forward import forward_func
# from Model.train_eval import eval_func
# from Utils.console import console

# def run(args, tr_info:tuple, va_info:tuple, te_info:tuple, history:Dict, name:str, device:torch.device = None):
#     model_name = '{}.pt'.format(name)
#     X_train, X_mal, X_fem, y_train, y_mal, y_fem = tr_info
#     X_valid, X_mal_val, X_fem_val, y_valid, y_mal_val, y_fem_val = va_info
#     X_test, X_mal_te, X_fem_te, y_test, y_mal_te, y_fem_te = te_info

#     if args.gsmode == 'func' or args.gsmode == 'funcsm':
#         f_coff_0, f_coff_1, f_coff_2, Q_f = get_coefficient(X=X_fem, y=y_fem, epsilon=args.eps, lbda=args.lamda)
#         m_coff_0, m_coff_1, m_coff_2, Q_m = get_coefficient(X=X_mal, y=y_mal, epsilon=args.eps, lbda=args.lamda)
#     elif args.gsmode == 'funcog':
#         g_coff_0, g_coff_1, g_coff_2, Q_g = get_coefficient(X=X_train, y=y_train, epsilon=args.eps, lbda=args.lamda)

#     model_mal = torch.randn((len(args.feature), 1), requires_grad=True).float()
#     model_fem = torch.randn((len(args.feature), 1), requires_grad=True).float()
#     if args.gsmode == 'funcog':
#         global_model = torch.randn((len(args.feature), 1), requires_grad=True).float()

#     # training process
#     i = 0
#     noise_mal = []
#     noise_fem = []

#     with Progress(console=console) as progress:
#         tk_tr = progress.add_task("[red]Training...", total=args.epochs)
#         for epoch in range(args.epochs):
#             loss_mal = 0
#             loss_fem = 0
#             if args.gsmode == 'funcsm':
#                 for i in range(args.ndraw):
#                     noise_m = torch.normal(0, args.ns_, size=model_mal.size(), requires_grad=False).float()
#                     noise_f = torch.normal(0, args.ns_, size=model_fem.size(), requires_grad=False).float()
#                     noise_mal.append(noise_m)
#                     noise_fem.append(noise_f)
#                     loss_m = forward_func(args=args, model=model_mal, model_=model_fem,
#                                             coff=(m_coff_0, m_coff_1, m_coff_2),
#                                             Q=Q_m, Q_=Q_f, noise=noise_m)
#                     loss_f = forward_func(args=args, model=model_fem, model_=model_mal,
#                                             coff=(f_coff_0, f_coff_1, f_coff_2),
#                                             Q=Q_f, Q_=Q_m, noise=noise_f)
#                     loss_mal += loss_m
#                     loss_fem += loss_f
#             elif args.gsmode == 'func':
#                 loss_m = forward_func(args=args, model=model_mal, model_=model_fem,
#                                         coff=(m_coff_0, m_coff_1, m_coff_2),
#                                         Q=Q_m, Q_=None, noise=None)
#                 loss_f = forward_func(args=args, model=model_fem, model_=model_mal,
#                                         coff=(f_coff_0, f_coff_1, f_coff_2),
#                                         Q=Q_f, Q_=None, noise=None)
#                 loss_mal = loss_m
#                 loss_fem = loss_f
#             elif args.gsmode == 'funcog':
#                 loss = forward_func(args=args, model=global_model, model_=None,
#                                     coff=(g_coff_0, g_coff_1, g_coff_2),
#                                     Q=Q_g, Q_=None, noise=None)

#             if args.gsmode != 'funcog':
#                 model_mal = model_mal - args.lr * model_mal.grad
#                 model_fem = model_fem - args.lr * model_fem.grad
#                 global_model = (model_mal + model_fem) / 2
#             else:
#                 global_model = global_model - args.lr * global_model.grad


#             if args.gsmode == 'func' or args.gsmode == 'funcog':
#                 tr_perf, tr_loss = eval_func(args=args, model=global_model, noise=None, X=X_train, y=y_train)
#                 va_perf, va_loss = eval_func(args=args, model=global_model, noise=None, X=X_valid, y=y_valid)
#                 te_perf, te_loss = eval_func(args=args, model=global_model, noise=None, X=X_test, y=y_test)
#                 mal_tpr, mal_fpr, mal_prob = eval_func(args=args, model=global_model, noise=None, X=X_mal_te, y=y_mal_te, fair=True)
#                 fem_tpr, fem_fpr, fem_prob = eval_func(args=args, model=global_model, noise=None, X=X_fem_te, y=y_fem_te, fair=True)

#             else:
#                 acc, loss, tpr, fpr, prob = eval_func(args=args, model=(global_model, model_mal, model_fem),
#                                                             noise=(noise_mal, noise_fem),
#                                                             X=(X_train, X_valid, X_test, X_mal_te, X_fem_te),
#                                                             y=(y_train, y_valid, y_test, y_mal_te, y_fem_te))
#                 tr_perf, va_perf, te_perf= acc
#                 tr_loss, va_loss, te_loss = loss
#                 mal_tpr, fem_tpr = tpr
#                 mal_fpr, fem_fpr = fpr
#                 mal_prob, fem_prob = prob

#             model_mal.grad = torch.zeros(model_mal.size())
#             model_fem.grad = torch.zeros(model_fem.size())
#             if args.gsmode == 'funcog':
#                 global_model.grad = torch.zeros(global_model.size())

#             demo_p = np.abs(mal_prob - fem_prob)
#             eq_opp = np.abs(mal_tpr - fem_tpr)

#             val, wei = np.unique(ar=y_test, return_counts=True)
#             neg_wei = wei[0] / np.sum(wei)
#             pos_wei = 1 - neg_wei
#             eq_odd = np.abs(mal_tpr - fem_tpr) * pos_wei + np.abs(mal_fpr - fem_fpr) * neg_wei


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

#             torch.save(global_model, args.model_path + model_name)
#             torch.save(model_mal, args.model_path + 'male_{}'.format(model_name))
#             torch.save(model_fem, args.model_path + 'female_{}'.format(model_name))
#             if args.gsmode == 'funcog':
#                 if tr_loss < 0:
#                     break

#             if loss_mal < 0 or loss_fem < 0:
#                 break

#     return history
