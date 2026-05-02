import math
import torch
from copy import deepcopy
from typing import Dict
from rich.progress import Progress
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from Data.utils import generate_distshift, init_subloader
from Model.utils import init_model, init_optim
from Model.train_eval import tr_fairdp, eval_multi_fn, eval_fn
from Model.core.metrics import (
    fair_metric_multi,
    fair_metric,
    emp_bound,
)
from opacus import PrivacyEngine
from Utils.console import console

Device = torch.device


def run(
    args,
    tr_info: tuple,
    va_info: tuple,
    te_info: tuple,
    name: str,
    history: Dict,
    device: Device,
):

    mtr_loader, ftr_loader = tr_info
    va_loader = va_info
    te_loader, df_te = te_info

    with console.status("Initializing Model") as status:
        model_name = "{}.pt".format(name)
        model_glo = init_model(args=args, clip=False)
        model_mal = init_model(args=args, clip=False)
        model_fem = init_model(args=args, clip=False)
        model_glo = model_glo.to(device)
        obj = torch.nn.BCEWithLogitsLoss(reduction="none").to(device)
        opt_mal = init_optim(opt=args.opt, lr=args.lr, model=model_mal)
        opt_fem = init_optim(opt=args.opt, lr=args.lr, model=model_fem)

        privacy_engine = PrivacyEngine()
        _, mal_optim_, _ = privacy_engine.make_private_with_epsilon(
            module=model_mal,
            optimizer=opt_mal,
            data_loader=deepcopy(mtr_loader),
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            max_grad_norm=args.cgrad,
        )
        _, fem_optim_, _ = privacy_engine.make_private_with_epsilon(
            module=model_fem,
            optimizer=opt_fem,
            data_loader=deepcopy(ftr_loader),
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            max_grad_norm=args.cgrad,
        )
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = {"acc": BinaryAccuracy().to(device), "auc": BinaryAUROC().to(device)}
        args.ns = max(mal_optim_.noise_multiplier, fem_optim_.noise_multiplier)
        console.log(
            f"For Male have to use: sigma={mal_optim_.noise_multiplier}",
            f"For Female have to use: sigma={fem_optim_.noise_multiplier}",
            f"\nTherefore overall need to use: sigma={args.ns}",
        )
        del fem_optim_, mal_optim_
        del opt_fem, opt_mal
        del model_fem, model_mal

        model_mal = deepcopy(model_glo).to(device)
        model_fem = deepcopy(model_glo).to(device)
        opt_mal = init_optim(opt=args.opt, lr=args.lr, model=model_mal)
        opt_fem = init_optim(opt=args.opt, lr=args.lr, model=model_fem)

        console.log(f"Target model's configuration: {model_glo}")
        console.log(
            f"[green]Train / Optimizing model for MALE with optimizer[/green]: {opt_mal}"
        )
        console.log(
            f"[green]Train / Optimizing model for FEMALE with optimizer[/green]: {opt_fem}"
        )
        console.log(f"[green]Train / Objective of the training process[/green]: {obj}")
        console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
        console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")
        console.log(f"Done Initializing Model: :white_check_mark:")

    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)
        global_dict = model_glo.state_dict()

        for epoch in range(args.epochs):

            if epoch < args.epochs - 1:

                l_m, l_f = tr_fairdp(
                    loaders=(mtr_loader, ftr_loader),
                    models=(model_mal, model_fem, model_glo),
                    obj=obj,
                    opts=(opt_mal, opt_fem),
                    device=device,
                    clip=args.cgrad,
                    clay=args.clay,
                    ns=args.ns,
                    get=False,
                )

                va_loss, va_perf = eval_fn(
                    loader=va_loader,
                    model=model_glo,
                    obj=obj,
                    metrics=metrics,
                    pred_fn=pred_fn,
                    device=device,
                )
                te_loss, te_perf = eval_fn(
                    loader=te_loader,
                    model=model_glo,
                    obj=obj,
                    metrics=metrics,
                    pred_fn=pred_fn,
                    device=device,
                )
                demo_p, eq_opp, eq_odd = fair_metric(
                    loader=te_loader, model=model_glo, device=device
                )
                torch.save(model_glo.state_dict(), args.model_path + model_name)
            else:
                # last_lay, num_pt, loss_gen
                grad, bz, loss_gen = tr_fairdp(
                    loaders=(mtr_loader, ftr_loader),
                    models=(model_mal, model_fem, model_glo),
                    obj=obj,
                    opts=(opt_mal, opt_fem),
                    device=device,
                    clip=args.cgrad,
                    ns=args.ns,
                    clay=args.clay,
                    get=True,
                )
                grad_mal, grad_fem = grad
                bz_mal, bz_fem = bz
                l_m, l_f = loss_gen
                with torch.no_grad():

                    glo_models = []

                    model_mal_ = deepcopy(model_mal)
                    model_fem_ = deepcopy(model_fem)
                    model_glo_ = deepcopy(model_glo)
                    mal_dict = model_mal_.state_dict()
                    fem_dict = model_fem_.state_dict()
                    for key in global_dict.keys():
                        global_dict[key] = torch.div(
                            deepcopy(mal_dict[key]) + deepcopy(fem_dict[key]), 2
                        )
                    model_glo_.load_state_dict(global_dict)

                    std = args.cgrad * args.ns * (args.lr * 0.1) / math.sqrt(2)
                    emp_demo, emp_eq, emp_odd = emp_bound(
                        mal_loader=mtr_loader,
                        fem_loader=ftr_loader,
                        model=model_glo_,
                        device=device,
                        std=std,
                        mode="test",
                    )
                    history["emp_demo"] = emp_demo
                    history["emp_eq"] = emp_eq
                    history["emp_odd"] = emp_odd
                    console.log(
                        f"Empirical bound: demo = {emp_demo}, eq = {emp_eq}, odd = {emp_odd}"
                    )
                    del model_glo_
                    del model_mal_
                    del model_fem_

                    bz_m = int(bz_mal / args.n_mo)
                    bz_f = int(bz_fem / args.n_mo)
                    if args.debug_tr:
                        console.log(f"Last step: {bz_m} male, {bz_f} female")

                    with torch.no_grad():
                        for i in range(args.n_mo):
                            model_m = deepcopy(model_mal)
                            model_f = deepcopy(model_fem)

                            saved_var_mal = {}
                            for tensor_name, tensor in model_m.named_parameters():
                                if "out_layer" in tensor_name:
                                    saved_var_mal[tensor_name] = torch.zeros_like(
                                        tensor
                                    ).to(device)

                            for pos, j in enumerate(
                                grad_mal[i * bz_m : (i + 1) * bz_m]
                            ):
                                for tensor_name, tensor in model_m.named_parameters():
                                    if "out_layer" in tensor_name:
                                        saved_var_mal[tensor_name].add_(j)

                            for tensor_name, tensor in model_m.named_parameters():
                                if "out_layer" in tensor_name:
                                    saved_var_mal[tensor_name].add_(
                                        torch.FloatTensor(
                                            saved_var_mal[tensor_name].shape
                                        )
                                        .normal_(0, args.cgrad * args.ns)
                                        .to(device)
                                    )
                                    tensor.data = (
                                        tensor.data
                                        - (args.lr * 0.1) * saved_var_mal[tensor_name]
                                    )

                            saved_var_fem = {}
                            for tensor_name, tensor in model_f.named_parameters():
                                if "out_layer" in tensor_name:
                                    saved_var_fem[tensor_name] = torch.zeros_like(
                                        tensor
                                    ).to(device)

                            for pos, j in enumerate(
                                grad_fem[int(i * bz_f) : int((i + 1) * bz_f)]
                            ):
                                for tensor_name, tensor in model_f.named_parameters():
                                    if "out_layer" in tensor_name:
                                        saved_var_fem[tensor_name].add_(j)

                            for tensor_name, tensor in model_f.named_parameters():
                                if "out_layer" in tensor_name:
                                    saved_var_fem[tensor_name].add_(
                                        torch.FloatTensor(
                                            saved_var_fem[tensor_name].shape
                                        )
                                        .normal_(0, args.cgrad * args.ns)
                                        .to(device)
                                    )
                                    tensor.data = (
                                        tensor.data
                                        - (args.lr * 0.1) * saved_var_fem[tensor_name]
                                    )

                            glo_m = deepcopy(model_glo)
                            mal_dict = model_m.state_dict()
                            fem_dict = model_f.state_dict()
                            for key in global_dict.keys():
                                global_dict[key] = torch.div(
                                    deepcopy(mal_dict[key]) + deepcopy(fem_dict[key]), 2
                                )
                            glo_m.load_state_dict(global_dict)
                            glo_models.append(glo_m)

                va_loss, va_perf = eval_multi_fn(
                    loader=va_loader,
                    models=glo_models,
                    obj=obj,
                    metrics=metrics,
                    device=device,
                    pred_fn=pred_fn,
                )
                te_loss, te_perf = eval_multi_fn(
                    loader=te_loader,
                    models=glo_models,
                    obj=obj,
                    metrics=metrics,
                    device=device,
                    pred_fn=pred_fn,
                )
                demo_p, eq_opp, eq_odd = fair_metric_multi(
                    loader=te_loader, models=glo_models, device=device
                )
                for i, m in enumerate(glo_models):
                    torch.save(m.state_dict(), args.model_path + f"model_{i}_{name}.pt")

            progress.console.log(
                f"Epoch {epoch}: mal_loss {l_m}, fem_loss {l_f}, va_loss {va_loss}, va_perf (acc:{va_perf['acc']}, auc:{va_perf['auc']})"
            )

            for key in metrics.keys():
                history[f"va_{key}"] = va_perf[key]
                history[f"va_{key}"] = va_perf[key]
            history["va_loss"].append(va_loss)
            history["te_loss"].append(te_loss)
            history["dp"].append(demo_p)
            history["eqopp"].append(eq_opp)
            history["eqodd"].append(eq_odd)
            progress.advance(tk_tr)

    va_loss, va_perf = eval_multi_fn(
        loader=va_loader,
        models=glo_models,
        obj=obj,
        metrics=metrics,
        device=device,
        pred_fn=pred_fn,
    )
    te_loss, te_perf = eval_multi_fn(
        loader=te_loader,
        models=glo_models,
        obj=obj,
        metrics=metrics,
        device=device,
        pred_fn=pred_fn,
    )
    demo_p, eq_opp, eq_odd = fair_metric_multi(
        loader=te_loader, models=glo_models, device=device
    )
    history["best_test_acc"] = te_perf["acc"]
    history["best_test_auc"] = te_perf["auc"]
    history["best_dp"] = demo_p
    history["best_eqopp"] = eq_opp
    history["best_eqodd"] = eq_odd
    console.log(
        "=" * 100,
        "\n",
        f"Best result: te_perf {te_perf}, demo {demo_p}, opp {eq_opp}, odd {eq_odd}",
        "\n",
        "=" * 100,
    )

    # eval distribution shift
    gamma, dfs = generate_distshift(df=df_te, att=[args.att_lb, args.att_pr])
    perf_acc, perf_auc, demo, eqopp, eqodd = [], [], [], [], []
    for i, df in enumerate(dfs):
        dtype = "image" if args.data == "utk" else "tabular"
        te_loader = init_subloader(
            df=df,
            target=args.att_lb,
            protect=args.att_pr,
            feat=args.att_fe,
            bste=args.bs,
            type=dtype,
        )
        te_loss, te_perf = eval_multi_fn(
            loader=te_loader,
            models=glo_models,
            obj=obj,
            metrics=metrics,
            device=device,
            pred_fn=pred_fn,
        )
        demo_p, eq_opp, eq_odd = fair_metric_multi(
            loader=te_loader, models=glo_models, device=device
        )
        perf_acc.append(te_perf["acc"])
        perf_auc.append(te_perf["auc"])
        demo.append(demo_p)
        eqopp.append(eq_opp)
        eqodd.append(eq_odd)

    history["gamma"] = gamma
    history["perf_shift_acc"] = perf_acc
    history["perf_shift_auc"] = perf_auc
    history["dp_shift"] = demo
    history["op_shift"] = eqopp
    history["od_shift"] = eqodd

    return history
