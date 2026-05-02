import torch
from copy import deepcopy
from typing import Dict
from rich.progress import Progress
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from Model.utils import init_model
from Model.train_eval import tr_dpsgd_sm, eval_smooth
from Model.core.metrics import fair_metric_sm
from Utils.console import console
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

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
    te_loader = te_info

    with console.status("Initializing Model") as status:
        model_name = "{}.pt".format(name)
        model_glo = init_model(args=args)
        model_glo = ModuleValidator.fix(model_glo)
        model_mal = deepcopy(model_glo).to(device)
        model_fem = deepcopy(model_glo).to(device)
        model_glo = model_glo.to(device)

        obj = torch.nn.BCEWithLogitsLoss().to(device)
        optim_mal = torch.optim.Adam(model_mal.parameters(), lr=args.lr)
        optim_fem = torch.optim.Adam(model_fem.parameters(), lr=args.lr)
        privacy_engine = PrivacyEngine()
        model_mal, optim_mal, mtr_loader = privacy_engine.make_private_with_epsilon(
            module=model_mal,
            optimizer=optim_mal,
            data_loader=mtr_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            max_grad_norm=args.cgrad,
        )
        model_fem, optim_fem, ftr_loader = privacy_engine.make_private_with_epsilon(
            module=model_fem,
            optimizer=optim_fem,
            data_loader=ftr_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=1e-5,
            max_grad_norm=args.cgrad,
        )

        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = {"acc": BinaryAccuracy().to(device), "auc": BinaryAUROC().to(device)}

        console.log(f"Target model's configuration: {model_glo}")
        console.log(
            f"[green]Train / Optimizing model for MALE with optimizer[/green]: {optim_mal}"
        )
        console.log(
            f"[green]Train / Optimizing model for FEMALE with optimizer[/green]: {optim_fem}"
        )
        console.log(f"[green]Train / Objective of the training process[/green]: {obj}")
        console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
        console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")
        console.log(f"Done Initializing Model: :white_check_mark:")

    # History dictionary to store everything
    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)
        global_dict = model_glo.state_dict()

        for epoch in range(args.epochs):
            model_mal_ = deepcopy(model_mal)
            model_fem_ = deepcopy(model_fem)

            l_m = tr_dpsgd_sm(
                loader=mtr_loader,
                models=(model_mal, model_fem_),
                obj=obj,
                opt=optim_mal,
                device=device,
                alpha=args.wd,
                batch_size=args.bs,
            )
            l_f = tr_dpsgd_sm(
                loader=ftr_loader,
                models=(model_fem, model_mal_),
                obj=obj,
                opt=optim_fem,
                device=device,
                alpha=args.wd,
                batch_size=args.bs,
            )

            mal_dict = model_mal.state_dict()
            fem_dict = model_fem.state_dict()
            # console.log(mal_dict)
            # console.log(fem_dict)
            for key in global_dict.keys():
                global_dict[key] = torch.div(
                    deepcopy(mal_dict[f"_module.{key}"])
                    + deepcopy(fem_dict[f"_module.{key}"]),
                    2,
                )
            model_glo.load_state_dict(global_dict)
            va_loss, va_perf = eval_smooth(
                loader=va_loader,
                model=model_glo,
                obj=obj,
                metrics=metrics,
                pred_fn=pred_fn,
                device=device,
                ns_=args.ns_,
                ndraw=args.ndraw,
            )
            te_loss, te_perf = eval_smooth(
                loader=te_loader,
                model=model_glo,
                obj=obj,
                metrics=metrics,
                pred_fn=pred_fn,
                device=device,
                ns_=args.ns_,
                ndraw=args.ndraw,
            )
            demo_p, eq_opp, eq_odd = fair_metric_sm(
                loader=te_loader,
                model=model_glo,
                device=device,
                num_draw=args.ndraw,
                noise_scale=args.ns_,
            )

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
            torch.save(obj=model_glo.state_dict(), f=args.model_path + model_name)

    model_glo.load_state_dict(torch.load(args.model_path + model_name))
    te_loss, te_perf = eval_smooth(
        loader=te_loader,
        model=model_glo,
        obj=obj,
        metrics=metrics,
        pred_fn=pred_fn,
        device=device,
        ns_=args.ns_,
        ndraw=args.ndraw,
    )
    demo_p, eq_opp, eq_odd = fair_metric_sm(
        loader=te_loader,
        model=model_glo,
        device=device,
        num_draw=args.ndraw,
        noise_scale=args.ns_,
    )
    history["best_test_acc"] = te_perf["acc"]
    history["best_test_auc"] = te_perf["auc"]
    history["best_dp"] = demo_p
    history["best_eqopp"] = eq_opp
    history["best_eqodd"] = eq_odd
    console.log(
        "=" * 100,
        "\n",
        f"Best result: te_perf (acc:{te_perf['acc']}, auc:{te_perf['auc']}), demo {demo_p}, opp {eq_opp}, odd {eq_odd}",
        "\n",
        "=" * 100,
    )
    return history
