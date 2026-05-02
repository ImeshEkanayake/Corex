import torch
from copy import deepcopy
from typing import Dict
from rich.progress import Progress
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from Utils.console import console
from Model.modules.early_stopping import EarlyStopping
from Model.train_eval import tr_smooth, eval_smooth, eval_fn
from Model.core.metrics import fair_metric_sm
from Model.utils import init_model

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
        model_mal = deepcopy(model_glo).to(device)
        model_fem = deepcopy(model_glo).to(device)
        model_glo = model_glo.to(device)
        obj = torch.nn.BCEWithLogitsLoss().to(device)
        optim_mal = torch.optim.Adam(model_mal.parameters(), lr=args.lr)
        optim_fem = torch.optim.Adam(model_fem.parameters(), lr=args.lr)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = {"acc": BinaryAccuracy().to(device), "auc": BinaryAUROC().to(device)}
        es = EarlyStopping(patience=20, verbose=False)

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

    with Progress(console=console) as progress:

        tk_tr = progress.add_task("[red]Training...", total=args.epochs)
        global_dict = model_glo.state_dict()

        for epoch in range(args.epochs):
            model_mal_ = deepcopy(model_mal)
            model_fem_ = deepcopy(model_fem)

            l_m = tr_smooth(
                loader=mtr_loader,
                models=(model_mal, model_fem_),
                obj=obj,
                opt=optim_mal,
                alpha=args.wd,
                device=device,
            )
            l_f = tr_smooth(
                loader=ftr_loader,
                models=(model_fem, model_mal_),
                obj=obj,
                opt=optim_fem,
                alpha=args.wd,
                device=device,
            )

            with torch.no_grad():
                mal_dict = model_mal.state_dict()
                fem_dict = model_fem.state_dict()
                for key in global_dict.keys():
                    global_dict[key] = torch.div(
                        deepcopy(mal_dict[key]) + deepcopy(fem_dict[key]), 2
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

            es(
                epoch=epoch,
                epoch_score=va_perf,
                model=model_glo,
                model_path=args.model_path + model_name,
            )
            progress.advance(tk_tr)

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
