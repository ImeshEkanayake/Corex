import torch
from rich.progress import Progress
from typing import Dict
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from Utils.console import console
from Model.modules.early_stopping import EarlyStopping
from Model.train_eval import tr_clean, eval_fn
from Model.core.metrics import fair_metric
from Model.utils import init_model, init_optim

Device = torch.device


def run(
    args,
    tr_info: tuple,
    va_info: tuple,
    te_info: tuple,
    history: Dict,
    name: str,
    device: Device,
):

    tr_loader = tr_info
    va_loader = va_info
    te_loader = te_info

    with console.status("Initializing Model") as status:
        model_name = "{}.pt".format(name)
        model = init_model(args=args).to(device)
        obj = torch.nn.BCEWithLogitsLoss().to(device)
        optim = init_optim(opt=args.opt, lr=args.lr, model=model)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = {"acc": BinaryAccuracy().to(device), "auc": BinaryAUROC().to(device)}
        es = EarlyStopping(patience=5, verbose=False)

        console.log(f"Target model's configuration: {model}")
        console.log(f"[green]Train / Optimizing model with optimizer[/green]: {optim}")
        console.log(f"[green]Train / Objective of the training process[/green]: {obj}")
        console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
        console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")
        console.log(f"Done Initializing Model: :white_check_mark:")

    with Progress(console=console) as progress:
        tk_tr = progress.add_task("[red]Training...", total=args.epochs)

        for epoch in range(args.epochs):

            tr_loss, tr_perf = tr_clean(
                loader=tr_loader,
                model=model,
                obj=obj,
                opt=optim,
                metrics=metrics,
                pred_fn=pred_fn,
                device=device,
            )
            va_loss, va_perf = eval_fn(
                loader=va_loader,
                model=model,
                obj=obj,
                metrics=metrics,
                pred_fn=pred_fn,
                device=device,
            )
            te_loss, te_perf = eval_fn(
                loader=te_loader,
                model=model,
                obj=obj,
                metrics=metrics,
                pred_fn=pred_fn,
                device=device,
            )
            demo_p, eq_opp, eq_odd = fair_metric(
                loader=te_loader, model=model, device=device
            )

            progress.console.log(
                f"Epoch {epoch}: tr_loss {tr_loss}, tr_perf (acc:{tr_perf['acc']}, auc:{tr_perf['auc']}), va_loss {va_loss}, va_perf (acc:{va_perf['acc']}, auc:{va_perf['auc']})"
            )

            for key in metrics.keys():
                history[f"tr_{key}"] = tr_perf[key]
                history[f"va_{key}"] = va_perf[key]
                history[f"va_{key}"] = va_perf[key]
            history["tr_loss"].append(tr_loss)
            history["va_loss"].append(va_loss)
            history["te_loss"].append(te_loss)
            history["dp"].append(demo_p)
            history["eqopp"].append(eq_opp)
            history["eqodd"].append(eq_odd)
            progress.advance(tk_tr)
            es(
                epoch=epoch,
                epoch_score=va_perf,
                model=model,
                model_path=args.model_path + model_name,
            )

    model.load_state_dict(torch.load(args.model_path + model_name))
    te_loss, te_perf = eval_fn(
        loader=te_loader,
        model=model,
        obj=obj,
        metrics=metrics,
        pred_fn=pred_fn,
        device=device,
    )
    demo_p, eq_opp, eq_odd = fair_metric(loader=te_loader, model=model, device=device)
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
