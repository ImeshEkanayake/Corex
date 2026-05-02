import os
import torch
import datetime
import warnings
from config import parse_args
from Data.read_data import read_data, init_loader
from Runs.utils import init_run
from Utils.console import console
from Utils.utils import seed_everything, print_args, save_dict, init_history

warnings.filterwarnings("ignore")


def run(args, date, device):

    history = init_history()
    args.name = (
        f"{args.pname}-run-{args.seed}-{date.day}{date.month}-{date.hour}{date.minute}"
    )
    # read data
    with console.status("Initializing Data") as status:
        data_dict = read_data(args=args)
        args.feature = data_dict["features"]
        args.target = data_dict["target"]
        args.protect = data_dict["protect"]
        args.indim = len(args.feature)
        args.outdim = 1
        tr_info, va_info, te_info = init_loader(args=args, data_dict=data_dict)
        console.log(f"Done Reading data: :white_check_mark:")

    run = init_run(gmode=args.gmode)
    history = run(
        args=args,
        tr_info=tr_info,
        va_info=va_info,
        te_info=te_info,
        name=args.name,
        history=history,
        device=device,
    )
    save_dict(path=os.path.join(args.res_path, f"{args.name}.pkl"), dct=history)


if __name__ == "__main__":
    date = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.pname}")
    with console.status("Initializing...") as status:
        print_args(args)
        seed_everything(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.log(f"DEVICE USING: {device}")
        console.log(f"[bold][green]Done Initializing!")
        if os.path.exists(f"results/dict/{args.data}") == False:
            os.mkdir(f"results/dict/{args.data}")
        if os.path.exists(f"results/dict/{args.data}/{args.gmode}") == False:
            os.mkdir(f"results/dict/{args.data}/{args.gmode}")
        if os.path.exists(f"results/models/{args.data}") == False:
            os.mkdir(f"results/models/{args.data}")
        if os.path.exists(f"results/models/{args.data}/{args.gmode}") == False:
            os.mkdir(f"results/models/{args.data}/{args.gmode}")
        args.model_path = f"results/models/{args.data}/{args.gmode}/"
        args.res_path = f"results/dict/{args.data}/{args.gmode}/"
    run(args=args, date=date, device=device)
