import torch
from Model.modules.models import NN, CNN


def init_model(args, clip: bool = False):

    dout = None if args.dout == 0.0 else args.dout

    if args.model == "NN":
        model = NN(
            input_dim=args.indim,
            hidden_dim=args.nhid,
            output_dim=args.outdim,
            n_layer=args.nlay,
            dropout=dout,
            clip=clip,
        )
        return model
    elif args.model == "CNN":
        # channel:list, hid_dim:list, img_size:int, channel_in:int, out_dim:int, debug:int=1, kernal_size:int=5, padding:int=0, stride:int=1, dropout:float=0.2, clip:bool=False
        return CNN(
            channel=[32, 64],
            hid_dim=[256, 64],
            img_size=48,
            channel_in=1,
            out_dim=1,
            kernel_size=3,
            debug=args.debug,
        )


def init_optim(opt: str, lr: float, model: torch.nn.Module, wd: float = 0.0):
    if opt == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optim = None
    return optim
