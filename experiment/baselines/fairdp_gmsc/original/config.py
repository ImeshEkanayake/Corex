import argparse


def add_general_group(group):
    group.add_argument("--pname", type=str, default="", help="", required=True)
    group.add_argument(
        "--model_path",
        type=str,
        default="results/models/",
        help="dir path for saving model file",
    )
    group.add_argument(
        "--res_path", type=str, default="results/dict/", help="dir path for output file"
    )
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument(
        "--gmode",
        type=str,
        default="clean",
        help="Mode of running ['clean', 'dp', 'fair', 'proposed', 'alg1', 'onebatch']",
    )
    # group.add_argument("--gsmode", type=str, default="fair", help="")
    # group.add_argument(
    #     "--met",
    #     type=str,
    #     default="acc",
    #     help="Metrics of performance ['acc', 'f1', 'auc', 'pre']",
    # )


def add_data_group(group):
    group.add_argument(
        "--data_path", type=str, default="Data/", help="dir path to dataset"
    )
    group.add_argument("--data", type=str, default="adult", help="name of dataset")
    group.add_argument("--dmode", type=str, default="none", help="mode of data")
    group.add_argument(
        "--rat",
        type=float,
        default=0.5,
        help="ratio group0/group1 where group 0 always has less data points compare to group 1",
    )


def add_model_group(group):
    group.add_argument("--model", type=str, default="NN", help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument(
        "--bs", type=int, default=512, help="batch size for training process"
    )
    group.add_argument(
        "--nhid", type=int, default=32, help="number hidden embedding dim"
    )
    group.add_argument("--nlay", type=int, default=4, help="number of layer")
    group.add_argument("--opt", type=str, default="adam")
    group.add_argument("--epochs", type=int, default=100, help="training step")
    group.add_argument("--ndraw", type=int, default=50)
    group.add_argument("--dout", type=float, default=0.2, help="dropout")
    group.add_argument(
        "--wd", type=float, default=0.0, help="weight decay of the optimizers"
    )
    group.add_argument("--n_mo", type=int, default=10, help="number of models")
    group.add_argument(
        "--sqrt_lr", type=int, default=1, help="learning rate reduce by sqrt of epoch"
    )


def add_dp_group(group):
    group.add_argument(
        "--srate", type=float, default=0.08, help="batch size for training process"
    )
    group.add_argument(
        "--clipsrate", type=float, default=0.08, help="Clip sampling rate for dpissgd"
    )
    group.add_argument("--ns", type=float, default=1.0, help="noise scale for dp")
    group.add_argument("--ns_", type=float, default=1.0, help="noise scale for icml")
    group.add_argument(
        "--cgrad", type=float, default=1.0, help="clipping gradient bound"
    )
    group.add_argument(
        "--clay", type=float, default=1.0, help="clipping last layer bound"
    )
    group.add_argument("--epsilon", type=float, default=1.0, help="targeted epsilon")
    group.add_argument("--lamda", type=float, default=0.5, help="regularizer")
    group.add_argument("--bdset", type=str, default="train")
    group.add_argument("--swprop", type=float, default=0.5)


def add_debug_group(group):
    group.add_argument("--debug", type=int, default=0)
    group.add_argument("--debug_tr", type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    dp_group = parser.add_argument_group(title="DP configuration")
    debug_group = parser.add_argument_group(title="Debug configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    add_dp_group(dp_group)
    add_debug_group(debug_group)
    return parser.parse_args()
