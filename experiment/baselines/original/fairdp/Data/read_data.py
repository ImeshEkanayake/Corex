import os
import pickle
import pandas as pd
from typing import Dict
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from Data.utils import get_data_info, choose_data
from Data.prepare_loader import (
    prepare_clean,
    prepare_smooth,
    prepare_dpsgd,
    prepare_fairdp,
    prepare_dpissgd,
)
from Data.get_data import get_adult, get_ccc, get_utk
from Utils.console import log_table, console
from Utils.utils import save_dict


def read_data(args):

    if args.data == "adult":
        df = get_adult(dmode=args.dmode, rat=args.rat, seed=args.seed)
        target = "income"
        protect = "sex"
    elif args.data == "ccc":
        df = get_ccc(dmode=args.dmode, rat=args.rat, seed=args.seed)
        target = "y"
        protect = "SEX"
    elif args.data == "utk":
        df = get_utk(dmode=args.dmode, rat=args.rat, seed=args.seed)
        target = "age"
        protect = "ethnicity"

    if args.data != "utk":
        feature_cols = list(df.columns)
        feature_cols.remove(target)
        feature_cols.remove(protect)
    else:
        feature_cols = "pixels"

    if args.dmode == "rat":
        adv_df = df.loc[df[protect] == 1].reset_index(drop=True)
        disadv_df = df.loc[df[protect] == 0].reset_index(drop=True)
        adv_df, disadv_df, switch = choose_data(
            df0=disadv_df, df1=adv_df, rat=args.rat, lab=target, seed=args.seed
        )

        if switch:
            df_ = adv_df.copy()
            adv_df = disadv_df.copy()
            disadv_df = df_.copy()
            del df_
        df = (
            pd.concat([adv_df, disadv_df], axis=0)
            .sample(frac=1.0)
            .reset_index(drop=True)
        )
        idx = list(df.index)
        id_tr, id_te, _, _ = train_test_split(
            idx,
            df[[target, protect]],
            test_size=0.2,
            stratify=df[[target, protect]],
            random_state=args.seed,
        )
        id_tr, id_va, _, _ = train_test_split(
            id_tr,
            df.iloc[id_tr][[target, protect]],
            test_size=0.1,
            stratify=df.iloc[id_tr][[target, protect]],
            random_state=args.seed,
        )
        split_dict = {"id_tr": id_tr, "id_va": id_va, "id_te": id_te}
    else:
        if os.path.exists(
            os.path.join(
                f"results/dict/{args.data}", f"{args.data}-run-{args.seed}.pkl"
            )
        ):
            console.log("Found previous set up, and using previous split")
            with open(
                os.path.join(
                    f"results/dict/{args.data}", f"{args.data}-run-{args.seed}.pkl"
                ),
                "rb",
            ) as file:
                split_dict = pickle.load(file)
        else:
            idx = list(df.index)
            id_tr, id_te, _, _ = train_test_split(
                idx,
                df[[target, protect]],
                test_size=0.2,
                stratify=df[[target, protect]],
                random_state=args.seed,
            )
            id_tr, id_va, _, _ = train_test_split(
                id_tr,
                df.iloc[id_tr][[target, protect]],
                test_size=0.1,
                stratify=df.iloc[id_tr][[target, protect]],
                random_state=args.seed,
            )
            split_dict = {"id_tr": id_tr, "id_va": id_va, "id_te": id_te}
            save_dict(
                path=os.path.join(
                    f"results/dict/{args.data}", f"{args.data}-run-{args.seed}.pkl"
                ),
                dct=split_dict,
            )

    tr_df = df.iloc[split_dict["id_tr"]].copy().reset_index(drop=True)
    va_df = df.iloc[split_dict["id_va"]].copy().reset_index(drop=True)
    te_df = df.iloc[split_dict["id_te"]].copy().reset_index(drop=True)

    data_dict = {
        "tr_df": tr_df,
        "va_df": va_df,
        "te_df": te_df,
        "features": feature_cols,
        "protect": protect,
        "target": target,
    }
    args.att_lb = target
    args.att_pr = protect
    args.att_fe = feature_cols

    if args.data != "utk":
        data_info = get_data_info(data_dict=data_dict)
        log_table(dct=data_info, name=f"Data information of: {args.data}")
    else:
        target = data_dict["target"]
        protect = data_dict["protect"]
        tr_df = data_dict["tr_df"].copy()
        va_df = data_dict["va_df"].copy()
        te_df = data_dict["te_df"].copy()

        data_info = {
            "target": f"{target}",
            "protect": f"{protect}",
            "# train": f"{data_dict['tr_df'].shape[0]}",
            "# train pos": f"{tr_df.loc[tr_df[target] == 1].shape[0]}",
            "# train neg": f"{tr_df.loc[tr_df[target] == 0].shape[0]}",
            "# train adv": f"{tr_df.loc[tr_df[protect] == 1].shape[0]}",
            "# train disadv": f"{tr_df.loc[tr_df[protect] == 0].shape[0]}",
            "train MI target & protect": f"{mutual_info_score(labels_true=tr_df[target], labels_pred=tr_df[protect])}",
            "# valid": f"{data_dict['va_df'].shape[0]}",
            "# valid pos": f"{va_df.loc[va_df[target] == 1].shape[0]}",
            "# valid neg": f"{va_df.loc[va_df[target] == 0].shape[0]}",
            "# valid adv": f"{va_df.loc[va_df[protect] == 1].shape[0]}",
            "# valid disadv": f"{va_df.loc[va_df[protect] == 0].shape[0]}",
            "valid MI target & protect": f"{mutual_info_score(labels_true=va_df[target], labels_pred=va_df[protect])}",
            "# test": f"{data_dict['te_df'].shape[0]}",
            "# test pos": f"{te_df.loc[te_df[target] == 1].shape[0]}",
            "# test neg": f"{te_df.loc[te_df[target] == 0].shape[0]}",
            "# test adv": f"{te_df.loc[te_df[protect] == 1].shape[0]}",
            "# test disadv": f"{te_df.loc[te_df[protect] == 0].shape[0]}",
            "valid MI target & protect": f"{mutual_info_score(labels_true=te_df[target], labels_pred=te_df[protect])}",
        }
        log_table(dct=data_info, name=f"Data information of: {args.data}")
    return data_dict


def init_loader(args, data_dict: Dict):

    target = data_dict["target"]
    protect = data_dict["protect"]
    feat_col = data_dict["features"]
    tr_df = data_dict["tr_df"].copy()
    va_df = data_dict["va_df"].copy()
    te_df = data_dict["te_df"].copy()

    dfs = (tr_df, va_df, te_df)

    if args.gmode in ["clean"]:
        dtype = "image" if args.data == "utk" else "tabular"
        tr_info, va_info, te_info = prepare_clean(
            dfs=dfs,
            feat=feat_col,
            target=target,
            protect=protect,
            bs=args.bs,
            type=dtype,
        )
    elif args.gmode in ["smooth", "dpsgds"]:
        dtype = "image" if args.data == "utk" else "tabular"
        tr_info, va_info, te_info = prepare_smooth(
            dfs=dfs,
            feat=feat_col,
            target=target,
            protect=protect,
            batch_size=args.bs,
            type=dtype,
            mode=args.gmode,
            sampling_rate=args.srate,
        )
    elif args.gmode in ["dpsgd", "dpsgdf"]:
        dtype = "image" if args.data == "utk" else "tabular"
        tr_info, va_info, te_info = prepare_dpsgd(
            dfs=dfs,
            feat=feat_col,
            target=target,
            protect=protect,
            sprate=args.srate,
            bste=args.bs,
            type=dtype,
        )
    elif args.gmode in [
        "fairdp",
        "mixed",
        "mixed2",
    ]:
        dtype = "image" if args.data == "utk" else "tabular"
        tr_info, va_info, te_info = prepare_fairdp(
            dfs=dfs,
            feat=feat_col,
            target=target,
            protect=protect,
            sprate=args.srate,
            bste=args.bs,
            type=dtype,
        )
    elif args.gmode in ["dpissgd"]:
        dtype = "image" if args.data == "utk" else "tabular"
        tr_info, va_info, te_info = prepare_dpissgd(
            dfs=dfs,
            feat=feat_col,
            target=target,
            protect=protect,
            sprate=args.srate,
            clip_sprate=args.clipsrate,
            bste=args.bs,
            type=dtype,
        )
    else:
        console.log(f"Unrecogizable running mode: {args.gmode}")

    return tr_info, va_info, te_info


# if args.gmode == "func":
#     df = minmax_scale(df=df, cols=feature_cols)
#     df["bias"] = 1.0
#     feature_cols.append("bias")


# if args.gmode == "func":
#     tr_info, va_info, te_info = prepare_func(
#         dfs=dfs, features=feat_col, target=target
#     )
