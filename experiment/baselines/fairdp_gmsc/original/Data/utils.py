import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.manifold import TSNE
from typing import Dict, Sequence, Union
from torch.utils.data import DataLoader
from Data.datasets import Data
from Utils.console import console


def minmax_scale(df: pd.DataFrame, cols: list):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for col in cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


def choose_data(df0: pd.DataFrame, df1: pd.DataFrame, rat: float, lab: str, seed: int):

    switch = False

    if len(df0) > len(df1):
        df = df1.copy()
        df1 = df0.copy()
        df0 = df.copy()
        switch = True
        del df

    df0 = df0.reset_index(drop=True)
    df1 = df1.reset_index(drop=True)

    console.log(
        f"Advantage group has {df1.shape[0]} data, Disadvantage group has {df0.shape[0]}, with ratio: {df1.shape[0] / (df0.shape[0] + 1e-12)}"
    )

    if rat < df1.shape[0] / (df0.shape[0] + 1e-12):
        indx = list(df1.index)
        target = df1[lab].values
        num_left = int(rat * df0.shape[0])
        id_tk, id_rm, _, _ = train_test_split(
            indx,
            target,
            test_size=(1 - num_left / (df1.shape[0] + 1e-12)),
            stratify=target,
            random_state=seed,
        )
        df1 = df1.iloc[id_tk, :].copy()
    else:
        indx = list(df0.index)
        target = df0[lab].values
        num_left = int(df1.shape[0] / (rat + 1e-12))
        id_tk, id_rm, _, _ = train_test_split(
            indx,
            target,
            test_size=(1 - num_left / (df0.shape[0] + 1e-12)),
            stratify=target,
            random_state=seed,
        )
        df0 = df0.iloc[id_tk, :].copy()

    return df0.reset_index(drop=True), df1.reset_index(drop=True), switch


def get_data_info(data_dict: Dict):

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
        "test MI target & protect": f"{mutual_info_score(labels_true=te_df[target], labels_pred=te_df[protect])}",
    }

    return data_info


def get_pixel(df: pd.DataFrame):
    X = df.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
    X = np.stack(X)
    X = X / 255.0
    X = X.astype("float32").reshape(X.shape[0], 1, 48, 48)
    return X


def generate_distshift(df: pd.DataFrame, att: Union[Sequence[str], str]):

    if isinstance(att, str):
        df0 = df.loc[df[att] == 0].copy().reset_index(drop=True)
        df1 = df.loc[df[att] == 1].copy().reset_index(drop=True)

        rat0 = df0.shape[0] / df.shape[0]
        rat1 = df1.shape[0] / df.shape[0]

        min_rat = min([rat0, rat1])
        gamma = np.linspace(start=0, stop=min_rat / 2, num=5)
        dfs = []

        for gam in gamma[1:]:

            rat0_ = rat0 - gam
            rat1_ = rat1 - gam

            df0_ = df0.sample(n=int(df0.shape[0] * rat0_))
            df1_ = df1.sample(n=int(df1.shape[0] * rat1_))

            df_ = (
                pd.concat([df0_, df1_], axis=0)
                .copy()
                .sample(frac=1.0)
                .reset_index(drop=True)
            )
            dfs.append(df_)

    else:
        att1, att2 = att
        df00 = df.loc[(df[att1] == 0) & (df[att2] == 0)].copy().reset_index(drop=True)
        df01 = df.loc[(df[att1] == 0) & (df[att2] == 1)].copy().reset_index(drop=True)
        df10 = df.loc[(df[att1] == 1) & (df[att2] == 0)].copy().reset_index(drop=True)
        df11 = df.loc[(df[att1] == 1) & (df[att2] == 1)].copy().reset_index(drop=True)

        rat00 = df00.shape[0] / df.shape[0]
        rat01 = df01.shape[0] / df.shape[0]
        rat10 = df10.shape[0] / df.shape[0]
        rat11 = df11.shape[0] / df.shape[0]

        min_rat = min([rat00, rat01, rat10, rat11])
        gamma = np.linspace(start=0, stop=min_rat, num=5)
        dfs = []

        for gam in gamma[1:]:

            rat00_ = rat00 - gam
            rat01_ = rat01 - gam
            rat10_ = rat10 - gam
            rat11_ = rat11 - gam

            df00_ = df00.sample(n=int(df00.shape[0] * rat00_))
            df01_ = df01.sample(n=int(df01.shape[0] * rat01_))
            df10_ = df10.sample(n=int(df10.shape[0] * rat10_))
            df11_ = df11.sample(n=int(df11.shape[0] * rat11_))

            df_ = (
                pd.concat([df00_, df01_, df10_, df11_], axis=0)
                .copy()
                .sample(frac=1.0)
                .reset_index(drop=True)
            )
            dfs.append(df_)

    return gamma[1:], dfs


def init_subloader(
    df: pd.DataFrame,
    target: str,
    protect: str,
    feat: Sequence[str],
    bste: int,
    type: str,
):

    mal_df = df.loc[df[protect] == 1].copy().reset_index(drop=True)
    fem_df = df.loc[df[protect] == 0].copy().reset_index(drop=True)

    if type == "tabular":
        te_dataset = Data(
            X=df[feat].values,
            y=df[target].values,
            ismale=df[protect].values,
        )
    else:
        X = get_pixel(df=df)
        te_dataset = Data(X=X, y=df[target].values, ismale=df[protect].values)

    te_loader = DataLoader(te_dataset, batch_size=bste, shuffle=False, drop_last=False)
    return te_loader


# def prepare_func(dfs: tuple, features: list, target: str):

#     maltr_df, femtr_df, malva_df, femva_df, malte_df, femte_df = dfs

#     df_tr = (
#         pd.concat([maltr_df, femtr_df], axis=0).sample(frac=1.0).reset_index(drop=True)
#     )
#     df_va = (
#         pd.concat([malva_df, femva_df], axis=0).sample(frac=1.0).reset_index(drop=True)
#     )
#     df_te = (
#         pd.concat([malte_df, femte_df], axis=0).sample(frac=1.0).reset_index(drop=True)
#     )

#     # female
#     X_fem = femtr_df[features].values
#     y_fem = femtr_df[target].values.reshape(-1, 1)
#     X_fem = X_fem / np.linalg.norm(X_fem, ord=2, axis=1).reshape(-1, 1)

#     # male
#     X_mal = maltr_df[features].values
#     y_mal = maltr_df[target].values.reshape(-1, 1)
#     X_mal = X_mal / np.linalg.norm(X_mal, ord=2, axis=1).reshape(-1, 1)

#     # train
#     X_train = df_tr[features].values
#     y_train = df_tr[target].values.reshape(-1, 1)
#     X_train = X_train / np.linalg.norm(X_train, ord=2, axis=1).reshape(-1, 1)

#     # valid
#     X_mal_val = malva_df[features].values
#     y_mal_val = malva_df[target].values.reshape(-1, 1)
#     X_mal_val = X_mal_val / np.linalg.norm(X_mal_val, ord=2, axis=1).reshape(-1, 1)

#     X_fem_val = femva_df[features].values
#     y_fem_val = femva_df[target].values.reshape(-1, 1)
#     X_fem_val = X_fem_val / np.linalg.norm(X_fem_val, ord=2, axis=1).reshape(-1, 1)

#     X_valid = df_va[features].values
#     y_valid = df_va[target].values.reshape(-1, 1)
#     X_valid = X_valid / np.linalg.norm(X_valid, ord=2, axis=1).reshape(-1, 1)

#     # test
#     X_test = df_te[features].values
#     y_test = df_te[target].values.reshape(-1, 1)
#     X_test = X_test / np.linalg.norm(X_test, ord=2, axis=1).reshape(-1, 1)

#     X_mal_te = malte_df[features].values
#     y_mal_te = malte_df[target].values.reshape(-1, 1)
#     X_mal_te = X_mal_te / np.linalg.norm(X_mal_te, ord=2, axis=1).reshape(-1, 1)

#     X_fem_te = femte_df[features].values
#     y_fem_te = femte_df[target].values.reshape(-1, 1)
#     X_fem_te = X_fem_te / np.linalg.norm(X_fem_te, ord=2, axis=1).reshape(-1, 1)

#     tr_info = (X_train, X_mal, X_fem, y_train, y_mal, y_fem)
#     va_info = (X_valid, X_mal_val, X_fem_val, y_valid, y_mal_val, y_fem_val)
#     te_info = (X_test, X_mal_te, X_fem_te, y_test, y_mal_te, y_fem_te)
#     return tr_info, va_info, te_info
