import torch
import pandas as pd
from torch.utils.data import DataLoader
from Data.datasets import Data
from Data.utils import get_pixel
from Data.sampler import WeightedUniformWithReplacementSampler
from Utils.console import console
from opacus.data_loader import DPDataLoader


def prepare_clean(
    dfs: tuple,
    feat: list,
    target: str,
    protect: str,
    bs: int,
    type: str = "tabular",
):

    tr_df, va_df, te_df = dfs
    if type == "tabular":
        tr_data = Data(
            X=tr_df[feat].values, y=tr_df[target].values, ismale=tr_df[protect].values
        )

        ## valid
        va_data = Data(
            X=va_df[feat].values, y=va_df[target].values, ismale=va_df[protect].values
        )

        te_data = Data(
            X=te_df[feat].values, y=te_df[target].values, ismale=te_df[protect].values
        )
    else:
        X_tr = get_pixel(df=tr_df)
        X_va = get_pixel(df=va_df)
        X_te = get_pixel(df=te_df)

        tr_data = Data(X=X_tr, y=tr_df[target].values, ismale=tr_df[protect].values)
        va_data = Data(X=X_va, y=va_df[target].values, ismale=va_df[protect].values)
        te_data = Data(X=X_te, y=te_df[target].values, ismale=te_df[protect].values)

    tr_loader = DataLoader(tr_data, batch_size=bs, drop_last=True, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=bs, drop_last=False, shuffle=False)
    te_loader = DataLoader(te_data, batch_size=bs, drop_last=False, shuffle=False)

    tr_info = tr_loader
    va_info = va_loader
    te_info = te_loader
    return tr_info, va_info, te_info


def prepare_smooth(
    dfs: tuple,
    feat: list,
    target: str,
    protect: str,
    batch_size: int,
    type: str = "tabular",
    mode: str = "smooth",
    sampling_rate: float = None,
):

    tr_df, df_va, df_te = dfs
    mtr_df = tr_df.loc[tr_df[protect] == 1].copy().reset_index(drop=True)
    ftr_df = tr_df.loc[tr_df[protect] == 0].copy().reset_index(drop=True)

    if type == "tabular":
        mtr_data = Data(
            X=mtr_df[feat].values,
            y=mtr_df[target].values,
            ismale=mtr_df[protect].values,
        )
        ftr_data = Data(
            X=ftr_df[feat].values,
            y=ftr_df[target].values,
            ismale=ftr_df[protect].values,
        )
        va_data = Data(
            X=df_va[feat].values, y=df_va[target].values, ismale=df_va[protect].values
        )
        te_data = Data(
            X=df_te[feat].values, y=df_te[target].values, ismale=df_te[protect].values
        )
    else:
        X_maltr = get_pixel(df=mtr_df)
        X_femtr = get_pixel(df=ftr_df)
        X_va = get_pixel(df=df_va)
        X_te = get_pixel(df=df_te)

        mtr_data = Data(
            X=X_maltr, y=mtr_df[target].values, ismale=mtr_df[protect].values
        )
        ftr_data = Data(
            X=X_femtr, y=ftr_df[target].values, ismale=ftr_df[protect].values
        )
        va_data = Data(X=X_va, y=df_va[target].values, ismale=df_va[protect].values)
        te_data = Data(X=X_te, y=df_te[target].values, ismale=df_te[protect].values)

    if mode == "smooth":
        mtr_loader = DataLoader(mtr_data, batch_size=batch_size, drop_last=True)
        ftr_loader = DataLoader(ftr_data, batch_size=batch_size, drop_last=True)
    elif mode == "dpsgds":
        mtr_loader = DataLoader(
            mtr_data, batch_size=int(sampling_rate * len(mtr_data)), drop_last=False
        )
        ftr_loader = DataLoader(
            ftr_data, batch_size=int(sampling_rate * len(ftr_data)), drop_last=False
        )
    va_loader = DataLoader(
        va_data, batch_size=batch_size, shuffle=False, drop_last=False
    )
    te_loader = DataLoader(
        te_data, batch_size=batch_size, shuffle=False, drop_last=False
    )
    tr_info = (mtr_loader, ftr_loader)
    va_info = va_loader
    te_info = te_loader
    return tr_info, va_info, te_info


def prepare_dpsgd(
    dfs: tuple,
    feat: list,
    target: str,
    protect: str,
    sprate: float,
    bste: int,
    type: str = "tabular",
):

    df_tr, df_va, df_te = dfs

    if type == "tabular":
        tr_data = Data(
            X=df_tr[feat].values, y=df_tr[target].values, ismale=df_tr[protect].values
        )
        va_data = Data(
            X=df_va[feat].values, y=df_va[target].values, ismale=df_va[protect].values
        )
        te_data = Data(
            X=df_te[feat].values, y=df_te[target].values, ismale=df_te[protect].values
        )
    else:
        X_tr = get_pixel(df=df_tr)
        X_va = get_pixel(df=df_va)
        X_te = get_pixel(df=df_te)
        tr_data = Data(X=X_tr, y=df_tr[target].values, ismale=df_tr[protect].values)
        va_data = Data(X=X_va, y=df_va[target].values, ismale=df_va[protect].values)
        te_data = Data(X=X_te, y=df_te[target].values, ismale=df_te[protect].values)

    tr_loader = DataLoader(
        tr_data, batch_size=int(sprate * len(df_tr)), drop_last=False
    )
    # tr_loader = DPDataLoader.from_data_loader(
    #     tr_loader, generator=None, distributed=False
    # )

    va_loader = DataLoader(va_data, batch_size=bste, shuffle=False, drop_last=False)
    te_loader = DataLoader(te_data, batch_size=bste, shuffle=False, drop_last=False)

    tr_info = tr_loader
    va_info = va_loader
    te_info = te_loader
    return tr_info, va_info, te_info


def prepare_fairdp(
    dfs: tuple,
    feat: list,
    target: str,
    protect: str,
    sprate: float,
    bste: int,
    type: str = "tabular",
):

    tr_df, df_va, df_te = dfs
    mtr_df = tr_df.loc[tr_df[protect] == 1].copy().reset_index(drop=True)
    ftr_df = tr_df.loc[tr_df[protect] == 0].copy().reset_index(drop=True)

    if type == "tabular":
        mtr_data = Data(
            X=mtr_df[feat].values,
            y=mtr_df[target].values,
            ismale=mtr_df[protect].values,
        )
        ftr_data = Data(
            X=ftr_df[feat].values,
            y=ftr_df[target].values,
            ismale=ftr_df[protect].values,
        )
        va_data = Data(
            X=df_va[feat].values, y=df_va[target].values, ismale=df_va[protect].values
        )
        te_data = Data(
            X=df_te[feat].values, y=df_te[target].values, ismale=df_te[protect].values
        )
    else:
        X_maltr = get_pixel(df=mtr_df)
        X_femtr = get_pixel(df=ftr_df)
        X_va = get_pixel(df=df_va)
        X_te = get_pixel(df=df_te)

        mtr_data = Data(
            X=X_maltr, y=mtr_df[target].values, ismale=mtr_df[protect].values
        )
        ftr_data = Data(
            X=X_femtr, y=ftr_df[target].values, ismale=ftr_df[protect].values
        )
        va_data = Data(X=X_va, y=df_va[target].values, ismale=df_va[protect].values)
        te_data = Data(X=X_te, y=df_te[target].values, ismale=df_te[protect].values)

    mtr_loader = DataLoader(
        mtr_data, batch_size=int(sprate * len(mtr_data)), drop_last=False
    )
    ftr_loader = DataLoader(
        ftr_data, batch_size=int(sprate * len(ftr_data)), drop_last=False
    )
    mtr_loader = DPDataLoader.from_data_loader(
        mtr_loader, generator=None, distributed=False
    )
    ftr_loader = DPDataLoader.from_data_loader(
        ftr_loader, generator=None, distributed=False
    )
    va_loader = DataLoader(va_data, batch_size=bste, shuffle=False, drop_last=False)
    te_loader = DataLoader(te_data, batch_size=bste, shuffle=False, drop_last=False)
    tr_info = (mtr_loader, ftr_loader)
    va_info = va_loader
    te_info = (te_loader, df_te)
    return tr_info, va_info, te_info


def prepare_dpissgd(
    dfs: tuple,
    feat: list,
    target: str,
    protect: str,
    sprate: float,
    clip_sprate: float,
    bste: int,
    type: str = "tabular",
):

    df_tr, df_va, df_te = dfs

    if type == "tabular":
        tr_data = Data(
            X=df_tr[feat].values, y=df_tr[target].values, ismale=df_tr[protect].values
        )
        va_data = Data(
            X=df_va[feat].values, y=df_va[target].values, ismale=df_va[protect].values
        )
        te_data = Data(
            X=df_te[feat].values, y=df_te[target].values, ismale=df_te[protect].values
        )
    else:
        X_tr = get_pixel(df=df_tr)
        X_va = get_pixel(df=df_va)
        X_te = get_pixel(df=df_te)
        tr_data = Data(X=X_tr, y=df_tr[target].values, ismale=df_tr[protect].values)
        va_data = Data(X=X_va, y=df_va[target].values, ismale=df_va[protect].values)
        te_data = Data(X=X_te, y=df_te[target].values, ismale=df_te[protect].values)

    protect_tensor = torch.Tensor(df_tr[protect].astype(int).tolist())
    group, group_couts = protect_tensor.unique(return_counts=True)
    group_weights = 1 / (group_couts + 1e-12)
    weights = group_weights[protect_tensor.long()]
    weights = weights / weights.sum() * len(tr_data)
    sampler = WeightedUniformWithReplacementSampler(
        weights,
        num_samples=len(tr_data),
        sample_rate=sprate,
        clip_sample_rate=clip_sprate,
    )

    tr_loader = DataLoader(
        tr_data,
        shuffle=False,
        batch_sampler=sampler,
        batch_size=1,
    )
    va_loader = DataLoader(va_data, batch_size=bste, shuffle=False, drop_last=False)
    te_loader = DataLoader(te_data, batch_size=bste, shuffle=False, drop_last=False)

    tr_info = tr_loader
    va_info = va_loader
    te_info = te_loader
    return tr_info, va_info, te_info
