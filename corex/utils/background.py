"""Background and baseline normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_baseline(background_or_baseline: Any, width: int | None = None) -> list[float] | None:
    if background_or_baseline is None:
        return None
    array = np.asarray(background_or_baseline, dtype=float)
    if array.ndim == 0:
        values = [float(array)]
    elif array.ndim == 1:
        values = array.astype(float).tolist()
    elif array.ndim == 2:
        if array.shape[0] == 0:
            values = [0.0] * int(width or array.shape[1])
        else:
            values = array.mean(axis=0).astype(float).tolist()
    else:
        raise ValueError(f"Background/baseline must be 1D or 2D, got shape {array.shape}.")

    if width is not None and len(values) != int(width):
        raise ValueError(f"Background/baseline width {len(values)} does not match expected feature width {width}.")
    return values
