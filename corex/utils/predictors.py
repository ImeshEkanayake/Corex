"""Predictor normalization helpers."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np


PredictFn = Callable[[list[float]], float]
BatchPredictFn = Callable[[np.ndarray], np.ndarray]
_ROW_METHOD_NAMES = {"predict", "predict_proba", "decision_function"}
CallableMode = Literal["row", "batch"]


def _normalize_scalar_output(output: Any, target_class: int | None = None) -> float:
    array = np.asarray(output, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        if array.size == 1:
            return float(array[0])
        if target_class is not None:
            return float(array[int(target_class)])
        if array.size == 2:
            return float(array[1])
        raise ValueError(
            "Multiclass predictor outputs require target_class=...; "
            f"received output of shape {array.shape}."
        )
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError(f"Predictor output must resolve to a single row; got shape {array.shape}.")
        return _normalize_scalar_output(array[0], target_class=target_class)
    raise ValueError(f"Unsupported predictor output shape: {array.shape}")


def _normalize_batch_output(output: Any, n_rows: int, target_class: int | None = None) -> np.ndarray:
    array = np.asarray(output, dtype=float)
    if array.ndim == 0:
        if n_rows != 1:
            raise ValueError(f"Predictor output must resolve to {n_rows} rows; got scalar output.")
        return np.asarray([float(array)], dtype=float)
    if array.ndim == 1:
        if array.size == n_rows:
            return array.astype(float)
        if n_rows == 1:
            return np.asarray([_normalize_scalar_output(array, target_class=target_class)], dtype=float)
        raise ValueError(f"Predictor output must resolve to {n_rows} rows; got shape {array.shape}.")
    if array.ndim == 2:
        if array.shape[0] != n_rows:
            raise ValueError(f"Predictor output must resolve to {n_rows} rows; got shape {array.shape}.")
        if array.shape[1] == 1:
            return array[:, 0].astype(float)
        if target_class is not None:
            return array[:, int(target_class)].astype(float)
        if array.shape[1] == 2:
            return array[:, 1].astype(float)
        raise ValueError(
            "Multiclass predictor outputs require target_class=...; "
            f"received output of shape {array.shape}."
        )
    raise ValueError(f"Unsupported predictor output shape: {array.shape}")


def _is_row_method_callable(predictor: Any) -> bool:
    return getattr(predictor, "__name__", None) in _ROW_METHOD_NAMES


def _wrap_row_method(method: Callable[..., Any], target_class: int | None = None) -> PredictFn:
    def adapted(values: list[float]) -> float:
        row = np.asarray([values], dtype=float)
        return _normalize_scalar_output(method(row), target_class=target_class)

    setattr(adapted, "_corex_scalar_predictor", True)
    return adapted


def resolve_scalar_predictor(
    predictor: Any,
    prediction_method: str | None = None,
    target_class: int | None = None,
    callable_mode: CallableMode | None = None,
) -> PredictFn:
    if getattr(predictor, "_corex_scalar_predictor", False):
        return predictor

    method_order = [prediction_method] if prediction_method is not None else ["predict_proba", "decision_function", "predict"]
    for method_name in method_order:
        if method_name is None:
            continue
        method = getattr(predictor, method_name, None)
        if method is None:
            continue
        return _wrap_row_method(method, target_class=target_class)

    if _is_row_method_callable(predictor):
        return _wrap_row_method(predictor, target_class=target_class)

    if callable(predictor):
        def fallback_callable(values: list[float]) -> float:
            row = np.asarray(values, dtype=float)
            if callable_mode == "batch":
                return _normalize_scalar_output(predictor(row.reshape(1, -1)), target_class=target_class)
            return _normalize_scalar_output(predictor(row.tolist()), target_class=target_class)

        setattr(fallback_callable, "_corex_scalar_predictor", True)
        return fallback_callable

    raise TypeError(
        "Unsupported predictor. Provide a callable or an estimator exposing "
        "predict_proba, decision_function, or predict."
    )


def resolve_batch_predictor(
    predictor: Any,
    prediction_method: str | None = None,
    target_class: int | None = None,
    callable_mode: CallableMode | None = None,
) -> BatchPredictFn:
    method_order = [prediction_method] if prediction_method is not None else ["predict_proba", "decision_function", "predict"]
    for method_name in method_order:
        if method_name is None:
            continue
        method = getattr(predictor, method_name, None)
        if method is None:
            continue

        def adapted(rows: np.ndarray, method: Callable[..., Any] = method) -> np.ndarray:
            matrix = np.asarray(rows, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            return _normalize_batch_output(method(matrix), matrix.shape[0], target_class=target_class)

        setattr(adapted, "_corex_batch_predictor", True)
        return adapted

    if _is_row_method_callable(predictor):
        def adapted_row_callable(rows: np.ndarray) -> np.ndarray:
            matrix = np.asarray(rows, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            return _normalize_batch_output(predictor(matrix), matrix.shape[0], target_class=target_class)

        setattr(adapted_row_callable, "_corex_batch_predictor", True)
        return adapted_row_callable

    def adapted_generic(rows: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if callable_mode == "batch":
            return _normalize_batch_output(predictor(matrix), matrix.shape[0], target_class=target_class)
        return np.asarray(
            [
                _normalize_scalar_output(predictor(row.astype(float).tolist()), target_class=target_class)
                for row in matrix
            ],
            dtype=float,
        )

    setattr(adapted_generic, "_corex_batch_predictor", True)
    return adapted_generic
