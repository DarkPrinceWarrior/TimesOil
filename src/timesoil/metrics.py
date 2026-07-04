"""Метрики качества прогноза.

Основная — WAPE (взвешенная абсолютная ошибка): устойчива к скважинам с
дебитом около нуля (скв. 1 полностью обводнена), в отличие от MAPE.
"""

from __future__ import annotations

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ok = denom > 0
    return float(np.mean(np.abs(y_true - y_pred)[ok] / denom[ok])) if ok.any() else np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray) -> float:
    """Масштаб — MAE наивного прогноза (лаг 1) на истории."""
    y_insample = np.asarray(y_insample, float)
    scale = np.mean(np.abs(np.diff(y_insample)))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def cum_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ошибка суммарной добычи за горизонт, % (знак сохраняется)."""
    s = np.asarray(y_true, float).sum()
    return float((np.asarray(y_pred, float).sum() - s) / s * 100) if s > 0 else np.nan


def pinball(y_true: np.ndarray, q_pred: np.ndarray, quantiles: np.ndarray) -> float:
    """Средний pinball loss; q_pred: [n_q, h]."""
    y = np.asarray(y_true, float)[None, :]
    q = np.asarray(q_pred, float)
    tau = np.asarray(quantiles, float)[:, None]
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage_80(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(q10)) & (y <= np.asarray(q90))))
