"""Бейслайны прогноза дебита: наивный, экспоненциальный тренд, Арпс.

Все функции принимают историю среднесуточного дебита (np.ndarray, без NaN,
начиная с первого месяца работы) и возвращают прогноз на h месяцев.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

EPS = 1e-6


def forecast_naive(y: np.ndarray, h: int) -> np.ndarray:
    """Последнее значение держится весь горизонт."""
    return np.full(h, float(y[-1]))


def forecast_exp(y: np.ndarray, h: int, k: int = 24) -> np.ndarray:
    """Линейный тренд в лог-пространстве за последние k месяцев
    (эквивалент экспоненциального закона падения Арпса)."""
    tail = np.maximum(np.asarray(y[-k:], float), EPS)
    t = np.arange(len(tail))
    b1, b0 = np.polyfit(t, np.log(tail), 1)
    tf = np.arange(len(tail), len(tail) + h)
    return np.exp(b0 + b1 * tf)


def _arps(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def forecast_arps(y: np.ndarray, h: int, k: int = 36) -> np.ndarray:
    """Гиперболический Арпс по последним k месяцам; при неудаче фита — exp."""
    tail = np.maximum(np.asarray(y[-k:], float), EPS)
    t = np.arange(len(tail), dtype=float)
    try:
        p0 = (float(tail[0]), 0.02, 0.5)
        bounds = ([EPS, 1e-5, 0.01], [tail.max() * 10, 1.0, 2.0])
        popt, _ = curve_fit(_arps, t, tail, p0=p0, bounds=bounds, maxfev=5000)
        tf = np.arange(len(tail), len(tail) + h, dtype=float)
        return _arps(tf, *popt)
    except (RuntimeError, ValueError):
        return forecast_exp(y, h, k=min(k, 24))
