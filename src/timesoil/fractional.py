"""Фракционная модель обводнённости (закон Джентила).

Водонефтяное отношение зрелой скважины растёт степенным законом от
накопленной закачки, отнесённой к скважине:

    WOR(W) = alpha * W^beta,   f_o = 1 / (1 + WOR),

нефть = жидкость * f_o. Подгонка — линейная регрессия ln WOR на ln W по
последним k месяцам; для полностью обводнённых скважин (нет валидных
точек) — доля нефти держится на последнем наблюдённом уровне.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
WCT_MAX = 0.999


def fit_gentil(
    oil: pd.Series, liq: pd.Series, w_cum: pd.Series, k: int = 48
) -> tuple[float, float] | float:
    """Параметры (ln alpha, beta) либо резервная доля нефти (float)."""
    wct = 1.0 - oil / liq.replace(0.0, np.nan)
    ok = (liq > 0) & (wct > 0.01) & (wct < WCT_MAX) & (w_cum > 0)
    wor = (wct[ok] / (1.0 - wct[ok])).tail(k)
    if len(wor) < 6:
        last_fo = float((oil / liq.replace(0.0, np.nan)).dropna().iloc[-1]) if (liq > 0).any() else 0.0
        return max(last_fo, 0.0)
    x = np.log(w_cum[wor.index].to_numpy(float))
    y = np.log(wor.to_numpy(float))
    beta, ln_alpha = np.polyfit(x, y, 1)
    return float(ln_alpha), float(beta)


def predict_fo(params: tuple[float, float] | float, w_future: np.ndarray) -> np.ndarray:
    """Доля нефти в жидкости на будущих значениях накопленной закачки."""
    if isinstance(params, float):
        return np.full(len(w_future), params)
    ln_alpha, beta = params
    wor = np.exp(ln_alpha + beta * np.log(np.maximum(w_future, EPS)))
    return 1.0 / (1.0 + np.maximum(wor, 0.0))
