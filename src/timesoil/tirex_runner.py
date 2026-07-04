"""Прогон TiRex-2 (zero-shot) в нескольких конфигурациях.

Варианты:
- "u"          — по одной скважине (унивариатно);
- "m"          — все 33 добывающие одним тензором (вариатный микшер);
- "blocks"     — по TimeseriesType на блок разломов (A..E);
- "blocks_cov" — блоки + ковариаты: закачка нагнетательных блока как
                 future-known (план ППД известен), пластовое давление
                 добывающих блока как past.

Скважины стартуют в разные месяцы: ведущие NaN допустимы (маскируются
моделью), поэтому все ряды выравниваются по общей календарной сетке.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .wells import PRODUCERS, block_wells

QUANTILES = np.round(np.arange(0.1, 0.95, 0.1), 1)
Q_MED = 4  # индекс медианы


def _tt(target: np.ndarray, past: np.ndarray | None, future: np.ndarray | None):
    from tirex2 import TimeseriesType

    def t(a: np.ndarray | None) -> torch.Tensor | None:
        return None if a is None else torch.tensor(np.asarray(a, np.float32))

    return TimeseriesType(target=t(target), past_covariates=t(past), future_covariates=t(future))


def _groups(variant: str) -> list[list[int]]:
    if variant == "u":
        return [[w] for w in PRODUCERS]
    if variant in ("m", "m_cov"):
        return [list(PRODUCERS)]
    if variant in ("blocks", "blocks_cov"):
        return [block_wells(b, injectors=False) for b in ("A", "B", "B2", "C", "D", "E")]
    raise ValueError(variant)


def forecast_tirex(
    model,
    target_mat: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    variant: str,
    inj_mat: pd.DataFrame | None = None,
    pres_mat: pd.DataFrame | None = None,
    inj_future: pd.DataFrame | None = None,
    **forecast_kwargs,
) -> pd.DataFrame:
    """Прогноз всех добывающих на horizon месяцев после cutoff.

    inj_future: закачка на горизонте (для бэктеста — факт, для прогноза
    вперёд — продление последнего режима). Возвращает длинную таблицу
    [well, step, date, q10..q90, y_pred(медиана)].
    """
    ctx = target_mat.loc[:cutoff]
    dates_future = pd.date_range(cutoff, periods=horizon + 1, freq="MS")[1:]
    ts_list, well_groups = [], []
    for wells in _groups(variant):
        tgt = ctx[wells].to_numpy().T  # [n, T]
        past = future = None
        if variant in ("blocks_cov", "m_cov"):
            if variant == "m_cov":
                from .wells import INJECTORS

                blk_inj = sorted(INJECTORS)
            else:
                blk_inj = block_wells(_block_of(wells[0]), injectors=True)
            if blk_inj and inj_mat is not None and inj_future is not None:
                hist = inj_mat.loc[:cutoff, blk_inj].to_numpy().T
                fut = inj_future.loc[dates_future, blk_inj].to_numpy().T
                future = np.concatenate([hist, fut], axis=1)  # [k, T+h]
            if pres_mat is not None:
                past = pres_mat.loc[:cutoff, wells].to_numpy().T
        ts_list.append(_tt(tgt, past, future))
        well_groups.append(wells)

    fcs = model.forecast(ts_list, prediction_length=horizon, output_type="numpy", **forecast_kwargs)
    rows = []
    for wells, fc in zip(well_groups, fcs):
        for i, w in enumerate(wells):
            q = np.maximum(fc[i], 0.0)  # [9, h], дебит неотрицателен
            for step in range(horizon):
                row = dict(well=w, step=step + 1, date=dates_future[step], y_pred=float(q[Q_MED, step]))
                for qi, tau in enumerate(QUANTILES):
                    row[f"q{int(tau * 100)}"] = float(q[qi, step])
                rows.append(row)
    return pd.DataFrame(rows)


def _block_of(well: int) -> str:
    from .wells import WELL_BLOCK

    return WELL_BLOCK[well]
