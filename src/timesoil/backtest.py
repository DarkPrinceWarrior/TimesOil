"""Скользящий бэктест: 3 среза по 6 месяцев + сводка метрик.

Срезы (конец контекста -> тестовое окно):
  2014-05 -> 2014-06..2014-11
  2014-11 -> 2014-12..2015-05
  2015-05 -> 2015-06..2015-11  (основной)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import metrics as M

HORIZON = 6
CUTOFFS = (
    pd.Timestamp("2014-05-01"),
    pd.Timestamp("2014-11-01"),
    pd.Timestamp("2015-05-01"),
)
# расширенный набор для весов ансамбля и калибровки (14 срезов, шаг 2 мес)
EXT_CUTOFFS = tuple(pd.date_range("2013-03-01", "2015-05-01", freq="2MS"))

# сигнатура: (история одной скважины без NaN, горизонт) -> прогноз [h]
PointForecaster = Callable[[np.ndarray, int], np.ndarray]


def run_pointwise(
    mat: pd.DataFrame,
    forecaster: PointForecaster,
    cutoffs: tuple[pd.Timestamp, ...] = CUTOFFS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Прогон по-скважинного бейслайна на всех срезах.

    mat: date x well (NaN до старта скважины). Возвращает длинную таблицу
    [cutoff, well, step, date, y_true, y_pred].
    """
    rows = []
    for cutoff in cutoffs:
        ctx = mat.loc[:cutoff]
        test = mat.loc[cutoff:].iloc[1 : horizon + 1]
        assert len(test) == horizon, f"нет {horizon} тестовых месяцев после {cutoff.date()}"
        for w in mat.columns:
            hist = ctx[w].dropna().to_numpy()
            if len(hist) < 12:
                continue
            pred = np.maximum(forecaster(hist, horizon), 0.0)
            for step, (dt, yt) in enumerate(test[w].items(), 1):
                rows.append(
                    dict(cutoff=cutoff, well=w, step=step, date=dt,
                         y_true=float(yt), y_pred=float(pred[step - 1]))
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, insample: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сводка метрик: по срезу и в целом (все скважины и месяцы вместе)."""
    out = []
    for cutoff, g in results.groupby("cutoff"):
        row = dict(
            cutoff=str(pd.Timestamp(cutoff).date()),
            wape=M.wape(g.y_true, g.y_pred),
            smape=M.smape(g.y_true, g.y_pred),
            rmse=M.rmse(g.y_true, g.y_pred),
        )
        # ошибка суммарной добычи за 6 мес по полю
        row["cum_err_pct"] = M.cum_error_pct(g.y_true.to_numpy(), g.y_pred.to_numpy())
        # медианный по скважинам WAPE
        row["wape_med_well"] = float(
            g.groupby("well").apply(lambda x: M.wape(x.y_true, x.y_pred), include_groups=False).median()
        )
        out.append(row)
    total = dict(
        cutoff="ALL",
        wape=M.wape(results.y_true, results.y_pred),
        smape=M.smape(results.y_true, results.y_pred),
        rmse=M.rmse(results.y_true, results.y_pred),
        cum_err_pct=M.cum_error_pct(results.y_true.to_numpy(), results.y_pred.to_numpy()),
        wape_med_well=float(
            results.groupby("well").apply(lambda x: M.wape(x.y_true, x.y_pred), include_groups=False).median()
        ),
    )
    out.append(total)
    return pd.DataFrame(out)
