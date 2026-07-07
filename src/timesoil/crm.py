"""Ёмкостно-резистивная модель (CRM, pywaterflood) на месячных данных.

Единицы: добыча жидкости в т/сут, закачка в м3/сут — рассогласование
масштабов поглощается коэффициентами связности (жидкость преимущественно
вода, плотность ~1). Ось времени — календарные сутки от начала окна,
постоянные времени tau — в сутках.

Два режима:
- fit по блокам (экраны разломов зашиты нулями связей) — рабочий прогноз;
- fit по всему полю без ограничений — проверка блоков данными: сильные
  связи должны концентрироваться внутри блоков.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pywaterflood.crm import CRM, CrmCompensated

from .wells import INJECTORS, PRODUCERS, WELL_BLOCK, block_wells

FULL_START = pd.Timestamp("2008-07-01")
BLOCKS = ("A", "B", "B2", "C", "D", "E")


def _time_axis(index: pd.DatetimeIndex) -> np.ndarray:
    return (index - index[0]).days.to_numpy(float) + index[0].days_in_month


def fit_block(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    producers: list,
    injectors: list,
    cutoff: pd.Timestamp,
    start: pd.Timestamp = FULL_START,
) -> CRM:
    """Подгонка CRM одного блока на истории [start, cutoff]
    (start — месяц, с которого работают все добывающие блока)."""
    hist = liq.loc[start:cutoff, producers]
    inj_hist = inj.loc[start:cutoff, injectors]
    model = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
    model.fit(
        hist.to_numpy(),
        inj_hist.to_numpy(),
        _time_axis(hist.index),
        num_cores=4,
    )
    return model


def predict_block(
    model: CRM,
    inj_full: pd.DataFrame,
    producers: list,
) -> pd.DataFrame:
    """Прогноз подогнанного CRM на произвольном графике закачки (быстро)."""
    pred = model.predict(
        injection=inj_full.to_numpy(),
        time=_time_axis(inj_full.index),
    )
    return pd.DataFrame(np.maximum(pred, 0.0), index=inj_full.index, columns=producers)


def fit_block_compensated(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    bhp: pd.DataFrame,
    producers: list,
    injectors: list,
    cutoff: pd.Timestamp,
    start: pd.Timestamp = FULL_START,
) -> CrmCompensated:
    """CRM с компенсацией забойного давления добывающих (член -J tau dPзаб/dt);
    нужен, когда добывающие работают при переменном забойном давлении."""
    hist = liq.loc[start:cutoff, producers]
    pres = bhp.loc[start:cutoff, producers]
    inj_hist = inj.loc[start:cutoff, injectors]
    model = CrmCompensated(primary=True, tau_selection="per-pair", constraints="up-to one")
    model.fit(
        hist.to_numpy(), pres.to_numpy(), inj_hist.to_numpy(),
        _time_axis(hist.index), num_cores=4,
    )
    return model


def predict_block_compensated(
    model: CrmCompensated,
    inj_full: pd.DataFrame,
    bhp_full: pd.DataFrame,
    producers: list,
) -> pd.DataFrame:
    """Прогноз CRM-P; забойное давление контрольного окна — фактическое
    (допущение известного режима, аналогично плану закачки)."""
    pred = model.predict(
        injection=inj_full.to_numpy(),
        time=_time_axis(inj_full.index),
        pressure=bhp_full.to_numpy(),
    )
    return pd.DataFrame(np.maximum(pred, 0.0), index=inj_full.index, columns=producers)


def fit_predict_block(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    producers: list[int],
    injectors: list[int],
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CRM одного блока: подгонка до cutoff, ряд «история+горизонт».

    Возвращает (прогноз [даты x добывающие], связности f_ij
    [добывающие x нагнетательные]). Закачка на горизонте — фактическая
    (план ППД считается известным).
    """
    full_idx = liq.loc[FULL_START:].index
    end = full_idx[full_idx.get_loc(cutoff) + horizon]
    model = fit_block(liq, inj, producers, injectors, cutoff)
    pred_df = predict_block(model, inj.loc[FULL_START:end, injectors], producers)
    gains = pd.DataFrame(model.gains, index=producers, columns=injectors)
    return pred_df, gains


def crm_forecast(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CRM по всем блокам: (ряд история+горизонт по 33 скв., связности f_ij).

    Связности между блоками — структурные нули (экраны разломов).
    """
    preds, gains_parts = [], []
    for b in BLOCKS:
        prods = block_wells(b, injectors=False)
        prods = [w for w in prods if w in liq.columns]
        injs = block_wells(b, injectors=True)
        p, g = fit_predict_block(liq, inj, prods, injs, cutoff, horizon)
        preds.append(p)
        gains_parts.append(g)
    pred = pd.concat(preds, axis=1)[list(PRODUCERS)]
    gains = pd.concat(gains_parts).reindex(
        index=list(PRODUCERS), columns=sorted(INJECTORS)
    )
    return pred, gains.fillna(0.0)


def crm_full_field_gains(
    liq: pd.DataFrame, inj: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """CRM всего поля без блоковых ограничений — для проверки блоков данными."""
    hist = liq.loc[FULL_START:cutoff, list(PRODUCERS)]
    inj_hist = inj.loc[FULL_START:cutoff, sorted(INJECTORS)]
    model = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
    model.fit(hist.to_numpy(), inj_hist.to_numpy(), _time_axis(hist.index), num_cores=8)
    return pd.DataFrame(model.gains, index=list(PRODUCERS), columns=sorted(INJECTORS))


def same_block_share(gains: pd.DataFrame) -> float:
    """Доля суммарной связности, приходящаяся на пары внутри одного блока."""
    total = gains.to_numpy().sum()
    same = sum(
        gains.at[p, i]
        for p in gains.index
        for i in gains.columns
        if WELL_BLOCK[p] == WELL_BLOCK[i]
    )
    return float(same / total) if total > 0 else np.nan
