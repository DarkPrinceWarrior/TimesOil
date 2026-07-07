"""Прогон Chronos-2 (zero-shot, amazon/chronos-2) — зеркало tirex_runner.

Табличный интерфейс модели: контекст — длинная таблица [item_id, timestamp,
target, колонки-ковариаты]; колонки, присутствующие и в future_df, —
известные наперёд (закачка, ряд CRM), остальные — исторические (давления).
Варианты:
- "u"       — по скважине, без ковариат, без со-обучения;
- "m"       — все скважины совместно (cross_learning), без ковариат;
- "cov"     — закачка блока (future) + давления (past), со-обучение;
- "cov_crm" — cov + ряд CRM скважины как известный наперёд.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = [round(0.1 * i, 1) for i in range(1, 10)]


def forecast_chronos(
    pipeline,
    target_mat: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    variant: str,
    groups: list[list],
    group_inj: list[list] | None = None,
    inj_mat: pd.DataFrame | None = None,
    pres_mat: pd.DataFrame | None = None,
    inj_future: pd.DataFrame | None = None,
    crm_mat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Возвращает длинную таблицу [well, step, date, y_pred, q10..q90]."""
    ctx = target_mat.loc[:cutoff]
    after = target_mat.index[target_mat.index > cutoff]
    if len(after) >= horizon:
        dates_future = after[:horizon]
    else:
        dates_future = pd.date_range(cutoff, periods=horizon + 1, freq="MS")[1:]

    use_cov = variant in ("cov", "cov_crm")
    rows_out = []
    for gi, wells in enumerate(groups):
        blk_inj = group_inj[gi] if (use_cov and group_inj is not None) else []
        recs, fut_recs = [], []
        for w in wells:
            s = ctx[w].dropna()
            if s.empty:
                continue
            crm_w = None
            if variant == "cov_crm" and crm_mat is not None and w in crm_mat.columns:
                # до старта CRM-ряда — продление первого значения назад
                crm_w = crm_mat[w].reindex(s.index.append(dates_future)).bfill().ffill()
            for t, y in s.items():
                r = {"item_id": str(w), "timestamp": t, "target": float(y)}
                if use_cov:
                    for i in blk_inj:
                        r[f"inj_{i}"] = float(inj_mat.at[t, i])
                    if pres_mat is not None and w in pres_mat.columns:
                        pv = pres_mat.at[t, w]
                        r["pres"] = float(pv) if pd.notna(pv) else np.nan
                    if crm_w is not None:
                        r["crm"] = float(crm_w.at[t])
                recs.append(r)
            for t in dates_future:
                fr = {"item_id": str(w), "timestamp": t}
                if use_cov:
                    for i in blk_inj:
                        fr[f"inj_{i}"] = float(inj_future.at[t, i])
                    if crm_w is not None:
                        fr["crm"] = float(crm_w.at[t])
                fut_recs.append(fr)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        fut = pd.DataFrame(fut_recs) if use_cov else None
        out = pipeline.predict_df(
            df,
            future_df=fut,
            prediction_length=horizon,
            quantile_levels=QUANTILES,
            cross_learning=(variant != "u"),
            batch_size=100,
        )
        name_map = {str(w): w for w in wells}
        for w_str, g in out.groupby("item_id"):
            g = g.sort_values("timestamp")
            for step, (_, r) in enumerate(g.iterrows(), 1):
                row = dict(
                    well=name_map[w_str], step=step, date=r["timestamp"],
                    y_pred=max(float(r["0.5"]), 0.0),
                )
                for q in QUANTILES:
                    row[f"q{int(q * 100)}"] = max(float(r[str(q)]), 0.0)
                rows_out.append(row)
    return pd.DataFrame(rows_out)
