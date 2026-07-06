"""Нефть = жидкость x (1 - обводнённость): закон Джентила поверх прогнозов жидкости.

Накопленная закачка на скважину — адресное распределение по весам
гидропроводности (allocation). Источники жидкости: CRM (crm_liq_tpd.csv)
и TiRex-2 blocks_cov_crm (tirex_blocks_cov_crm_liq_tpd.csv).
Выходы: results/frac_crm_oil_tpd.csv, results/frac_tirex_oil_tpd.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import CUTOFFS, summarize
from timesoil.data import (
    injection_matrix,
    load_monthly,
    producer_matrices,
    static_features,
    well_coords,
)
from timesoil.fractional import fit_gentil, predict_fo

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)
    alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
    days = pd.Series(liq.index.days_in_month, index=liq.index)
    w_cum = alloc.mul(days, axis=0).cumsum()  # накопленная закачка, м3

    for src, out_name in (
        ("crm_liq_tpd.csv", "frac_crm_oil_tpd.csv"),
        ("tirex_blocks_cov_crm_liq_tpd.csv", "frac_tirex_oil_tpd.csv"),
    ):
        path = OUT / src
        if not path.exists():
            print(f"нет {src} — пропуск")
            continue
        liq_pred = pd.read_csv(path, parse_dates=["date", "cutoff"])
        rows = []
        for cutoff in CUTOFFS:
            part = liq_pred[liq_pred.cutoff == cutoff]
            for w, g in part.groupby("well"):
                w = int(w)
                params = fit_gentil(oil.loc[:cutoff, w], liq.loc[:cutoff, w], w_cum.loc[:cutoff, w])
                dates = pd.DatetimeIndex(g.date)
                fo = predict_fo(params, w_cum.loc[dates, w].to_numpy(float))
                oil_hat = np.maximum(g.y_pred.to_numpy(float) * fo, 0.0)
                for (_, r), oh in zip(g.iterrows(), oil_hat):
                    rows.append(dict(cutoff=cutoff, well=w, step=int(r.step), date=r.date,
                                     y_true=float(oil.at[r.date, w]), y_pred=float(oh)))
        res = pd.DataFrame(rows)
        res.to_csv(OUT / out_name, index=False)
        print(f"\n=== oil_tpd | {out_name.removesuffix('.csv')} (жидкость: {src}) ===")
        print(summarize(res).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
