"""CRMP с настройкой на пластовое давление: бэктест и проверка связности.

Выходы:
- results/crmp_p_liq_tpd.csv          — бэктест жидкости (формат общий);
- results/crmp_p_pres_atm.csv         — прогноз пластового давления на тех же окнах;
- results/crmp_p_gains_<срез>.csv     — связности f_ij блочной модели;
- results/crmp_p_fullfield_gains.csv  — связности без блоковых ограничений
  (проверка блоков данными: доля внутриблочной связности).

Печатает: сводку WAPE жидкости (сравнение с CRM-базой), WAPE/MAE прогноза
давления, корреляцию f_ij с базовым CRM (results/crm_gains_201505.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.crmp_pressure import (
    crmp_pressure_forecast,
    crmp_pressure_full_field_gains,
)
from timesoil.crm import same_block_share
from timesoil.data import injection_matrix, load_monthly, producer_matrices
from timesoil.metrics import wape
from timesoil.wells import WELL_BLOCK

OUT = Path(__file__).resolve().parents[1] / "results"
W_P = 0.5          # вес невязки по давлению (подобран на 3 срезах)
USE_T = False      # межскважинные проводимости: выигрыша нет, время x5


def _rows(mat_true: pd.DataFrame, pred: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict]:
    test_dates = pred.index[pred.index > cutoff][:HORIZON]
    return [
        dict(cutoff=cutoff, well=w, step=step, date=dt,
             y_true=float(mat_true.at[dt, w]), y_pred=float(pred.at[dt, w]))
        for step, dt in enumerate(test_dates, 1)
        for w in pred.columns
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    liq, pres = mats["liq_tpd"], mats["p_res"]
    inj = injection_matrix(df)

    rows_q, rows_p = [], []
    gains_by_cutoff: dict[pd.Timestamp, pd.DataFrame] = {}
    for cutoff in CUTOFFS:
        pred_q, pred_p, gains = crmp_pressure_forecast(
            liq, inj, pres, cutoff, HORIZON, w_p=W_P, use_T=USE_T
        )
        gains.to_csv(OUT / f"crmp_p_gains_{cutoff:%Y%m}.csv")
        gains_by_cutoff[cutoff] = gains
        rows_q += _rows(liq, pred_q, cutoff)
        rows_p += _rows(pres, pred_p, cutoff)

    res_q = pd.DataFrame(rows_q)
    res_q.to_csv(OUT / "crmp_p_liq_tpd.csv", index=False)
    print("=== liq_tpd | CRMP c пластовым давлением (по блокам) ===")
    print(summarize(res_q).round(4).to_string(index=False))

    base = pd.read_csv(OUT / "crm_liq_tpd.csv")
    print(f"\nCRM-база: WAPE={wape(base.y_true, base.y_pred):.4f}"
          f" -> CRMP-давление: WAPE={wape(res_q.y_true, res_q.y_pred):.4f}")

    res_p = pd.DataFrame(rows_p)
    res_p.to_csv(OUT / "crmp_p_pres_atm.csv", index=False)
    print("\n=== пластовое давление на контрольных окнах ===")
    for cutoff, g in res_p.groupby("cutoff"):
        print(f"  срез {pd.Timestamp(cutoff):%Y-%m}: WAPE={wape(g.y_true, g.y_pred):.4f}"
              f"  MAE={np.abs(g.y_true - g.y_pred).mean():.2f} атм")
    print(f"  ВСЕГО: WAPE={wape(res_p.y_true, res_p.y_pred):.4f}"
          f"  MAE={np.abs(res_p.y_true - res_p.y_pred).mean():.2f} атм")

    # --- связности против базового CRM (внутриблочные пары) ---
    cut = CUTOFFS[-1]
    base_gains = pd.read_csv(OUT / f"crm_gains_{cut:%Y%m}.csv", index_col=0)
    base_gains.columns = base_gains.columns.astype(int)
    ours = gains_by_cutoff[cut]
    pairs = [
        (p, i)
        for p in ours.index
        for i in ours.columns
        if WELL_BLOCK[p] == WELL_BLOCK[i]
    ]
    a = np.array([ours.at[p, i] for p, i in pairs])
    b = np.array([base_gains.at[p, i] for p, i in pairs])
    print(f"\n=== связности f_ij против CRM-базы (срез {cut:%Y-%m}, "
          f"{len(pairs)} внутриблочных пар) ===")
    print(f"  Пирсон={pearsonr(a, b)[0]:.3f}  Спирмен={spearmanr(a, b)[0]:.3f}")
    print(f"  сумма f по нагнетательным: база={base_gains.to_numpy().sum():.2f},"
          f" наша={ours.to_numpy().sum():.2f}")

    # --- всё поле без блоковых ограничений: блоки, увиденные данными ---
    gains_ff = crmp_pressure_full_field_gains(liq, inj, pres, cut, w_p=W_P)
    gains_ff.to_csv(OUT / "crmp_p_fullfield_gains.csv")
    print(f"\nCRMP-давление всего поля: доля связности внутри блоков = "
          f"{same_block_share(gains_ff):.3f}")


if __name__ == "__main__":
    main()
