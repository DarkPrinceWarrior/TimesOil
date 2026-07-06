"""CRM: бэктест жидкости, проверка блоков данными, ковариаты для TiRex-2.

Выходы:
- results/crm_liq_tpd.csv        — бэктест CRM (формат общий);
- results/crm_gains_<срез>.csv   — связности f_ij блочного CRM;
- results/crm_cov_<срез>.csv     — ряд CRM «история+горизонт» (ковариата);
- results/crm_fullfield_gains.csv + доля внутриблочной связности (проверка).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.crm import crm_forecast, crm_full_field_gains, same_block_share
from timesoil.data import LAST_VALID, injection_matrix, load_monthly, producer_matrices

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    liq = producer_matrices(df)["liq_tpd"]
    inj = injection_matrix(df)

    rows = []
    for cutoff in CUTOFFS:
        pred, gains = crm_forecast(liq, inj, cutoff, HORIZON)
        gains.to_csv(OUT / f"crm_gains_{cutoff:%Y%m}.csv")
        pred.to_csv(OUT / f"crm_cov_{cutoff:%Y%m}.csv")
        test_dates = pred.index[pred.index > cutoff][:HORIZON]
        for step, dt in enumerate(test_dates, 1):
            for w in pred.columns:
                rows.append(
                    dict(cutoff=cutoff, well=w, step=step, date=dt,
                         y_true=float(liq.at[dt, w]), y_pred=float(pred.at[dt, w]))
                )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "crm_liq_tpd.csv", index=False)
    print("=== liq_tpd | CRM (по блокам) ===")
    print(summarize(res).round(4).to_string(index=False))

    # ковариата и для финального прогноза вперёд
    pred_fwd, _ = crm_forecast(liq, inj, LAST_VALID, 0)
    pred_fwd.to_csv(OUT / f"crm_cov_{LAST_VALID:%Y%m}.csv")

    # проверка блоков данными: CRM всего поля без ограничений
    gains_ff = crm_full_field_gains(liq, inj, CUTOFFS[-1])
    gains_ff.to_csv(OUT / "crm_fullfield_gains.csv")
    print(f"\nCRM всего поля: доля связности внутри блоков = {same_block_share(gains_ff):.3f}")
    print("(случайное распределение дало бы ~долю внутриблочных пар)")


if __name__ == "__main__":
    main()
