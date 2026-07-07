"""Прогноз вперёд (2015-12..2016-05) сетью TiDE — компонент ансамбля.

Запуск в окружении external/nf/.venv. Будущие признаки: закачка —
продлённый последний режим, ряд CRM — из results/crm_cov_201511.csv.
Выход: results/forward_tide.csv [well, date, oil_tpd, liq_tpd].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON
from timesoil.data import LAST_VALID
from timesoil.mlprep import field_dataset

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    frames, static_df, mats = field_dataset(OUT, CUTOFFS, fill_na=True)
    fwd_crm = pd.read_csv(OUT / f"crm_cov_{LAST_VALID:%Y%m}.csv",
                          index_col=0, parse_dates=True).rename(columns=int)
    future_idx = pd.date_range(LAST_VALID, periods=HORIZON + 1, freq="MS")[1:]

    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import TiDE

    preds = {}
    for tname, df_long in frames.items():
        model = TiDE(
            h=HORIZON, input_size=24, loss=MQLoss(level=[80]),
            futr_exog_list=["inj_alloc", "inj_block", "crm"],
            hist_exog_list=["pres_lag6"],
            stat_exog_list=["perm", "poro", "h_eff", "block"],
            scaler_type="robust", max_steps=1200, val_check_steps=50,
            early_stop_patience_steps=5, batch_size=32, random_seed=0,
            hidden_size=64, decoder_output_dim=8, dropout=0.3,
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=df_long, static_df=static_df, val_size=12)
        # будущие признаки: последний режим закачки, CRM вперёд, давление лагом
        futr_rows = []
        for w in df_long.unique_id.unique():
            hist = df_long[df_long.unique_id == w]
            last = hist.iloc[-1]
            for t in future_idx:
                crm_v = float(fwd_crm.at[t, int(w)]) if t in fwd_crm.index and int(w) in fwd_crm.columns else float(last["crm"])
                futr_rows.append(dict(
                    unique_id=w, ds=t,
                    inj_alloc=float(last["inj_alloc"]), inj_block=float(last["inj_block"]),
                    crm=crm_v, pres_lag6=float(last["pres_lag6"]),
                ))
        futr_df = pd.DataFrame(futr_rows)
        out = nf.predict(futr_df=futr_df)
        out = out.reset_index() if "unique_id" not in out.columns else out
        med = "TiDE-median" if "TiDE-median" in out.columns else "TiDE"
        preds[tname] = out.set_index(["unique_id", "ds"])[med]

    res = pd.DataFrame({
        "well": [i[0] for i in preds["oil_tpd"].index],
        "date": [i[1] for i in preds["oil_tpd"].index],
        "oil_tpd": np.maximum(preds["oil_tpd"].to_numpy(float), 0.0),
        "liq_tpd": np.maximum(preds["liq_tpd"].to_numpy(float), 0.0),
    })
    res.to_csv(OUT / "forward_tide.csv", index=False)
    print("forward_tide.csv:", res.shape)


if __name__ == "__main__":
    main()
