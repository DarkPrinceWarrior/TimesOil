"""Финальный прогноз вперёд: 2015-12..2016-05 (стек этапа 2).

Точечный прогноз: жидкость — CRM по блокам; нефть — жидкость x доля нефти
по закону Джентила. Интервалы: квантили TiRex-2 (blocks_cov_crm),
масштабированные конформными множителями results/interval_scale.csv и
перецентрированные на точечный прогноз стека.

Закачка на горизонте неизвестна -> продлеваем последний наблюдённый режим
(закачка в данных кусочно-постоянная, это штатный план ППД).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import HORIZON
from timesoil.crm import crm_forecast
from timesoil.data import (
    LAST_VALID,
    injection_matrix,
    load_monthly,
    producer_matrices,
    static_features,
    well_coords,
)
from timesoil.fractional import fit_gentil, predict_fo
from timesoil.tirex_runner import QUANTILES, forecast_tirex

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)

    future_idx = pd.date_range(LAST_VALID, periods=HORIZON + 1, freq="MS")[1:]
    inj_ext = pd.concat([inj, pd.DataFrame(
        [inj.iloc[-1]] * HORIZON, index=future_idx, columns=inj.columns)])

    # 1) жидкость: CRM по блокам (закачка на горизонте — продлённый режим)
    liq_ext = liq.reindex(liq.index.append(future_idx))
    crm_pred, _ = crm_forecast(liq_ext, inj_ext, LAST_VALID, HORIZON)

    # 2) нефть: доля нефти по Джентилу на накопленной адресной закачке
    alloc = allocate(inj_ext, hydro_weights(static_features(), well_coords()))
    days = pd.Series(alloc.index.days_in_month, index=alloc.index)
    w_cum = alloc.mul(days, axis=0).cumsum()

    # 3) интервалы: TiRex-2 + конформные множители
    scale = pd.read_csv(OUT / "interval_scale.csv") if (OUT / "interval_scale.csv").exists() else None
    tirex_fc = {}
    for target in ("oil_tpd", "liq_tpd"):
        tirex_fc[target] = forecast_tirex(
            model, mats[target], LAST_VALID, HORIZON, "blocks_cov_crm",
            inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj_ext, crm_mat=crm_pred,
        )

    OUT.mkdir(exist_ok=True)
    for target in ("liq_tpd", "oil_tpd"):
        fc = tirex_fc[target].copy()
        # точечный прогноз стека
        stack = []
        for w, g in fc.groupby("well"):
            w = int(w)
            liq_w = crm_pred.loc[future_idx, w].to_numpy(float)
            if target == "liq_tpd":
                point = liq_w
            else:
                params = fit_gentil(oil[w], liq[w], w_cum.loc[:LAST_VALID, w])
                fo = predict_fo(params, w_cum.loc[future_idx, w].to_numpy(float))
                point = liq_w * fo
            stack.append(pd.Series(point, index=g.index))
        fc["y_stack"] = pd.concat(stack)
        # интервалы: масштаб + перецентровка на y_stack
        if scale is not None:
            lam = scale[scale.target == target].set_index("step")["scale"]
            for qi, tau in enumerate(QUANTILES):
                col = f"q{int(tau * 100)}"
                lam_h = fc["step"].map(lam).to_numpy(float)
                fc[col] = np.maximum(
                    fc["y_stack"] + lam_h * (fc[col] - fc["y_pred"]), 0.0
                )
        fc["y_pred"] = fc["y_stack"]
        fc = fc.drop(columns=["y_stack"])
        fc.to_csv(OUT / f"forward_{target}.csv", index=False)
        field = fc.groupby("date")[["q10", "y_pred", "q90"]].sum()
        print(f"\n=== Прогноз по полю, {target} (т/сут; стек этапа 2) ===")
        print(field.round(1).to_string())


if __name__ == "__main__":
    main()
