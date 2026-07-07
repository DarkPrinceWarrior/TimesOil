"""Финальный прогноз вперёд: 2015-12..2016-05 — ансамбль этапа 5.

Компоненты (медианы): CRM-жидкость и нефть по Джентилу; TiRex-2
(blocks_cov_crm); Chronos-2 (cov_crm); LightGBM; TiDE (results/forward_tide.csv,
считается отдельно в окружении сетей). Веса — results/ensemble_weights.csv
(неотрицательная регрессия по 14 срезам). Интервалы — эмпирические
мультипликативные по горизонтам из остатков ансамбля на 14 срезах
(results/ext_ens_nnls_*.csv), без утечки.

Закачка на горизонте — продлённый последний режим (план ППД).
Выходы: results/forward_<цель>.csv (ансамбль + интервалы),
results/forward_components_<цель>.csv (все компоненты).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import CUTOFFS, HORIZON
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
from timesoil.mlprep import combined_crm, long_frame
from timesoil.wells import PRODUCERS, WELL_BLOCK, block_wells

OUT = Path(__file__).resolve().parents[1] / "results"
EPS = 1e-6


def main() -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq, pres = mats["oil_tpd"], mats["liq_tpd"], mats["p_res"]
    inj = injection_matrix(df)

    future_idx = pd.date_range(LAST_VALID, periods=HORIZON + 1, freq="MS")[1:]
    inj_ext = pd.concat([inj, pd.DataFrame(
        [inj.iloc[-1]] * HORIZON, index=future_idx, columns=inj.columns)])
    liq_ext = liq.reindex(liq.index.append(future_idx))

    # --- 1) CRM вперёд (жидкость) + Джентил (нефть) ---
    crm_pred, _ = crm_forecast(liq_ext, inj_ext, LAST_VALID, HORIZON)
    crm_pred.to_csv(OUT / f"crm_cov_{LAST_VALID:%Y%m}.csv")
    w_alloc = hydro_weights(static_features(), well_coords())
    alloc_ext = allocate(inj_ext, w_alloc)
    days_ext = pd.Series(alloc_ext.index.days_in_month, index=alloc_ext.index)
    w_cum = alloc_ext.mul(days_ext, axis=0).cumsum()
    frac_oil = {}
    for w in PRODUCERS:
        params = fit_gentil(oil[w], liq[w], w_cum.loc[:LAST_VALID, w])
        fo = predict_fo(params, w_cum.loc[future_idx, w].to_numpy(float))
        frac_oil[w] = crm_pred.loc[future_idx, w].to_numpy(float) * fo
    comp = {
        "oil_tpd": {"frac_crm": pd.DataFrame(frac_oil, index=future_idx)},
        "liq_tpd": {"crm": crm_pred.loc[future_idx, list(PRODUCERS)]},
    }

    # --- 2) TiRex-2 и Chronos-2 ---
    from chronos import Chronos2Pipeline
    from tirex2 import load_model

    from timesoil.chronos_runner import forecast_chronos
    from timesoil.tirex_runner import forecast_tirex

    tirex = load_model("NX-AI/TiRex-2", device="cpu")
    chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    blocks = ("A", "B", "B2", "C", "D", "E")
    groups = [block_wells(b, injectors=False) for b in blocks]
    group_inj = [block_wells(b, injectors=True) for b in blocks]
    for tname, mat in (("oil_tpd", oil), ("liq_tpd", liq)):
        ft = forecast_tirex(tirex, mat, LAST_VALID, HORIZON, "blocks_cov_crm",
                            inj_mat=inj, pres_mat=pres, inj_future=inj_ext,
                            crm_mat=crm_pred)
        comp[tname]["tirex"] = ft.pivot(index="date", columns="well", values="y_pred")[list(PRODUCERS)]
        fchr = forecast_chronos(chronos, mat, LAST_VALID, HORIZON, "cov_crm",
                                groups=groups, group_inj=group_inj,
                                inj_mat=inj, pres_mat=pres, inj_future=inj_ext,
                                crm_mat=crm_pred)
        comp[tname]["chronos"] = fchr.pivot(index="date", columns="well", values="y_pred")[list(PRODUCERS)]

    # --- 3) LightGBM ---
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    from run_lgbm import LGB_PARAMS  # noqa: F401 (одни и те же параметры)

    blk_sum_ext = pd.DataFrame({
        w: inj_ext[block_wells(WELL_BLOCK[w], injectors=True)].sum(axis=1)
        for w in PRODUCERS
    })
    crm_covs = {c: pd.read_csv(OUT / f"crm_cov_{c:%Y%m}.csv", index_col=0,
                               parse_dates=True).rename(columns=int) for c in CUTOFFS}
    crm_hist = combined_crm(crm_covs, CUTOFFS)
    frames = long_frame({"oil_tpd": oil, "liq_tpd": liq}, alloc_ext, blk_sum_ext,
                        crm_hist, pres)
    st = static_features().reset_index()
    st = st[st.well.isin(PRODUCERS)]
    static_df = pd.DataFrame(dict(
        unique_id=st.well.astype(str), perm=st.perm_md, poro=st.poro,
        h_eff=st.h_eff, block=st.block.astype("category").cat.codes,
    ))
    pres_lag6_fut = pres.shift(6).reindex(future_idx)
    for tname, mat in (("oil_tpd", oil), ("liq_tpd", liq)):
        df_long = frames[tname].merge(static_df, on="unique_id", how="left")
        mlf = MLForecast(models={"lgbm": lgb.LGBMRegressor(**LGB_PARAMS)}, freq="MS",
                         lags=[1, 2, 3, 4, 5, 6, 12],
                         lag_transforms={1: [RollingMean(3), RollingMean(6)]})
        mlf.fit(df_long, static_features=list(static_df.columns.drop("unique_id")))
        x_rows = []
        for w in df_long.unique_id.unique():
            for t in future_idx:
                x_rows.append(dict(
                    unique_id=w, ds=t,
                    inj_alloc=float(alloc_ext.at[t, int(w)]),
                    inj_block=float(blk_sum_ext.at[t, int(w)]),
                    crm=float(crm_pred.at[t, int(w)]),
                    pres_lag6=float(pres_lag6_fut.at[t, int(w)]),
                ))
        fc = mlf.predict(h=HORIZON, X_df=pd.DataFrame(x_rows))
        piv = fc.pivot(index="ds", columns="unique_id", values="lgbm")
        piv.columns = [int(c) for c in piv.columns]
        comp[tname]["lgbm"] = piv[list(PRODUCERS)].clip(lower=0.0)

    # --- 4) TiDE (посчитан отдельно) ---
    tide_path = OUT / "forward_tide.csv"
    if tide_path.exists():
        td = pd.read_csv(tide_path, parse_dates=["date"])
        for tname in ("oil_tpd", "liq_tpd"):
            piv = td.pivot(index="date", columns="well", values=tname)
            piv.columns = [int(c) for c in piv.columns]
            comp[tname]["tide"] = piv[list(PRODUCERS)].clip(lower=0.0)
    else:
        print("! forward_tide.csv не найден — TiDE исключён, веса перенормируются")

    # --- 5) ансамбль + эмпирические интервалы по горизонтам ---
    weights = pd.read_csv(OUT / "ensemble_weights.csv")
    for tname in ("oil_tpd", "liq_tpd"):
        wsub = weights[weights.target == tname].set_index("model")["weight"]
        wsub = wsub[[m for m in wsub.index if m in comp[tname]]]
        wsub = wsub / wsub.sum()
        ens = sum(comp[tname][m] * float(wv) for m, wv in wsub.items())

        resid = pd.read_csv(OUT / f"ext_ens_nnls_{tname}.csv", parse_dates=["date", "cutoff"])
        resid = resid[resid.y_pred > EPS]
        ratio = resid.y_true / resid.y_pred
        q_lo = ratio.groupby(resid.step).quantile(0.1)
        q_hi = ratio.groupby(resid.step).quantile(0.9)

        rows = []
        for w in PRODUCERS:
            for step, t in enumerate(future_idx, 1):
                y = float(ens.at[t, w])
                rows.append(dict(
                    well=w, step=step, date=t, y_pred=y,
                    q10=max(y * float(q_lo[step]), 0.0),
                    q90=y * float(q_hi[step]),
                ))
        out = pd.DataFrame(rows)
        out.to_csv(OUT / f"forward_{tname}.csv", index=False)
        comp_rows = []
        for m, c in comp[tname].items():
            for w in PRODUCERS:
                for t in future_idx:
                    comp_rows.append(dict(model=m, well=w, date=t, y_pred=float(c.at[t, w])))
        pd.DataFrame(comp_rows).to_csv(OUT / f"forward_components_{tname}.csv", index=False)
        field = out.groupby("date")[["q10", "y_pred", "q90"]].sum()
        print(f"\n=== Прогноз по полю, {tname} (т/сут; ансамбль NNLS) ===")
        print(field.round(1).to_string())
        print("веса:", dict(wsub.round(3)))


if __name__ == "__main__":
    main()
