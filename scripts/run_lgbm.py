"""LightGBM (mlforecast) на трёх полигонах: 3 среза x 6 мес.

Признаки: адресная закачка и закачка блока (известные наперёд), ряд CRM
(известный наперёд), давление лагом 6 мес (прошлое, но известно на весь
горизонт h<=6), статика скважины + блок; лаги цели 1..12 и скользящие
средние. Глобальная модель по всем скважинам полигона, прямой прогноз —
рекурсивный (лаги цели на горизонте — из собственных прогнозов).

Запуск: uv run python scripts/run_lgbm.py [--polygons field unisim volve]
Выходы: results/[<полигон>_]lgbm_<цель>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from timesoil.backtest import HORIZON, summarize
from timesoil.mlprep import combined_crm, long_frame

OUT = Path(__file__).resolve().parents[1] / "results"

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
    random_state=0, verbosity=-1,
)


def run_polygon(name: str, prefix: str, targets: dict, alloc, blk_inj_sum,
                crm_covs: dict, pres, static_df: pd.DataFrame | None, cutoffs,
                n_windows: int = 3, step_size: int = HORIZON) -> None:
    crm_mat = combined_crm(crm_covs, cutoffs)
    frames = long_frame(targets, alloc, blk_inj_sum, crm_mat, pres)
    for tname, df_long in frames.items():
        df_long = df_long.copy()
        if static_df is not None:
            df_long = df_long.merge(static_df, on="unique_id", how="left")
        mlf = MLForecast(
            models={"lgbm": lgb.LGBMRegressor(**LGB_PARAMS)},
            freq=pd.infer_freq(sorted(df_long.ds.unique())[:24]) or "MS",
            lags=[1, 2, 3, 4, 5, 6, 12],
            lag_transforms={1: [RollingMean(3), RollingMean(6)]},
        )
        static_cols = list(static_df.columns.drop("unique_id")) if static_df is not None else []
        cv = mlf.cross_validation(
            df_long, n_windows=n_windows, h=HORIZON, step_size=step_size, refit=True,
            static_features=static_cols,
        )
        cv = cv.dropna(subset=["y"]).reset_index(drop=True)
        res = pd.DataFrame(dict(
            cutoff=cv["cutoff"], well=cv["unique_id"], date=cv["ds"],
            y_true=cv["y"].astype(float),
            y_pred=np.maximum(cv["lgbm"].astype(float), 0.0),
        ))
        res["step"] = res.groupby(["cutoff", "well"]).cumcount() + 1
        res.to_csv(OUT / f"{prefix}lgbm_{tname}.csv", index=False)
        print(f"=== {name} {tname} | lgbm ===")
        print(summarize(res).round(4).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygons", nargs="*", default=["field", "unisim", "volve"])
    ap.add_argument("--ext", action="store_true", help="расширенная проверка (14 срезов, только поле)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if "field" in args.polygons:
        from timesoil.allocation import allocate, hydro_weights
        from timesoil.backtest import CUTOFFS, EXT_CUTOFFS
        from timesoil.data import (
            injection_matrix, load_monthly, producer_matrices,
            static_features, well_coords,
        )
        from timesoil.wells import PRODUCERS, WELL_BLOCK, block_wells

        df = load_monthly()
        mats = producer_matrices(df)
        inj = injection_matrix(df)
        alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
        blk_sum = pd.DataFrame({
            w: inj[block_wells(WELL_BLOCK[w], injectors=True)].sum(axis=1)
            for w in PRODUCERS
        })
        field_cutoffs = EXT_CUTOFFS if args.ext else CUTOFFS
        crm_covs = {}
        for cutoff in field_cutoffs:
            p = OUT / f"crm_cov_{cutoff:%Y%m}.csv"
            if p.exists():
                crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)
        st = static_features().reset_index()
        st = st[st.well.isin(PRODUCERS)]
        static_df = pd.DataFrame(dict(
            unique_id=st.well.astype(str), perm=st.perm_md, poro=st.poro,
            h_eff=st.h_eff, block=st.block.astype("category").cat.codes,
        ))
        run_polygon("field", "ext_" if args.ext else "",
                    {"oil_tpd": mats["oil_tpd"], "liq_tpd": mats["liq_tpd"]},
                    alloc, blk_sum, crm_covs, mats["p_res"], static_df, field_cutoffs,
                    n_windows=len(field_cutoffs), step_size=2 if args.ext else HORIZON)

    if "unisim" in args.polygons:
        from timesoil.allocation import allocate
        from timesoil.unisim import (
            BLOCKS_U, PRODUCERS_U, coords_unisim, crm_stack,
            distance_weights, load_unisim,
        )

        m = load_unisim()
        cutoffs = (pd.Timestamp("2022-11-30"), pd.Timestamp("2023-05-31"),
                   pd.Timestamp("2023-11-30"))
        alloc = allocate(m["winj"], distance_weights(coords_unisim()))
        inj_of = {w: b["injectors"] for b in BLOCKS_U.values() for w in b["producers"]}
        blk_sum = pd.DataFrame({w: m["winj"][inj_of[w]].sum(axis=1) for w in PRODUCERS_U})
        crm_covs = {c: crm_stack(m["liq"], m["winj"], c, HORIZON) for c in cutoffs}
        blocks_cat = {w: i for i, b in enumerate(BLOCKS_U.values()) for w in b["producers"]}
        static_df = pd.DataFrame(dict(
            unique_id=[str(w) for w in PRODUCERS_U],
            block=[blocks_cat[w] for w in PRODUCERS_U],
        ))
        run_polygon("unisim", "unisim_", {"oil": m["oil"], "liq": m["liq"]},
                    alloc, blk_sum, crm_covs, m["bhp"], static_df, cutoffs)

    if "volve" in args.polygons:
        from timesoil.allocation import allocate
        from timesoil.volve import PRODUCERS_V, crm_stack, load_volve, uniform_weights

        m = load_volve()
        cutoffs = (pd.Timestamp("2015-03-01"), pd.Timestamp("2015-09-01"),
                   pd.Timestamp("2016-03-01"))
        alloc = allocate(m["winj"], uniform_weights())
        blk_sum = pd.DataFrame({w: m["winj"].sum(axis=1) for w in PRODUCERS_V})
        crm_covs = {c: crm_stack(m["liq"], m["winj"], c, HORIZON).reindex(
            columns=list(PRODUCERS_V)) for c in cutoffs}
        run_polygon("volve", "volve_", {"oil": m["oil"], "liq": m["liq"]},
                    alloc, blk_sum, crm_covs, None, None, cutoffs)


if __name__ == "__main__":
    main()
