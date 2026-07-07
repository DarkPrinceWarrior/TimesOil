"""Расширенная скользящая проверка на нашем поле: 14 срезов x 6 мес.

Модели без обучения: наивный, Арпс, CRM, CRM x Джентил, TiRex-2
(blocks_cov_crm), Chronos-2 (cov_crm). Ряды CRM сохраняются в
results/crm_cov_<срез>.csv (пополняют набор для обучаемых моделей).
Выходы: results/ext_<модель>_<цель>.csv. Запуск частями:
  uv run python scripts/run_extended.py --models crm base tirex chronos
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from timesoil import baselines as B
from timesoil.backtest import EXT_CUTOFFS, HORIZON, run_pointwise, summarize
from timesoil.crm import crm_forecast
from timesoil.data import injection_matrix, load_monthly, producer_matrices
from timesoil.fractional import fit_gentil, predict_fo
from timesoil.wells import PRODUCERS, block_wells

OUT = Path(__file__).resolve().parents[1] / "results"


def attach_truth(fc: pd.DataFrame, mat: pd.DataFrame) -> pd.DataFrame:
    fc["y_true"] = [mat.at[d, w] for d, w in zip(fc.date, fc.well)]
    return fc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["crm", "base", "tirex", "chronos", "frac"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)

    # --- CRM по всем срезам (+ сохранение рядов-ковариат) ---
    if "crm" in args.models:
        rows = []
        for cutoff in EXT_CUTOFFS:
            pred, _ = crm_forecast(liq, inj, cutoff, HORIZON)
            pred.to_csv(OUT / f"crm_cov_{cutoff:%Y%m}.csv")
            test = pred.index[pred.index > cutoff][:HORIZON]
            for step, dt in enumerate(test, 1):
                for w in PRODUCERS:
                    rows.append(dict(cutoff=cutoff, well=w, step=step, date=dt,
                                     y_true=float(liq.at[dt, w]), y_pred=float(pred.at[dt, w])))
            print(f"CRM {cutoff:%Y-%m}: готов", flush=True)
        res = pd.DataFrame(rows)
        res.to_csv(OUT / "ext_crm_liq_tpd.csv", index=False)
        print("=== ext liq | CRM ===")
        print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- наивный и Арпс ---
    if "base" in args.models:
        for target, mat in (("oil_tpd", oil), ("liq_tpd", liq)):
            for name, fc in {"naive": B.forecast_naive,
                             "arps36": partial(B.forecast_arps, k=36)}.items():
                res = run_pointwise(mat, fc, cutoffs=EXT_CUTOFFS)
                res.to_csv(OUT / f"ext_{name}_{target}.csv", index=False)
                print(f"=== ext {target} | {name} ===")
                print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- TiRex-2 blocks_cov_crm ---
    if "tirex" in args.models:
        from tirex2 import load_model

        from timesoil.tirex_runner import forecast_tirex

        model = load_model("NX-AI/TiRex-2", device="cpu")
        for target, mat in (("oil_tpd", oil), ("liq_tpd", liq)):
            parts = []
            for cutoff in EXT_CUTOFFS:
                crm_mat = pd.read_csv(OUT / f"crm_cov_{cutoff:%Y%m}.csv",
                                      index_col=0, parse_dates=True).rename(columns=int)
                fc = forecast_tirex(model, mat, cutoff, HORIZON, "blocks_cov_crm",
                                    inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj,
                                    crm_mat=crm_mat)
                fc["cutoff"] = cutoff
                parts.append(attach_truth(fc, mat))
            res = pd.concat(parts, ignore_index=True)
            res.to_csv(OUT / f"ext_tirex_{target}.csv", index=False)
            print(f"=== ext {target} | tirex ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- Chronos-2 cov_crm ---
    if "chronos" in args.models:
        from chronos import Chronos2Pipeline

        from timesoil.chronos_runner import forecast_chronos
        from timesoil.wells import WELL_BLOCK

        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        blocks = ("A", "B", "B2", "C", "D", "E")
        groups = [block_wells(b, injectors=False) for b in blocks]
        group_inj = [block_wells(b, injectors=True) for b in blocks]
        for target, mat in (("oil_tpd", oil), ("liq_tpd", liq)):
            parts = []
            for cutoff in EXT_CUTOFFS:
                crm_mat = pd.read_csv(OUT / f"crm_cov_{cutoff:%Y%m}.csv",
                                      index_col=0, parse_dates=True).rename(columns=int)
                fc = forecast_chronos(pipeline, mat, cutoff, HORIZON, "cov_crm",
                                      groups=groups, group_inj=group_inj,
                                      inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj,
                                      crm_mat=crm_mat)
                fc["cutoff"] = cutoff
                parts.append(attach_truth(fc, mat))
            res = pd.concat(parts, ignore_index=True)
            res.to_csv(OUT / f"ext_chronos_{target}.csv", index=False)
            print(f"=== ext {target} | chronos ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- нефть по Джентилу поверх CRM-жидкости ---
    if "frac" in args.models:
        from timesoil.allocation import allocate, hydro_weights
        from timesoil.data import static_features, well_coords

        alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
        days = pd.Series(alloc.index.days_in_month, index=alloc.index)
        w_cum = alloc.mul(days, axis=0).cumsum()
        src = pd.read_csv(OUT / "ext_crm_liq_tpd.csv", parse_dates=["date", "cutoff"])
        rows = []
        for cutoff in EXT_CUTOFFS:
            part = src[src.cutoff == cutoff]
            for w, g in part.groupby("well"):
                w = int(w)
                params = fit_gentil(oil.loc[:cutoff, w], liq.loc[:cutoff, w],
                                    w_cum.loc[:cutoff, w])
                dates = pd.DatetimeIndex(g.date)
                fo = predict_fo(params, w_cum.loc[dates, w].to_numpy(float))
                for (_, r), f in zip(g.iterrows(), fo):
                    rows.append(dict(cutoff=cutoff, well=w, step=int(r.step), date=r.date,
                                     y_true=float(oil.at[r.date, w]),
                                     y_pred=max(float(r.y_pred) * float(f), 0.0)))
        res = pd.DataFrame(rows)
        res.to_csv(OUT / "ext_frac_crm_oil_tpd.csv", index=False)
        print("=== ext oil | frac_crm ===")
        print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
