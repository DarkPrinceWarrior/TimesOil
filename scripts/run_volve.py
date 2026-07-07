"""Переносимость контура на Volve (реальные данные): 3 среза x 6 мес.

Модели: наивный / экспонента / Арпс; CRM (по скважине, старт — её первый
месяц); TiRex-2 (u, m, cov, cov_crm); нефть по Джентилу поверх жидкости.
Строки без факта (скважина остановлена) исключаются из оценки.
Выходы: results/volve_*.csv.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from timesoil import baselines as B
from timesoil.allocation import allocate
from timesoil.backtest import HORIZON, run_pointwise, summarize
from timesoil.crm import fit_block, predict_block
from timesoil.fractional import fit_gentil, predict_fo
from timesoil.tirex_runner import forecast_tirex
from timesoil.volve import INJECTORS_V, PRODUCERS_V, load_volve, uniform_weights

OUT = Path(__file__).resolve().parents[1] / "results"
CUTOFFS_V = (
    pd.Timestamp("2015-03-01"),
    pd.Timestamp("2015-09-01"),
    pd.Timestamp("2016-03-01"),
)
MIN_FIT = 10  # минимум месяцев истории для CRM по скважине


def drop_missing(res: pd.DataFrame) -> pd.DataFrame:
    return res.dropna(subset=["y_true"]).reset_index(drop=True)


def crm_stack(liq: pd.DataFrame, winj: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """CRM по каждой добывающей от её собственного старта."""
    parts = []
    end_pos = liq.index.get_loc(cutoff) + HORIZON
    for w in PRODUCERS_V:
        start = liq[w].first_valid_index()
        n_hist = liq.loc[start:cutoff, w].shape[0]
        if start is None or n_hist < MIN_FIT:
            continue
        model = fit_block(liq, winj, [w], list(INJECTORS_V), cutoff, start=start)
        inj_full = winj.loc[start: liq.index[end_pos], list(INJECTORS_V)]
        parts.append(predict_block(model, inj_full, [w]))
    return pd.concat(parts, axis=1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m = load_volve()
    oil, liq, winj = m["oil"], m["liq"], m["winj"]

    models = {"naive": B.forecast_naive, "exp24": partial(B.forecast_exp, k=24),
              "arps36": partial(B.forecast_arps, k=36)}
    for target, mat in (("oil", oil), ("liq", liq)):
        for name, fc in models.items():
            res = drop_missing(run_pointwise(mat, fc, cutoffs=CUTOFFS_V))
            res.to_csv(OUT / f"volve_{name}_{target}.csv", index=False)
            print(f"=== volve {target} | {name} ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- CRM ---
    crm_covs, rows = {}, []
    for cutoff in CUTOFFS_V:
        pred = crm_stack(liq, winj, cutoff)
        crm_covs[cutoff] = pred
        test = pred.index[pred.index > cutoff][:HORIZON]
        for step, dt in enumerate(test, 1):
            for w in pred.columns:
                rows.append(dict(cutoff=cutoff, well=w, step=step, date=dt,
                                 y_true=float(liq.at[dt, w]) if pd.notna(liq.at[dt, w]) else float("nan"),
                                 y_pred=float(pred.at[dt, w])))
    crm_res = drop_missing(pd.DataFrame(rows))
    crm_res.to_csv(OUT / "volve_crm_liq.csv", index=False)
    print("=== volve liq | CRM ===")
    print(summarize(crm_res).round(4).to_string(index=False), flush=True)

    # --- TiRex-2 ---
    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    variants = {
        "u": dict(variant="u", groups=[[w] for w in PRODUCERS_V]),
        "m": dict(variant="m", groups=[list(PRODUCERS_V)]),
        "cov": dict(variant="blocks_cov", groups=[list(PRODUCERS_V)],
                    group_inj=[list(INJECTORS_V)]),
        "cov_crm": dict(variant="blocks_cov_crm", groups=[list(PRODUCERS_V)],
                        group_inj=[list(INJECTORS_V)]),
    }
    tirex_results = {}
    for target, mat in (("oil", oil), ("liq", liq)):
        for name, kw in variants.items():
            parts = []
            for cutoff in CUTOFFS_V:
                fc = forecast_tirex(
                    model, mat, cutoff, HORIZON,
                    inj_mat=winj, pres_mat=None, inj_future=winj,
                    crm_mat=crm_covs[cutoff].reindex(columns=list(PRODUCERS_V)), **kw,
                )
                fc["cutoff"] = cutoff
                fc["y_true"] = [mat.at[d, w] for d, w in zip(fc.date, fc.well)]
                parts.append(fc)
            res = drop_missing(pd.concat(parts, ignore_index=True))
            res.to_csv(OUT / f"volve_tirex_{name}_{target}.csv", index=False)
            tirex_results[(name, target)] = res
            print(f"=== volve {target} | tirex-{name} ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- нефть по Джентилу ---
    alloc = allocate(winj, uniform_weights())
    days = pd.Series(alloc.index.days_in_month, index=alloc.index)
    w_cum = alloc.mul(days, axis=0).cumsum()
    for src_name, src in (("crm", crm_res), ("tirex", tirex_results[("cov_crm", "liq")])):
        rows = []
        for cutoff in CUTOFFS_V:
            part = src[src.cutoff == cutoff]
            for w, g in part.groupby("well"):
                params = fit_gentil(oil[w].loc[:cutoff].dropna(),
                                    liq[w].loc[:cutoff].dropna(),
                                    w_cum.loc[:cutoff, w], k=36)
                dates = pd.DatetimeIndex(g.date)
                fo = predict_fo(params, w_cum.loc[dates, w].to_numpy(float))
                for (_, r), f in zip(g.iterrows(), fo):
                    yt = oil.at[r.date, w]
                    if pd.isna(yt):
                        continue
                    rows.append(dict(cutoff=cutoff, well=w, step=int(r.step), date=r.date,
                                     y_true=float(yt),
                                     y_pred=max(float(r.y_pred) * float(f), 0.0)))
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f"volve_frac_{src_name}_oil.csv", index=False)
        print(f"=== volve oil | frac_{src_name} (Джентил) ===")
        print(summarize(res).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
