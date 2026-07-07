"""Переносимость контура на эталон UNISIM-I-H: 3 среза x 6 мес.

Модели: наивный / экспонента / Арпс; CRM по блокам (f3 — экран);
TiRex-2 (u, m, blocks_cov, blocks_cov_crm); нефть по Джентилу поверх
жидкости CRM и TiRex-2. Единицы: м3/сут. Выходы: results/unisim_*.csv.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path


import pandas as pd

from timesoil import baselines as B
from timesoil.backtest import HORIZON, run_pointwise, summarize
from timesoil.crm import (
    fit_block,
    fit_block_compensated,
    predict_block,
    predict_block_compensated,
)
from timesoil.fractional import fit_gentil, predict_fo
from timesoil.tirex_runner import forecast_tirex
from timesoil.unisim import (
    BLOCKS_U,
    INJECTORS_U,
    PRODUCERS_U,
    coords_unisim,
    distance_weights,
    load_unisim,
)

OUT = Path(__file__).resolve().parents[1] / "results"
CUTOFFS_U = (
    pd.Timestamp("2022-11-30"),
    pd.Timestamp("2023-05-31"),
    pd.Timestamp("2023-11-30"),
)


def block_start(liq: pd.DataFrame, producers: list[str]) -> pd.Timestamp:
    """Первый месяц, с которого работают все добывающие блока."""
    return max(liq[w].first_valid_index() for w in producers)


def crm_stack(liq: pd.DataFrame, winj: pd.DataFrame, cutoff: pd.Timestamp,
              bhp: pd.DataFrame | None = None):
    """CRM по двум блокам (с компенсацией забойного давления, если дано bhp):
    ряд «история+горизонт»."""
    parts = []
    for b in BLOCKS_U.values():
        prods, injs = b["producers"], b["injectors"]
        start = block_start(liq, prods)
        end_pos = liq.index.get_loc(cutoff) + HORIZON
        inj_full = winj.loc[start: liq.index[end_pos], injs]
        if bhp is None:
            model = fit_block(liq, winj, prods, injs, cutoff, start=start)
            parts.append(predict_block(model, inj_full, prods))
        else:
            model = fit_block_compensated(liq, winj, bhp, prods, injs, cutoff, start=start)
            bhp_full = bhp.loc[start: liq.index[end_pos], prods]
            parts.append(predict_block_compensated(model, inj_full, bhp_full, prods))
    return pd.concat(parts, axis=1, sort=False)[list(PRODUCERS_U)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m = load_unisim()
    oil, liq, bhp, winj = m["oil"], m["liq"], m["bhp"], m["winj"]

    # --- эталоны ---
    models = {"naive": B.forecast_naive, "exp24": partial(B.forecast_exp, k=24),
              "arps36": partial(B.forecast_arps, k=36)}
    for target, mat in (("oil", oil), ("liq", liq)):
        for name, fc in models.items():
            res = run_pointwise(mat, fc, cutoffs=CUTOFFS_U)
            res.to_csv(OUT / f"unisim_{name}_{target}.csv", index=False)
            s = summarize(res)
            print(f"=== unisim {target} | {name} ===")
            print(s.round(4).tail(1).to_string(index=False), flush=True)

    # --- CRM по блокам (жидкость) + ковариата для TiRex-2 ---
    crm_covs, rows = {}, []
    for cutoff in CUTOFFS_U:
        pred = crm_stack(liq, winj, cutoff)
        crm_covs[cutoff] = pred
        test = pred.index[pred.index > cutoff][:HORIZON]
        for step, dt in enumerate(test, 1):
            for w in PRODUCERS_U:
                rows.append(dict(cutoff=cutoff, well=w, step=step, date=dt,
                                 y_true=float(liq.at[dt, w]), y_pred=float(pred.at[dt, w])))
    crm_res = pd.DataFrame(rows)
    crm_res.to_csv(OUT / "unisim_crm_liq.csv", index=False)
    print("=== unisim liq | CRM (по блокам) ===")
    print(summarize(crm_res).round(4).to_string(index=False), flush=True)

    # --- CRM-P: с компенсацией забойного давления (Pзаб контрольного окна —
    # фактическое: допущение известного режима) ---
    rows = []
    for cutoff in CUTOFFS_U:
        pred = crm_stack(liq, winj, cutoff, bhp=bhp)
        test = pred.index[pred.index > cutoff][:HORIZON]
        for step, dt in enumerate(test, 1):
            for w in PRODUCERS_U:
                rows.append(dict(cutoff=cutoff, well=w, step=step, date=dt,
                                 y_true=float(liq.at[dt, w]), y_pred=float(pred.at[dt, w])))
    crmp_res = pd.DataFrame(rows)
    crmp_res.to_csv(OUT / "unisim_crmp_liq.csv", index=False)
    print("=== unisim liq | CRM-P (компенсация Pзаб) ===")
    print(summarize(crmp_res).round(4).to_string(index=False), flush=True)

    # --- TiRex-2 ---
    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    grp_u = [[w] for w in PRODUCERS_U]
    grp_blocks = [b["producers"] for b in BLOCKS_U.values()]
    grp_blocks_inj = [b["injectors"] for b in BLOCKS_U.values()]
    variants = {
        "u": dict(variant="u", groups=grp_u),
        "m": dict(variant="m", groups=[list(PRODUCERS_U)]),
        "blocks_cov": dict(variant="blocks_cov", groups=grp_blocks, group_inj=grp_blocks_inj),
        "blocks_cov_crm": dict(variant="blocks_cov_crm", groups=grp_blocks, group_inj=grp_blocks_inj),
    }
    tirex_results = {}
    for target, mat in (("oil", oil), ("liq", liq)):
        for name, kw in variants.items():
            parts = []
            for cutoff in CUTOFFS_U:
                fc = forecast_tirex(
                    model, mat, cutoff, HORIZON,
                    inj_mat=winj, pres_mat=bhp, inj_future=winj,
                    crm_mat=crm_covs[cutoff], **kw,
                )
                fc["cutoff"] = cutoff
                fc["y_true"] = [mat.at[d, w] for d, w in zip(fc.date, fc.well)]
                parts.append(fc)
            res = pd.concat(parts, ignore_index=True)
            res.to_csv(OUT / f"unisim_tirex_{name}_{target}.csv", index=False)
            tirex_results[(name, target)] = res
            print(f"=== unisim {target} | tirex-{name} ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)

    # --- нефть по Джентилу поверх жидкости (CRM и TiRex-2) ---
    from timesoil.allocation import allocate

    alloc = allocate(winj, distance_weights(coords_unisim()))
    days = pd.Series(alloc.index.days_in_month, index=alloc.index)
    w_cum = alloc.mul(days, axis=0).cumsum()
    for src_name, src in (("crm", crm_res), ("crmp", crmp_res),
                          ("tirex", tirex_results[("blocks_cov_crm", "liq")])):
        rows = []
        for cutoff in CUTOFFS_U:
            part = src[src.cutoff == cutoff]
            for w, g in part.groupby("well"):
                params = fit_gentil(oil[w].loc[:cutoff].dropna(),
                                    liq[w].loc[:cutoff].dropna(),
                                    w_cum.loc[:cutoff, w])
                dates = pd.DatetimeIndex(g.date)
                fo = predict_fo(params, w_cum.loc[dates, w].to_numpy(float))
                for (_, r), f in zip(g.iterrows(), fo):
                    rows.append(dict(cutoff=cutoff, well=w, step=int(r.step), date=r.date,
                                     y_true=float(oil.at[r.date, w]),
                                     y_pred=max(float(r.y_pred) * float(f), 0.0)))
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f"unisim_frac_{src_name}_oil.csv", index=False)
        print(f"=== unisim oil | frac_{src_name} (Джентил) ===")
        print(summarize(res).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
