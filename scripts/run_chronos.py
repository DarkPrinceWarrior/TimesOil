"""Chronos-2 (amazon/chronos-2, zero-shot) на трёх полигонах: 3 среза x 6 мес.

Варианты: u (по скважине), m (совместно), cov (закачка future + давления
past), cov_crm (+ ряд CRM). Запуск: uv run python scripts/run_chronos.py
[--polygons field unisim volve]. Выходы: results/[<полигон>_]chronos_*.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.chronos_runner import forecast_chronos

OUT = Path(__file__).resolve().parents[1] / "results"
VARIANTS = ("u", "m", "cov", "cov_crm")


def run_polygon(pipeline, name: str, targets: dict, inj, pres, groups, group_inj,
                cutoffs, crm_covs: dict, prefix: str, drop_na: bool = False) -> None:
    for tname, mat in targets.items():
        for variant in VARIANTS:
            parts = []
            for cutoff in cutoffs:
                fc = forecast_chronos(
                    pipeline, mat, cutoff, HORIZON, variant,
                    groups=groups, group_inj=group_inj,
                    inj_mat=inj, pres_mat=pres, inj_future=inj,
                    crm_mat=crm_covs.get(cutoff),
                )
                fc["cutoff"] = cutoff
                fc["y_true"] = [mat.at[d, w] for d, w in zip(fc.date, fc.well)]
                parts.append(fc)
            res = pd.concat(parts, ignore_index=True)
            if drop_na:
                res = res.dropna(subset=["y_true"]).reset_index(drop=True)
            res.to_csv(OUT / f"{prefix}chronos_{variant}_{tname}.csv", index=False)
            print(f"=== {name} {tname} | chronos-{variant} ===")
            print(summarize(res).round(4).tail(1).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygons", nargs="*", default=["field", "unisim", "volve"])
    args = ap.parse_args()

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    OUT.mkdir(exist_ok=True)

    if "field" in args.polygons:
        from timesoil.data import injection_matrix, load_monthly, producer_matrices
        from timesoil.wells import PRODUCERS, block_wells

        df = load_monthly()
        mats = producer_matrices(df)
        inj = injection_matrix(df)
        blocks = ("A", "B", "B2", "C", "D", "E")
        groups = [block_wells(b, injectors=False) for b in blocks]
        group_inj = [block_wells(b, injectors=True) for b in blocks]
        crm_covs = {}
        for cutoff in CUTOFFS:
            p = OUT / f"crm_cov_{cutoff:%Y%m}.csv"
            if p.exists():
                crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)
        run_polygon(pipeline, "field",
                    {"oil_tpd": mats["oil_tpd"], "liq_tpd": mats["liq_tpd"]},
                    inj, mats["p_res"], groups, group_inj, CUTOFFS, crm_covs, "")

    if "unisim" in args.polygons:
        from timesoil.unisim import BLOCKS_U, crm_stack, load_unisim

        m = load_unisim()
        cutoffs = (pd.Timestamp("2022-11-30"), pd.Timestamp("2023-05-31"),
                   pd.Timestamp("2023-11-30"))
        crm_covs = {c: crm_stack(m["liq"], m["winj"], c, HORIZON) for c in cutoffs}
        run_polygon(pipeline, "unisim", {"oil": m["oil"], "liq": m["liq"]},
                    m["winj"], m["bhp"],
                    [b["producers"] for b in BLOCKS_U.values()],
                    [b["injectors"] for b in BLOCKS_U.values()],
                    cutoffs, crm_covs, "unisim_")

    if "volve" in args.polygons:
        from timesoil.volve import INJECTORS_V, PRODUCERS_V, crm_stack, load_volve

        m = load_volve()
        cutoffs = (pd.Timestamp("2015-03-01"), pd.Timestamp("2015-09-01"),
                   pd.Timestamp("2016-03-01"))
        crm_covs = {
            c: crm_stack(m["liq"], m["winj"], c, HORIZON).reindex(columns=list(PRODUCERS_V))
            for c in cutoffs
        }
        run_polygon(pipeline, "volve", {"oil": m["oil"], "liq": m["liq"]},
                    m["winj"], None, [list(PRODUCERS_V)], [list(INJECTORS_V)],
                    cutoffs, crm_covs, "volve_", drop_na=True)


if __name__ == "__main__":
    main()
