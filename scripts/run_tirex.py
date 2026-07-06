"""Бэктест TiRex-2: варианты u / m / blocks / blocks_cov, нефть и жидкость.

Запуск:  uv run python scripts/run_tirex.py [--tta-diff-off]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.data import injection_matrix, load_monthly, producer_matrices
from timesoil.tirex_runner import forecast_tirex

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta-diff-off", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["u", "m", "blocks", "blocks_cov"])
    args = ap.parse_args()

    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    kw = {"tta_diff": False} if args.tta_diff_off else {}
    suffix = "_nodiff" if args.tta_diff_off else ""

    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)

    crm_covs = {}
    for cutoff in CUTOFFS:
        p = OUT / f"crm_cov_{cutoff:%Y%m}.csv"
        if p.exists():
            crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)

    # адресная закачка: веса гидропроводности (статика) и связности CRM (по срезам)
    alloc_maps: dict[str, dict] = {"blocks_wcov": {}, "blocks_fcov": {}, "blocks_wcov_crm": {}}
    if "blocks_wcov" in args.variants or "blocks_wcov_crm" in args.variants:
        from timesoil.allocation import allocate, hydro_weights
        from timesoil.data import static_features, well_coords

        w_h = hydro_weights(static_features(), well_coords())
        alloc_h = allocate(inj, w_h)
        alloc_maps["blocks_wcov"] = {c: alloc_h for c in CUTOFFS}
        alloc_maps["blocks_wcov_crm"] = {c: alloc_h for c in CUTOFFS}
    if "blocks_fcov" in args.variants:
        from timesoil.allocation import allocate

        for cutoff in CUTOFFS:
            p = OUT / f"crm_gains_{cutoff:%Y%m}.csv"
            if p.exists():
                g = pd.read_csv(p, index_col=0).rename(columns=int)
                alloc_maps["blocks_fcov"][cutoff] = allocate(inj, g)

    for target in ("oil_tpd", "liq_tpd"):
        mat = mats[target]
        for variant in args.variants:
            parts = []
            for cutoff in CUTOFFS:
                fc = forecast_tirex(
                    model, mat, cutoff, HORIZON, variant,
                    inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj,
                    crm_mat=crm_covs.get(cutoff),
                    alloc_mat=alloc_maps.get(variant, {}).get(cutoff), **kw,
                )
                fc["cutoff"] = cutoff
                truth = mat.loc[fc.date.unique()]
                fc["y_true"] = [truth.loc[d, w] for d, w in zip(fc.date, fc.well)]
                parts.append(fc)
            res = pd.concat(parts, ignore_index=True)
            res.to_csv(OUT / f"tirex_{variant}{suffix}_{target}.csv", index=False)
            print(f"\n=== {target} | tirex-{variant}{suffix} ===")
            print(summarize(res).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
