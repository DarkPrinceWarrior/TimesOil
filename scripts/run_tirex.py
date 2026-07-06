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

    for target in ("oil_tpd", "liq_tpd"):
        mat = mats[target]
        for variant in args.variants:
            parts = []
            for cutoff in CUTOFFS:
                fc = forecast_tirex(
                    model, mat, cutoff, HORIZON, variant,
                    inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj,
                    crm_mat=crm_covs.get(cutoff), **kw,
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
