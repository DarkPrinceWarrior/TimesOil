"""CRM2P: бэктест двухфазной CRM — одновременно жидкость и нефть.

Выходы (формат общий: cutoff, well, step, date, y_true, y_pred):
- results/crm2p_liq_tpd.csv, results/crm2p_oil_tpd.csv — 3 канонических среза;
- с флагом --ext — 14 срезов, файлы с префиксом ext_.

Сравнение печатается против баз: жидкость — CRM по блокам
(crm_liq_tpd.csv), нефть — CRM x Джентил (frac_crm_oil_tpd.csv).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from timesoil.backtest import CUTOFFS, EXT_CUTOFFS, HORIZON, summarize
from timesoil.crm2p import crm2p_forecast
from timesoil.data import injection_matrix, load_monthly, producer_matrices

OUT = Path(__file__).resolve().parents[1] / "results"
BASELINES = {"liq_tpd": "crm_liq_tpd.csv", "oil_tpd": "frac_crm_oil_tpd.csv"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext", action="store_true", help="14 срезов вместо 3")
    args = ap.parse_args()
    cutoffs = EXT_CUTOFFS if args.ext else CUTOFFS
    prefix = "ext_" if args.ext else ""

    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)

    rows_liq, rows_oil = [], []
    for cutoff in cutoffs:
        t0 = time.perf_counter()
        liq_p, oil_p, info = crm2p_forecast(liq, oil, inj, cutoff, HORIZON)
        for rows, pred, truth in ((rows_liq, liq_p, liq), (rows_oil, oil_p, oil)):
            for step, dt in enumerate(pred.index, 1):
                for w in pred.columns:
                    rows.append(
                        dict(cutoff=cutoff, well=w, step=step, date=dt,
                             y_true=float(truth.at[dt, w]), y_pred=float(pred.at[dt, w]))
                    )
        print(
            f"{cutoff:%Y-%m}: mu_o/mu_w = {info['mu_ratio']:.2f}, "
            f"{time.perf_counter() - t0:.1f} c"
        )

    for name, rows in (("liq_tpd", rows_liq), ("oil_tpd", rows_oil)):
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f"{prefix}crm2p_{name}.csv", index=False)
        print(f"\n=== {name} | CRM2P (двухфазная, по блокам) ===")
        print(summarize(res).round(4).to_string(index=False))
        base_path = OUT / BASELINES[name]
        if not args.ext and base_path.exists():
            base = pd.read_csv(base_path, parse_dates=["date", "cutoff"])
            print(f"--- база: {BASELINES[name]} ---")
            print(summarize(base).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
