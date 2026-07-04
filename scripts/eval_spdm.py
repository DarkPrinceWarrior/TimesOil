"""Оценка прогнозов SPDM: real_prediction.npy -> единый формат результатов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.data import load_monthly, producer_matrices
from timesoil.wells import PRODUCERS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "spdm"


def main() -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    for target, key in (("oil", "oil_tpd"), ("liq", "liq_tpd")):
        mat = mats[key]
        parts = []
        for cutoff in CUTOFFS:
            tag = pd.Timestamp(cutoff).strftime("%Y%m")
            npy = RES / f"{target}_{tag}" / "real_prediction.npy"
            if not npy.exists():
                print(f"нет {npy}, пропуск")
                continue
            pred = np.load(npy).reshape(HORIZON, -1)[:, : len(PRODUCERS)]
            pred = np.maximum(pred, 0.0)
            dates = pd.date_range(cutoff, periods=HORIZON + 1, freq="MS")[1:]
            for step, dt in enumerate(dates, 1):
                for i, w in enumerate(PRODUCERS):
                    parts.append(
                        dict(cutoff=cutoff, well=w, step=step, date=dt,
                             y_true=float(mat.loc[dt, w]), y_pred=float(pred[step - 1, i]))
                    )
        if not parts:
            continue
        res = pd.DataFrame(parts)
        res.to_csv(ROOT / "results" / f"spdm_{key}.csv", index=False)
        print(f"\n=== {key} | SPDM (ManiMamba) ===")
        print(summarize(res).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
