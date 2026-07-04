"""CSV для SPDM (ManiMamba): по срезу бэктеста и целевой величине.

Формат их Dataset_Custom/Dataset_Pred: колонка date + числовые каналы.
Каналы: 33 добывающих (т/сут) + 16 нагнетательных (закачка м3/сут).
Матрица начинается с 2008-07 (весь фонд запущен) — NaN их загрузчик не
переносит. Колонка inj42 последняя: --target inj42 сохраняет порядок
каналов при их внутренней перестановке (target уходит в конец).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from timesoil.backtest import CUTOFFS
from timesoil.data import LAST_VALID, injection_matrix, load_monthly, producer_matrices
from timesoil.wells import INJECTORS, PRODUCERS

FULL_START = pd.Timestamp("2008-07-01")
OUT = Path(__file__).resolve().parents[1] / "data" / "spdm"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)
    ends = list(CUTOFFS) + [LAST_VALID]
    for target, key in (("oil", "oil_tpd"), ("liq", "liq_tpd")):
        prod = mats[key].loc[FULL_START:]
        assert not prod.isna().any().any(), "NaN в матрице добычи после FULL_START"
        wide = pd.concat(
            [
                prod.rename(columns={w: f"w{w}" for w in PRODUCERS}),
                inj.loc[FULL_START:].rename(columns={w: f"inj{w}" for w in sorted(INJECTORS)}),
            ],
            axis=1,
        )
        for end in ends:
            part = wide.loc[:end].reset_index().rename(columns={"index": "date"})
            part["date"] = part["date"].dt.strftime("%Y-%m-%d")
            tag = pd.Timestamp(end).strftime("%Y%m")
            path = OUT / f"{target}_{tag}.csv"
            part.to_csv(path, index=False)
            print(f"{path.name}: {part.shape[0]} мес x {part.shape[1] - 1} каналов, до {end.date()}")


if __name__ == "__main__":
    main()
