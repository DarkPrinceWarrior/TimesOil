"""Финальный прогноз вперёд: 2015-12..2016-05, TiRex-2 blocks_cov.

Закачка на горизонте неизвестна -> продлеваем последний наблюдённый режим
(закачка в данных кусочно-постоянная, это штатный план ППД).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from timesoil.backtest import HORIZON
from timesoil.data import LAST_VALID, injection_matrix, load_monthly, producer_matrices
from timesoil.tirex_runner import forecast_tirex

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)

    future_idx = pd.date_range(LAST_VALID, periods=HORIZON + 1, freq="MS")[1:]
    inj_future = pd.concat([inj, pd.DataFrame(
        [inj.iloc[-1]] * HORIZON, index=future_idx, columns=inj.columns)])

    OUT.mkdir(exist_ok=True)
    for target in ("oil_tpd", "liq_tpd"):
        fc = forecast_tirex(
            model, mats[target], LAST_VALID, HORIZON, "blocks_cov",
            inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj_future,
        )
        fc.to_csv(OUT / f"forward_{target}.csv", index=False)
        field = fc.groupby("date")[["q10", "y_pred", "q90"]].sum()
        print(f"\n=== Прогноз по полю, {target} (т/сут) ===")
        print(field.round(1).to_string())


if __name__ == "__main__":
    main()
