"""Бейслайны (наивный / экспонента / Арпс) для нефти и жидкости, т/сут."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from timesoil import baselines as B
from timesoil.backtest import run_pointwise, summarize
from timesoil.data import load_monthly, producer_matrices

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    models = {
        "naive": B.forecast_naive,
        "exp24": partial(B.forecast_exp, k=24),
        "arps36": partial(B.forecast_arps, k=36),
    }
    for target in ("oil_tpd", "liq_tpd"):
        mat = mats[target]
        for name, fc in models.items():
            res = run_pointwise(mat, fc)
            res.to_csv(OUT / f"baseline_{name}_{target}.csv", index=False)
            summ = summarize(res)
            print(f"\n=== {target} | {name} ===")
            print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
