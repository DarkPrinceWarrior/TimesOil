"""Конформная калибровка интервалов TiRex-2 (вариант blocks_cov_crm).

Схема: 14 срезов (2013-03..2015-05, шаг 2 мес) x 6 мес; для каждого из трёх
канонических срезов множители подбираются по остальным срезам, чьи
контрольные окна не пересекаются с его окном (без утечки). Множитель на
горизонт h — эмпирический квантиль уровня 0.8 относительного выхода факта
за половины интервала:

    r = (y - med) / (q90 - med)  при y >= med,
    r = (med - y) / (med - q10)  при y <  med;
    lambda_h = квантиль_{0.8}(r_h);  интервал масштабируется в lambda_h раз.

Выходы: results/calibration.csv (накрытие до/после по срезам),
results/interval_scale.csv (итоговые множители по всем срезам — для
прогноза вперёд).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON
from timesoil.crm import crm_forecast
from timesoil.data import injection_matrix, load_monthly, producer_matrices
from timesoil.tirex_runner import forecast_tirex

OUT = Path(__file__).resolve().parents[1] / "results"
CAL_CUTOFFS = pd.date_range("2013-03-01", "2015-05-01", freq="2MS")
VARIANT = "blocks_cov_crm"
TARGET_COVERAGE = 0.8
EPS = 1e-9


def scale_ratios(fc: pd.DataFrame) -> pd.Series:
    med, lo, hi = fc["y_pred"], fc["y_pred"] - fc["q10"], fc["q90"] - fc["y_pred"]
    up = fc["y_true"] >= med
    r = np.where(
        up,
        (fc["y_true"] - med) / np.maximum(hi, EPS),
        (med - fc["y_true"]) / np.maximum(lo, EPS),
    )
    return pd.Series(np.minimum(r, 1e6), index=fc.index)


def months_apart(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return abs((a.year - b.year) * 12 + a.month - b.month)


def main() -> None:
    from tirex2 import load_model

    model = load_model("NX-AI/TiRex-2", device="cpu")
    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)
    liq = mats["liq_tpd"]

    runs: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {"oil_tpd": {}, "liq_tpd": {}}
    for cutoff in CAL_CUTOFFS:
        crm_mat, _ = crm_forecast(liq, inj, cutoff, HORIZON)
        for target in runs:
            fc = forecast_tirex(
                model, mats[target], cutoff, HORIZON, VARIANT,
                inj_mat=inj, pres_mat=mats["p_res"], inj_future=inj, crm_mat=crm_mat,
            )
            truth = mats[target]
            fc["y_true"] = [truth.at[d, w] for d, w in zip(fc.date, fc.well)]
            fc["r"] = scale_ratios(fc)
            runs[target][cutoff] = fc
        print(f"срез {cutoff:%Y-%m}: готов", flush=True)

    rows, scales = [], []
    for target, per_cut in runs.items():
        # оценка без утечки на канонических срезах
        for fold in CUTOFFS:
            calib = pd.concat(
                [f for c, f in per_cut.items() if months_apart(c, fold) >= HORIZON],
                ignore_index=True,
            )
            lam = calib.groupby("step")["r"].quantile(TARGET_COVERAGE, interpolation="higher")
            ev = per_cut[fold]
            cov_before = float((ev["r"] <= 1.0).mean())
            cov_after = float((ev["r"] <= ev["step"].map(lam)).mean())
            rows.append(dict(target=target, fold=str(fold.date()),
                             n_calib=len(calib), coverage_before=round(cov_before, 4),
                             coverage_after=round(cov_after, 4)))
        # итоговые множители по всем срезам — для прогноза вперёд
        allr = pd.concat(per_cut.values(), ignore_index=True)
        lam_all = allr.groupby("step")["r"].quantile(TARGET_COVERAGE, interpolation="higher")
        for h, v in lam_all.items():
            scales.append(dict(target=target, step=int(h), scale=float(v)))

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(scales).to_csv(OUT / "interval_scale.csv", index=False)
    print(rep.to_string(index=False))
    print("\nМножители (по всем срезам):")
    print(pd.DataFrame(scales).pivot(index="step", columns="target", values="scale").round(3).to_string())


if __name__ == "__main__":
    main()
