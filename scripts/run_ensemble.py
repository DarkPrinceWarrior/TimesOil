"""Ансамбль моделей на расширенной проверке (14 срезов x 6 мес).

Схема без утечки: для каждого среза веса подбираются на остальных 13
(исключаемый срез — контрольный). Три схемы: среднее двух лучших,
взвешивание по обратной ошибке, неотрицательная линейная регрессия (NNLS).
Итоговые веса для прогноза вперёд подбираются на всех срезах.

Выходы: results/ext_ens_<схема>_<цель>.csv, results/ensemble_weights.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from timesoil import metrics as M
from timesoil.backtest import CUTOFFS, summarize

OUT = Path(__file__).resolve().parents[1] / "results"

COMPONENTS = {
    "oil_tpd": {
        "frac_crm": "ext_frac_crm_oil_tpd.csv",
        "chronos": "ext_chronos_oil_tpd.csv",
        "tirex": "ext_tirex_oil_tpd.csv",
        "lgbm": "ext_lgbm_oil_tpd.csv",
        "tide": "ext_nf_tide_oil_tpd.csv",
    },
    "liq_tpd": {
        "crm": "ext_crm_liq_tpd.csv",
        "tirex": "ext_tirex_liq_tpd.csv",
        "chronos": "ext_chronos_liq_tpd.csv",
        "lgbm": "ext_lgbm_liq_tpd.csv",
    },
}


def load_matrix(files: dict[str, str]) -> pd.DataFrame:
    """Совмещение прогнозов моделей: строки (cutoff, well, date), колонки-модели."""
    base = None
    for name, fname in files.items():
        p = OUT / fname
        if not p.exists():
            print(f"  ! нет {fname} — модель {name} пропущена")
            continue
        df = pd.read_csv(p, parse_dates=["date", "cutoff"])
        df["well"] = df["well"].astype(str)
        cols = df[["cutoff", "well", "date", "y_true", "y_pred"]].rename(
            columns={"y_pred": name})
        if base is None:
            base = cols
        else:
            base = base.merge(cols.drop(columns="y_true"),
                              on=["cutoff", "well", "date"], how="inner")
    return base.dropna().reset_index(drop=True)


def fit_weights(train: pd.DataFrame, models: list[str], scheme: str) -> np.ndarray:
    if scheme == "mean_top2":
        wapes = [M.wape(train.y_true, train[m]) for m in models]
        idx = np.argsort(wapes)[:2]
        w = np.zeros(len(models)); w[idx] = 0.5
        return w
    if scheme == "inv_wape":
        wapes = np.array([M.wape(train.y_true, train[m]) for m in models])
        w = 1.0 / wapes**2
        return w / w.sum()
    if scheme == "nnls":
        A = train[models].to_numpy(float)
        w, _ = nnls(A, train.y_true.to_numpy(float))
        return w / w.sum() if w.sum() > 0 else np.full(len(models), 1 / len(models))
    raise ValueError(scheme)


def main() -> None:
    weight_rows = []
    for target, files in COMPONENTS.items():
        mat = load_matrix(files)
        models = [m for m in files if m in mat.columns]
        cutoffs = sorted(mat.cutoff.unique())
        print(f"\n##### {target}: {len(models)} моделей, {len(cutoffs)} срезов #####")
        for m in models:
            print(f"  {m:10s} WAPE(14 срезов) = {M.wape(mat.y_true, mat[m]):.4f}")

        for scheme in ("mean_top2", "inv_wape", "nnls"):
            parts = []
            for c in cutoffs:
                train, test = mat[mat.cutoff != c], mat[mat.cutoff == c].copy()
                w = fit_weights(train, models, scheme)
                test["y_pred"] = np.maximum(test[models].to_numpy(float) @ w, 0.0)
                parts.append(test)
            ens = pd.concat(parts, ignore_index=True)
            ens[["cutoff", "well", "date", "y_true", "y_pred"]].assign(
                step=ens.groupby(["cutoff", "well"]).cumcount() + 1
            ).to_csv(OUT / f"ext_ens_{scheme}_{target}.csv", index=False)
            wape_all = M.wape(ens.y_true, ens.y_pred)
            canon = ens[ens.cutoff.isin(CUTOFFS)]
            if scheme == "nnls":  # канонические 3 среза — для общей сводки
                canon[["cutoff", "well", "date", "y_true", "y_pred"]].assign(
                    step=canon.groupby(["cutoff", "well"]).cumcount() + 1
                ).to_csv(OUT / f"ens_nnls_{target}.csv", index=False)
            wape_canon = M.wape(canon.y_true, canon.y_pred)
            bias = M.cum_error_pct(ens.y_true.to_numpy(), ens.y_pred.to_numpy())
            print(f"  ансамбль {scheme:10s}: WAPE(14)={wape_all:.4f}  "
                  f"WAPE(3 канонич.)={wape_canon:.4f}  смещение={bias:+.2f}%")

        # итоговые веса на всех срезах — для прогноза вперёд
        w_final = fit_weights(mat, models, "nnls")
        for m, wv in zip(models, w_final):
            weight_rows.append(dict(target=target, model=m, weight=round(float(wv), 4)))
    wdf = pd.DataFrame(weight_rows)
    wdf.to_csv(OUT / "ensemble_weights.csv", index=False)
    print("\nИтоговые веса (NNLS, все срезы):")
    print(wdf.to_string(index=False))


if __name__ == "__main__":
    main()
