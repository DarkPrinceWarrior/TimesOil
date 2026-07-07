"""Оптимизация распределения закачки на горизонте 2015-12..2016-05.

Постановка: множитель режима каждой из 16 нагнетательных постоянен на
горизонте, суммарная закачка по полю фиксирована на базовом уровне,
границы множителей — сценарии ±20 % и ±30 %. Целевая функция — суммарная
добыча нефти за 6 месяцев по стеку «CRM-жидкость x доля нефти по Джентилу»
(доля нефти зависит от закачки через накопленную адресную закачку — рост
закачки ускоряет обводнение, компромисс учтён). Метод — SLSQP.

Прокси-модель откалибрована на истории; результат подлежит подтверждению
гидродинамической моделью до применения на промысле.

Выходы: results/injection_optimization.csv (множители и режимы),
results/injection_optimization_summary.csv, figures/injection_opt.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import HORIZON
from timesoil.crm import FULL_START, fit_block, predict_block
from timesoil.data import (
    LAST_VALID,
    injection_matrix,
    load_monthly,
    producer_matrices,
    static_features,
    well_coords,
)
from timesoil.fractional import fit_gentil, predict_fo
from timesoil.wells import INJECTORS, PRODUCERS, WELL_BLOCK, block_wells

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
BLOCKS = ("A", "B", "B2", "C", "D", "E")


def main() -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)
    inj_cols = sorted(INJECTORS)

    future_idx = pd.date_range(LAST_VALID, periods=HORIZON + 1, freq="MS")[1:]
    days_future = pd.Series(future_idx.days_in_month, index=future_idx).to_numpy(float)
    base_future = inj.iloc[-1][inj_cols].to_numpy(float)  # м3/сут по нагнетательным

    # --- один раз: CRM по блокам и Джентил по скважинам ---
    print("подгонка CRM по блокам...", flush=True)
    models = {}
    for b in BLOCKS:
        prods = block_wells(b, injectors=False)
        injs = block_wells(b, injectors=True)
        models[b] = (fit_block(liq, inj, prods, injs, LAST_VALID), prods, injs)

    w_alloc = hydro_weights(static_features(), well_coords())
    alloc_hist = allocate(inj, w_alloc)
    days_hist = pd.Series(alloc_hist.index.days_in_month, index=alloc_hist.index)
    w_cum_last = alloc_hist.mul(days_hist, axis=0).cumsum().loc[LAST_VALID]
    gentil = {
        w: fit_gentil(oil[w], liq[w], alloc_hist.mul(days_hist, axis=0).cumsum().loc[:LAST_VALID, w])
        for w in PRODUCERS
    }
    w_alloc_np = w_alloc[inj_cols].to_numpy()  # [33, 16]

    hist_idx = liq.loc[FULL_START:LAST_VALID].index
    full_idx = hist_idx.append(future_idx)
    inj_hist_np = inj.loc[FULL_START:LAST_VALID, inj_cols].to_numpy(float)

    def stack_oil(lam: np.ndarray, detail: bool = False):
        """Суммарная нефть за горизонт, т, при множителях lam по нагнетательным."""
        rates = base_future * lam
        inj_full = pd.DataFrame(
            np.vstack([inj_hist_np, np.tile(rates, (HORIZON, 1))]),
            index=full_idx, columns=inj_cols,
        )
        liq_pred = pd.concat(
            [predict_block(m, inj_full[injs], prods) for m, prods, injs in models.values()],
            axis=1,
        )[list(PRODUCERS)].loc[future_idx]
        # накопленная адресная закачка на горизонте
        alloc_fut = (np.tile(rates, (HORIZON, 1)) @ w_alloc_np.T) * days_future[:, None]
        w_cum_fut = w_cum_last[list(PRODUCERS)].to_numpy(float) + np.cumsum(alloc_fut, axis=0)
        oil_pred = np.empty_like(liq_pred.to_numpy())
        for k, w in enumerate(PRODUCERS):
            fo = predict_fo(gentil[w], w_cum_fut[:, k])
            oil_pred[:, k] = liq_pred.iloc[:, k].to_numpy() * fo
        total = float((oil_pred.sum(axis=1) * days_future).sum())
        if detail:
            return total, liq_pred, pd.DataFrame(oil_pred, index=future_idx, columns=list(PRODUCERS))
        return total

    lam0 = np.ones(len(inj_cols))
    base_oil, base_liq, _ = stack_oil(lam0, detail=True)
    print(f"базовый план: нефть за 6 мес = {base_oil:,.0f} т", flush=True)

    rows_sum, rows_det = [], []
    for bound in (0.2, 0.3):
        cons = {"type": "eq",
                "fun": lambda lam: float(np.dot(lam, base_future) - base_future.sum())}
        res = minimize(
            lambda lam: -stack_oil(lam), lam0, method="SLSQP",
            bounds=[(1 - bound, 1 + bound)] * len(inj_cols), constraints=[cons],
            options={"maxiter": 200, "ftol": 1e-8},
        )
        lam = res.x
        opt_oil, opt_liq, _ = stack_oil(lam, detail=True)
        gain_t = opt_oil - base_oil
        rows_sum.append(dict(
            bound=f"±{int(bound * 100)}%", success=bool(res.success),
            base_oil_t=round(base_oil), opt_oil_t=round(opt_oil),
            gain_t=round(gain_t), gain_pct=round(gain_t / base_oil * 100, 2),
            liq_change_pct=round(
                (opt_liq.sum().sum() - base_liq.sum().sum()) / base_liq.sum().sum() * 100, 2),
        ))
        for i, w in enumerate(inj_cols):
            rows_det.append(dict(
                bound=f"±{int(bound * 100)}%", injector=w, block=WELL_BLOCK[w],
                base_m3d=round(base_future[i], 1), multiplier=round(lam[i], 3),
                opt_m3d=round(base_future[i] * lam[i], 1),
            ))
        print(f"±{int(bound*100)}%: нефть {opt_oil:,.0f} т (+{gain_t:,.0f} т, "
              f"+{gain_t / base_oil * 100:.2f} %), сходимость={res.success}", flush=True)

    summary = pd.DataFrame(rows_sum)
    detail = pd.DataFrame(rows_det)
    summary.to_csv(OUT / "injection_optimization_summary.csv", index=False)
    detail.to_csv(OUT / "injection_optimization.csv", index=False)

    # график: множители по нагнетательным (сценарий ±30 %)
    d30 = detail[detail.bound == "±30%"].sort_values(["block", "injector"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {"A": "tab:red", "B": "tab:blue", "B2": "deepskyblue",
              "C": "tab:orange", "D": "magenta", "E": "tab:green"}
    ax.bar([f"{r.injector}\n{r.block}" for r in d30.itertuples()],
           d30.multiplier, color=[colors[b] for b in d30.block])
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_ylabel("множитель режима")
    ax.set_title("Оптимальное перераспределение закачки (границы ±30 %, сумма фиксирована)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "injection_opt.png", dpi=120)
    print("сохранено: results/injection_optimization*.csv, figures/injection_opt.png")


if __name__ == "__main__":
    main()
