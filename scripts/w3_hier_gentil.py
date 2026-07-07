"""Волна 3, секция 2: иерархический Джентил (частичный пуллинг).

Эмпирический байес поверх закона Джентила WOR = alpha * W^beta:
1) независимая OLS-подгонка по каждой скважине (те же точки, что в
   timesoil.fractional.fit_gentil: последние 48 валидных месяцев);
2) оценка среднего и дисперсии параметров по фонду (и по блокам) — в
   центрированных координатах (a' = ln WOR при опорном ln W, beta), где
   параметры слабо коррелированы;
3) пере-подгонка каждой скважины с квадратичным штрафом к среднему:
   min ||y - X theta||^2 + lam * sum_j (theta_j - mu_j)^2 / sigma_j^2
   (замкнутая форма); lam=0 — независимая подгонка (база ext_frac_crm),
   lam=inf — полный пуллинг.

Вес штрафа lam и режим пуллинга (фонд/блок) выбираются по 14 срезам без
утечки: leave-one-out по срезам — для каждого среза конфигурация берётся
по минимуму WAPE на остальных 13.

Нефть = CRM-жидкость (results/ext_crm_liq_tpd.csv) x f_o.
База: ext_frac_crm (WAPE(14)=0.0784, WAPE(3)=0.0640).

Запуск: uv run python scripts/w3_hier_gentil.py
Выход: results/ext_frac_hier_oil_tpd.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import CUTOFFS, EXT_CUTOFFS, summarize
from timesoil.data import (
    injection_matrix, load_monthly, producer_matrices,
    static_features, well_coords,
)
from timesoil.fractional import WCT_MAX, fit_gentil
from timesoil.metrics import wape
from timesoil.wells import WELL_BLOCK

OUT = Path(__file__).resolve().parents[1] / "results"

K_LAST = 48
LAM_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 64.0, 256.0,
            1024.0, 4096.0, 16384.0, 65536.0, np.inf]
MODES = ("global", "block")
VAR_FLOOR = 1e-3


def gentil_points(oil: pd.Series, liq: pd.Series, w_cum: pd.Series):
    """Точки (ln W, ln WOR) как в fit_gentil; None, если их < 6 (fallback)."""
    wct = 1.0 - oil / liq.replace(0.0, np.nan)
    ok = (liq > 0) & (wct > 0.01) & (wct < WCT_MAX) & (w_cum > 0)
    wor = (wct[ok] / (1.0 - wct[ok])).tail(K_LAST)
    if len(wor) < 6:
        return None
    x = np.log(w_cum[wor.index].to_numpy(float))
    y = np.log(wor.to_numpy(float))
    return x, y


def fit_penalized(x, y, mu, inv_var, lam):
    """(a', beta) с квадратичным штрафом к mu; x уже центрирован."""
    if np.isinf(lam):
        return float(mu[0]), float(mu[1])
    X = np.column_stack([np.ones_like(x), x])
    D = np.diag(inv_var)
    A = X.T @ X + lam * D
    b = X.T @ y + lam * (D @ mu)
    th = np.linalg.solve(A, b)
    return float(th[0]), float(th[1])


def fo_from_centered(a_c, beta, xbar, ln_w):
    wor = np.exp(a_c + beta * (ln_w - xbar))
    return 1.0 / (1.0 + np.maximum(wor, 0.0))


def main() -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    oil, liq = mats["oil_tpd"], mats["liq_tpd"]
    inj = injection_matrix(df)
    alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
    days = pd.Series(alloc.index.days_in_month, index=alloc.index)
    w_cum = alloc.mul(days, axis=0).cumsum()

    src = pd.read_csv(OUT / "ext_crm_liq_tpd.csv", parse_dates=["date", "cutoff"])

    # --- по срезам: точки, OLS-подгонки, статистики фонда/блоков ---
    per_cut: dict[pd.Timestamp, dict] = {}
    for cutoff in EXT_CUTOFFS:
        part = src[src.cutoff == cutoff]
        wells = sorted(part.well.unique())
        pts, fallback, ols = {}, {}, {}
        xs_all = []
        for w in wells:
            w = int(w)
            p = gentil_points(oil.loc[:cutoff, w], liq.loc[:cutoff, w],
                              w_cum.loc[:cutoff, w])
            if p is None:
                fb = fit_gentil(oil.loc[:cutoff, w], liq.loc[:cutoff, w],
                                w_cum.loc[:cutoff, w])
                assert isinstance(fb, float)
                fallback[w] = fb
            else:
                pts[w] = p
                xs_all.append(p[0])
        xbar = float(np.mean(np.concatenate(xs_all)))
        for w, (x, y) in pts.items():
            xc = x - xbar
            X = np.column_stack([np.ones_like(xc), xc])
            th, *_ = np.linalg.lstsq(X, y, rcond=None)
            ols[w] = th  # (a', beta)
        arr = np.array(list(ols.values()))
        mu_g = arr.mean(axis=0)
        var_g = np.maximum(arr.var(axis=0, ddof=1), VAR_FLOOR)
        mu_b, var_b = {}, {}
        for blk in sorted({WELL_BLOCK[w] for w in ols}):
            sub = np.array([ols[w] for w in ols if WELL_BLOCK[w] == blk])
            if len(sub) >= 3:
                mu_b[blk] = sub.mean(axis=0)
                var_b[blk] = np.maximum(sub.var(axis=0, ddof=1), VAR_FLOOR)
            else:
                mu_b[blk], var_b[blk] = mu_g, var_g
        per_cut[cutoff] = dict(part=part, pts=pts, fallback=fallback,
                               xbar=xbar, mu_g=mu_g, var_g=var_g,
                               mu_b=mu_b, var_b=var_b)

    # --- прогнозы для каждой конфигурации (mode, lam) ---
    preds: dict[tuple, dict[pd.Timestamp, pd.DataFrame]] = {}
    for mode in MODES:
        for lam in LAM_GRID:
            cfg = (mode, lam)
            preds[cfg] = {}
            for cutoff, d in per_cut.items():
                rows = []
                for w, g in d["part"].groupby("well"):
                    w = int(w)
                    dates = pd.DatetimeIndex(g.date)
                    if w in d["fallback"]:
                        fo = np.full(len(dates), d["fallback"][w])
                    else:
                        x, y = d["pts"][w]
                        mu = d["mu_g"] if mode == "global" else d["mu_b"][WELL_BLOCK[w]]
                        var = d["var_g"] if mode == "global" else d["var_b"][WELL_BLOCK[w]]
                        a_c, beta = fit_penalized(x - d["xbar"], y, mu, 1.0 / var, lam)
                        ln_w = np.log(np.maximum(
                            w_cum.loc[dates, w].to_numpy(float), 1e-9))
                        fo = fo_from_centered(a_c, beta, d["xbar"], ln_w)
                    for (_, r), f in zip(g.iterrows(), fo):
                        rows.append(dict(cutoff=cutoff, well=w, step=int(r.step),
                                         date=r.date, y_true=float(oil.at[r.date, w]),
                                         y_pred=max(float(r.y_pred) * float(f), 0.0)))
                preds[cfg][cutoff] = pd.DataFrame(rows)

    # --- сводка по конфигурациям (диагностика) ---
    print("WAPE(14) по конфигурациям:")
    scores = {}
    for cfg, by_cut in preds.items():
        allp = pd.concat(by_cut.values(), ignore_index=True)
        scores[cfg] = wape(allp.y_true, allp.y_pred)
        print(f"  mode={cfg[0]:6s} lam={cfg[1]:>8}: {scores[cfg]:.4f}")
    base14 = scores[("global", 0.0)]
    print(f"проверка базы (lam=0 == ext_frac_crm): WAPE(14)={base14:.4f}")

    # --- leave-one-out по срезам: выбор конфигурации без утечки ---
    abs_err = {cfg: {c: (np.abs(p.y_true - p.y_pred).sum(), np.abs(p.y_true).sum())
                     for c, p in by_cut.items()}
               for cfg, by_cut in preds.items()}
    chosen, final_parts = {}, []
    for c in EXT_CUTOFFS:
        best_cfg, best = None, np.inf
        for cfg in preds:
            num = sum(abs_err[cfg][cc][0] for cc in EXT_CUTOFFS if cc != c)
            den = sum(abs_err[cfg][cc][1] for cc in EXT_CUTOFFS if cc != c)
            if num / den < best:
                best, best_cfg = num / den, cfg
        chosen[c] = best_cfg
        final_parts.append(preds[best_cfg][c])
    print("\nвыбор LOO по срезам (mode, lam):")
    for c, cfg in chosen.items():
        print(f"  {c:%Y-%m}: {cfg[0]}, lam={cfg[1]}")

    final = pd.concat(final_parts, ignore_index=True)
    final.to_csv(OUT / "ext_frac_hier_oil_tpd.csv", index=False)

    print("\n=== ext oil | frac_hier (LOO) ===")
    print(summarize(final).round(4).to_string(index=False))
    w14 = wape(final.y_true, final.y_pred)
    f3 = final[final.cutoff.isin(CUTOFFS)]
    w3 = wape(f3.y_true, f3.y_pred)
    b3 = preds[("global", 0.0)]
    b3 = pd.concat([b3[c] for c in CUTOFFS], ignore_index=True)
    print(f"\nитог: WAPE(14)={w14:.4f} (база {base14:.4f}), "
          f"WAPE(3)={w3:.4f} (база {wape(b3.y_true, b3.y_pred):.4f})")


if __name__ == "__main__":
    main()
