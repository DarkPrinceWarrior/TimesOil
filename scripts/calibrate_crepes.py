"""Строгая конформная калибровка интервалов ансамбля пакетом crepes.

Данные: results/ext_ens_nnls_{oil,liq}_tpd.csv — точечные прогнозы ансамбля
на 14 срезах (2013-03..2015-05, шаг 2 мес) x 6 мес. Интервалы 80 % строятся
из остатков y_true - y_pred через crepes.ConformalRegressor:

  split       — простой сплит-конформал на абсолютных остатках;
  mond_step   — мондрианова таксономия «шаг горизонта» (6 категорий);
  mond_sxb    — «шаг x блок» (блоки с < MIN_BIN калибровочных точек на
                категорию объединяются в «other»);
  norm_pred   — нормализованные интервалы, трудность = величина прогноза
                (sigma = y_pred + beta);
  norm_x_step — нормализация по величине прогноза + мондриан по шагу.

Протокол без утечки — как в scripts/calibrate_intervals.py: для каждого из
3 канонических срезов калибровка только на срезах, чьи 6-месячные окна не
пересекаются с его окном (|месяцев между срезами| >= HORIZON).

Дополнительно: тест обмениваемости остатков по срезам (crepes.martingales,
SimpleJumper на полуонлайновых p-значениях).

Запуск: uv run --with crepes python scripts/calibrate_crepes.py
Выход: печать таблицы «метод — накрытие — ширина»; лучший метод
сохраняется в results/crepes_intervals_<цель>.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from crepes import ConformalRegressor

from timesoil.wells import WELL_BLOCK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

TARGETS = ("oil_tpd", "liq_tpd")
CONFIDENCE = 0.80
HORIZON = 6
CANON = (
    pd.Timestamp("2014-05-01"),
    pd.Timestamp("2014-11-01"),
    pd.Timestamp("2015-05-01"),
)
MIN_BIN = 30  # минимум калибровочных точек на мондрианову категорию
COV_TOL = 0.02  # допуск недобора накрытия при выборе лучшего метода


def months_apart(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return abs((a.year - b.year) * 12 + a.month - b.month)


def load(target: str) -> pd.DataFrame:
    df = pd.read_csv(
        OUT / f"ext_ens_nnls_{target}.csv", parse_dates=["cutoff", "date"]
    )
    df["well"] = df["well"].astype(int)
    df["block"] = df["well"].map(WELL_BLOCK)
    if df["block"].isna().any():
        raise ValueError("скважины без блока: "
                         f"{sorted(df.loc[df['block'].isna(), 'well'].unique())}")
    df["resid"] = df["y_true"] - df["y_pred"]
    return df


def step_block_bins(calib: pd.DataFrame, ev: pd.DataFrame,
                    verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Категории «шаг x блок»; блоки с < MIN_BIN точек на категорию -> other."""
    cnt = calib.groupby(["step", "block"]).size()
    ok = {b for b in calib["block"].unique()
          if cnt.xs(b, level="block").min() >= MIN_BIN}
    merged = sorted(set(calib["block"].unique()) - ok)
    if verbose and merged:
        pooled = int(cnt[cnt.index.get_level_values("block").isin(merged)]
                     .groupby("step").sum().min())
        print(f"    mond_sxb: блоки {merged} объединены в 'other' "
              f"(min точек/категорию до объединения "
              f"{int(cnt[cnt.index.get_level_values('block').isin(merged)].min())}, "
              f"после {pooled})")

    def cat(fr: pd.DataFrame) -> np.ndarray:
        blk = fr["block"].where(fr["block"].isin(ok), "other")
        return (fr["step"].astype(str) + "|" + blk).to_numpy()

    return cat(calib), cat(ev)


def sigmas_pred(calib: pd.DataFrame, ev: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray]:
    """Трудность = величина прогноза; beta страхует нулевые сигмы."""
    beta = 0.05 * float(np.abs(calib["y_pred"]).mean())
    return (np.abs(calib["y_pred"]).to_numpy() + beta,
            np.abs(ev["y_pred"]).to_numpy() + beta)


def predict_method(method: str, calib: pd.DataFrame, ev: pd.DataFrame,
                   verbose: bool = False) -> np.ndarray:
    """Интервалы [lo, hi] для eval-среза, калибровка на calib."""
    cr = ConformalRegressor()
    res = calib["resid"].to_numpy()
    y_hat = ev["y_pred"].to_numpy()
    kw: dict = {}
    if method == "split":
        cr.fit(res)
    elif method == "mond_step":
        cr.fit(res, bins=calib["step"].to_numpy())
        kw["bins"] = ev["step"].to_numpy()
    elif method == "mond_sxb":
        b_cal, b_ev = step_block_bins(calib, ev, verbose=verbose)
        cr.fit(res, bins=b_cal)
        kw["bins"] = b_ev
    elif method == "norm_pred":
        s_cal, s_ev = sigmas_pred(calib, ev)
        cr.fit(res, sigmas=s_cal)
        kw["sigmas"] = s_ev
    elif method == "norm_x_step":
        s_cal, s_ev = sigmas_pred(calib, ev)
        cr.fit(res, sigmas=s_cal, bins=calib["step"].to_numpy())
        kw["sigmas"] = s_ev
        kw["bins"] = ev["step"].to_numpy()
    else:
        raise ValueError(method)
    # дебиты неотрицательны — нижнюю границу режем нулём (накрытие не меняет)
    return cr.predict_int(y_hat, confidence=CONFIDENCE, y_min=0.0, **kw)


METHODS = ("split", "mond_step", "mond_sxb", "norm_pred", "norm_x_step")


def exchangeability_test(df: pd.DataFrame, target: str) -> None:
    """Обмениваемость остатков между срезами: мартингейл на p-значениях.

    Внутри среза порядок случайный (fix seed) — иначе детерминированный
    порядок «скважина x шаг» тривиально ломает обмениваемость и маскирует
    вопрос о дрейфе между срезами. Срезы идут хронологически.
    """
    try:
        from crepes.martingales import SimpleJumper, semi_online_p_values
    except ImportError:
        print(f"  {target}: crepes.martingales недоступен — тест пропущен")
        return
    rng = np.random.default_rng(0)
    parts = [g.iloc[rng.permutation(len(g))]
             for _, g in df.sort_values("cutoff").groupby("cutoff", sort=True)]
    alphas = np.abs(pd.concat(parts)["resid"].to_numpy())
    p = semi_online_p_values(alphas, seed=0)
    m = SimpleJumper().apply(p)
    log_max = float(np.log10(np.max(m)))
    verdict = ("обмениваемость ОТВЕРГАЕТСЯ (M_max > 100 <=> p < 0.01)"
               if log_max > 2 else
               "слабое свидетельство против" if log_max > 1 else
               "не отвергается")
    print(f"  {target}: SimpleJumper log10(max M) = {log_max:.1f} "
          f"(final {float(np.log10(m[-1])):.1f}) -> {verdict}")


def main() -> None:
    best_rows: dict[str, str] = {}
    for target in TARGETS:
        df = load(target)
        print(f"\n=== Цель: {target} (строк {len(df)}, "
              f"срезов {df['cutoff'].nunique()}) ===")

        # накрытие/ширина по 3 каноническим срезам, без утечки
        table: list[dict] = []
        saved: dict[str, list[pd.DataFrame]] = {m: [] for m in METHODS}
        for method in METHODS:
            covs, widths = [], []
            for fold in CANON:
                calib = df[df["cutoff"].map(
                    lambda c: months_apart(c, fold) >= HORIZON)]
                ev = df[df["cutoff"] == fold].reset_index(drop=True)
                ints = predict_method(method, calib, ev,
                                      verbose=(fold == CANON[0]))
                lo, hi = ints[:, 0], ints[:, 1]
                cov = float(((ev["y_true"] >= lo) & (ev["y_true"] <= hi)).mean())
                width = float((hi - lo).mean() / ev["y_true"].mean())
                covs.append(cov)
                widths.append(width)
                out = ev[["cutoff", "well", "step", "date", "y_pred"]].copy()
                out["q10"], out["q90"] = lo, hi
                saved[method].append(out)
            table.append(dict(
                method=method,
                **{f"cov_{f:%Y-%m}": c for f, c in zip(CANON, covs)},
                cov_mean=float(np.mean(covs)),
                width_mean=float(np.mean(widths)),
            ))

        rep = pd.DataFrame(table)
        print(rep.round(3).to_string(index=False))
        print(f"  (ширина — mean(hi-lo)/mean(y_true) по срезу; цель накрытия "
              f"{CONFIDENCE}; самописная мультипликативная калибровка TiRex-2 "
              f"даёт ~0.73)")

        # лучший метод: накрытие не хуже цели минус допуск,
        # затем min ширина; при равной ширине — накрытие ближе к цели
        rep["_gap"] = (rep["cov_mean"] - CONFIDENCE).abs()
        ok = rep[rep["cov_mean"] >= CONFIDENCE - COV_TOL]
        best = (ok.sort_values(["width_mean", "_gap"]).iloc[0] if len(ok)
                else rep.sort_values("cov_mean", ascending=False).iloc[0])
        best_rows[target] = str(best["method"])
        print(f"  лучший метод: {best['method']} "
              f"(накрытие {best['cov_mean']:.3f}, "
              f"ширина {best['width_mean']:.3f})")

        out_df = pd.concat(saved[best["method"]], ignore_index=True)
        out_df["cutoff"] = out_df["cutoff"].dt.date
        out_df["date"] = out_df["date"].dt.date
        path = OUT / f"crepes_intervals_{target}.csv"
        out_df.to_csv(path, index=False)
        print(f"  сохранено: {path.relative_to(ROOT)} ({len(out_df)} строк)")

    print("\n=== Тест применимости (обмениваемость остатков) ===")
    for target in TARGETS:
        exchangeability_test(load(target), target)

    print("\nИтог:", ", ".join(f"{t} -> {m}" for t, m in best_rows.items()))


if __name__ == "__main__":
    main()
