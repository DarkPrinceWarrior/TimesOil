"""PDF-приложение к отчёту: все скважины, ансамбль, важности признаков.

Содержание: титул со сводкой; сводная диаграмма моделей; карта блоков;
веса ансамбля и важности признаков LightGBM; по странице на каждую из 33
добывающих скважин (нефть и жидкость: факт, прогнозы ансамбля на трёх
канонических срезах, прогноз вперёд с 80-процентным интервалом,
обводнённость).

Выход: docs/отчёт_приложение_скважины.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from timesoil.backtest import CUTOFFS
from timesoil.data import load_monthly, producer_matrices
from timesoil.wells import PRODUCERS, WELL_BLOCK

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
OUT_PDF = ROOT / "docs" / "отчёт_приложение_скважины.pdf"

NAMES = {"oil_tpd": "нефть", "liq_tpd": "жидкость"}


def page_title(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    txt = (
        "ПРОГНОЗ ДЕБИТА НЕФТИ И ЖИДКОСТИ НА 6 МЕСЯЦЕВ\n"
        "Приложение: поскважинные графики\n\n"
        "Итоговая модель — ансамбль (неотрицательная регрессия, веса по 14 срезам):\n"
        "нефть:  CRM×Джентил + Chronos-2 + TiRex-2 + LightGBM + TiDE\n"
        "жидкость:  CRM + TiRex-2 + Chronos-2 + LightGBM\n\n"
        "Качество на трёх канонических срезах (2014-05, 2014-11, 2015-05; WAPE):\n"
        "нефть 5.10 %   |   жидкость 3.87 %\n"
        "(лучшие одиночные модели: 6.40 % и 4.38 %)\n\n"
        "Интервалы 80 % — эмпирические, по остаткам ансамбля на 14 срезах.\n"
        "Прогноз вперёд: 2015-12 … 2016-05 при продлении последнего режима закачки."
    )
    ax.text(0.5, 0.6, txt, ha="center", va="center", fontsize=13, family="DejaVu Sans")
    fig.suptitle("TimesOil — 2026-07-07", y=0.95, fontsize=10, color="gray")
    pdf.savefig(fig); plt.close(fig)


def page_summary(pdf: PdfPages) -> None:
    s = pd.read_csv(RES / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
    for ax, target in zip(axes, ("oil_tpd", "liq_tpd")):
        g = s[s.target == target].sort_values("wape")

        def color(m: str) -> str:
            if m.startswith("ens"):
                return "tab:purple"
            if m.startswith(("crm", "frac")):
                return "tab:blue"
            if "tirex" in m or m.startswith(("chronos", "nf_")):
                return "tab:green"
            return "tab:red" if m == "spdm" else "tab:gray"

        ax.barh(g.model, g.wape * 100, color=[color(m) for m in g.model])
        ax.set_title(f"{NAMES[target].capitalize()}: WAPE за 3 среза x 6 мес, %")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        for y, v in enumerate(g.wape * 100):
            ax.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_map(pdf: PdfPages) -> None:
    img = plt.imread(FIGS / "blocks_verify.png")
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.imshow(img); ax.axis("off")
    ax.set_title("Блоки разломов (оцифровка карты): цвет — блок, квадрат — нагнетательная")
    pdf.savefig(fig); plt.close(fig)


def page_weights_importance(pdf: PdfPages) -> None:
    weights = pd.read_csv(RES / "ensemble_weights.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    for ax, target in zip(axes[0], ("oil_tpd", "liq_tpd")):
        g = weights[weights.target == target]
        ax.bar(g.model, g.weight, color="tab:purple")
        ax.set_title(f"Веса ансамбля — {NAMES[target]}")
        ax.grid(axis="y", alpha=0.3)

    # важности признаков LightGBM (обучение на всей истории)
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    from timesoil.mlprep import field_dataset
    frames, static_df, _ = field_dataset(RES, CUTOFFS)
    from run_lgbm import LGB_PARAMS
    for ax, target in zip(axes[1], ("oil_tpd", "liq_tpd")):
        df_long = frames[target].merge(static_df, on="unique_id", how="left")
        mlf = MLForecast(models={"lgbm": lgb.LGBMRegressor(**LGB_PARAMS)}, freq="MS",
                         lags=[1, 2, 3, 4, 5, 6, 12],
                         lag_transforms={1: [RollingMean(3), RollingMean(6)]})
        mlf.fit(df_long, static_features=list(static_df.columns.drop("unique_id")))
        booster = mlf.models_["lgbm"].booster_
        imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        imp = imp.sort_values(ascending=True).tail(12)
        ax.barh(imp.index, imp.values / imp.sum(), color="tab:orange")
        ax.set_title(f"Важности признаков LightGBM (доля вклада) — {NAMES[target]}")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def well_pages(pdf: PdfPages) -> None:
    df = load_monthly()
    mats = producer_matrices(df)
    wct = 1.0 - (mats["oil_tpd"] / mats["liq_tpd"].replace(0, np.nan))
    bt = {t: pd.read_csv(RES / f"ens_nnls_{t}.csv", parse_dates=["date", "cutoff"])
          for t in ("oil_tpd", "liq_tpd")}
    fwd = {t: pd.read_csv(RES / f"forward_{t}.csv", parse_dates=["date"])
           for t in ("oil_tpd", "liq_tpd")}
    comp = {t: pd.read_csv(RES / f"forward_components_{t}.csv", parse_dates=["date"])
            for t in ("oil_tpd", "liq_tpd")}

    for w in PRODUCERS:
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), sharex=True)
        for ax, target in zip(axes, ("oil_tpd", "liq_tpd")):
            hist = mats[target][w].dropna().loc["2011-01-01":]
            ax.plot(hist.index, hist.values, "k-", lw=1.6, label="факт")
            # прогнозы ансамбля на канонических срезах
            b = bt[target]
            for i, c in enumerate(CUTOFFS):
                g = b[(b.well.astype(str) == str(w)) & (b.cutoff == c)].sort_values("date")
                ax.plot(g.date, g.y_pred, "-o", ms=3, lw=1.2, color="tab:purple",
                        alpha=0.85, label="ансамбль (срезы)" if i == 0 else None)
            # вперёд
            f = fwd[target][fwd[target].well.astype(str) == str(w)].sort_values("date")
            ax.fill_between(f.date, f.q10, f.q90, color="tab:purple", alpha=0.18,
                            label="80 % интервал")
            ax.plot(f.date, f.y_pred, "--s", ms=4, lw=1.6, color="tab:purple",
                    label="ансамбль (вперёд)")
            cm = comp[target]
            for mname, col in (("frac_crm", "tab:blue"), ("crm", "tab:blue"),
                               ("chronos", "tab:green")):
                gm = cm[(cm.model == mname) & (cm.well.astype(str) == str(w))].sort_values("date")
                if len(gm):
                    ax.plot(gm.date, gm.y_pred, ":", lw=1.2, color=col,
                            label={"frac_crm": "CRM×Джентил", "crm": "CRM",
                                   "chronos": "Chronos-2"}[mname])
            for c in CUTOFFS:
                ax.axvline(c, color="gray", ls=":", lw=0.7)
            ax.axvline(pd.Timestamp("2015-11-01"), color="k", ls=":", lw=1)
            ax.set_ylabel(f"{NAMES[target]}, т/сут")
            ax.grid(alpha=0.3)
            if target == "liq_tpd":
                ax2 = ax.twinx()
                wc = wct[w].dropna().loc["2011-01-01":]
                ax2.plot(wc.index, wc.values, color="tab:brown", lw=1, alpha=0.5)
                ax2.set_ylabel("обводнённость, доли", color="tab:brown")
                ax2.set_ylim(0, 1.05)
        axes[0].legend(fontsize=8, ncol=3, loc="upper right")
        axes[0].set_title(
            f"Скважина {w} (блок {WELL_BLOCK[w]}): факт, ансамбль на срезах, прогноз вперёд")
        fig.autofmt_xdate(); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    with PdfPages(OUT_PDF) as pdf:
        page_title(pdf)
        page_summary(pdf)
        page_map(pdf)
        page_weights_importance(pdf)
        well_pages(pdf)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
