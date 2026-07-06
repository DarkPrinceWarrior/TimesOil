"""Графики для отчёта: сравнение моделей, примеры скважин, прогноз вперёд."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from timesoil.data import load_monthly, producer_matrices

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "docs" / "figures"

MAIN_CUTOFF = "2015-05-01"
EXAMPLE_WELLS = (30, 9, 54, 33)


def fig_summary() -> None:
    s = pd.read_csv(RES / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    names = {"oil_tpd": "Нефть", "liq_tpd": "Жидкость"}
    for ax, target in zip(axes, ("oil_tpd", "liq_tpd")):
        g = s[s.target == target].sort_values("wape")
        def color(m: str) -> str:
            if m.startswith(("crm", "frac")):
                return "tab:blue"      # физические модели (CRM, Джентил)
            if "tirex" in m:
                return "tab:green"
            return "tab:red" if m == "spdm" else "tab:gray"

        colors = [color(m) for m in g.model]
        ax.barh(g.model, g.wape * 100, color=colors)
        ax.set_title(f"{names[target]}: WAPE за 3 среза x 6 мес, %")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for y, v in enumerate(g.wape * 100):
            ax.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "summary_wape.png", dpi=120)


def fig_examples() -> None:
    df = load_monthly()
    mat = producer_matrices(df)["oil_tpd"]
    tirex = pd.read_csv(RES / "tirex_blocks_cov_crm_oil_tpd.csv", parse_dates=["date", "cutoff"])
    arps = pd.read_csv(RES / "baseline_arps36_oil_tpd.csv", parse_dates=["date", "cutoff"])
    frac = pd.read_csv(RES / "frac_crm_oil_tpd.csv", parse_dates=["date", "cutoff"])
    tirex = tirex[tirex.cutoff == MAIN_CUTOFF]
    arps = arps[arps.cutoff == MAIN_CUTOFF]
    frac = frac[frac.cutoff == MAIN_CUTOFF]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, w in zip(axes.flat, EXAMPLE_WELLS):
        hist = mat[w].dropna().loc["2012-01-01":]
        ax.plot(hist.index, hist.values, "k-", lw=1.5, label="факт")
        t = tirex[tirex.well == w].sort_values("date")
        a = arps[arps.well == w].sort_values("date")
        ax.fill_between(t.date, t.q10, t.q90, alpha=0.25, color="tab:green",
                        label="TiRex-2: 10–90 проц.")
        ax.plot(t.date, t.y_pred, "-o", ms=3, color="tab:green", label="TiRex-2 (медиана)")
        ax.plot(a.date, a.y_pred, "--s", ms=3, color="tab:orange", label="Арпс")
        f = frac[frac.well == w].sort_values("date")
        ax.plot(f.date, f.y_pred, "-^", ms=3, color="tab:blue", label="CRM × Джентил")
        ax.axvline(pd.Timestamp(MAIN_CUTOFF), color="gray", ls=":", lw=1)
        ax.set_title(f"скв. {w}: нефть, т/сут")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "examples_oil.png", dpi=120)


def fig_forward() -> None:
    df = load_monthly()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = {"oil_tpd": "Нефть", "liq_tpd": "Жидкость"}
    for ax, target in zip(axes, ("oil_tpd", "liq_tpd")):
        fwd = pd.read_csv(RES / f"forward_{target}.csv", parse_dates=["date"])
        field = fwd.groupby("date")[["q10", "y_pred", "q90"]].sum()
        hist = (
            df[df.well.isin(fwd.well.unique())]
            .groupby("date")[target].sum().loc["2013-01-01":]
        )
        ax.plot(hist.index, hist.values, "k-", lw=1.5, label="факт")
        ax.fill_between(field.index, field.q10, field.q90, alpha=0.25, color="tab:blue")
        ax.plot(field.index, field.y_pred, "-o", ms=4, color="tab:blue",
                label="прогноз 2015-12..2016-05")
        ax.set_title(f"{names[target]} по полю, т/сут")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "forward_field.png", dpi=120)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_summary()
    fig_examples()
    fig_forward()
    print("figures saved to", FIGS)
