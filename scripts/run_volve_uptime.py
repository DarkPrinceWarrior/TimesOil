"""Volve с учётом наработки часов: прогноз «чистого» дебита на отработанные
сутки (без «шума простоев») и две оценки.

(а) физическая: WAPE на «чистом» дебите (насколько модель ловит физику);
(б) календарная при известном плане работ: прогноз чистого дебита x
    фактическая наработка контрольного окна -> сравнение с календарным
    фактом (сопоставимо с базовой таблицей Volve).

Модели: наивный, Арпс, TiRex-2 (u, cov — закачка как известный план).
Выходы: results/volve_up_*.csv.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from timesoil import baselines as B
from timesoil.backtest import HORIZON, run_pointwise, summarize
from timesoil.tirex_runner import forecast_tirex
from timesoil.volve import INJECTORS_V, PRODUCERS_V, load_volve

OUT = Path(__file__).resolve().parents[1] / "results"
CUTOFFS_V = (
    pd.Timestamp("2015-03-01"),
    pd.Timestamp("2015-09-01"),
    pd.Timestamp("2016-03-01"),
)


def calendar_eval(res: pd.DataFrame, cal_mat: pd.DataFrame, uptime: pd.DataFrame) -> pd.DataFrame:
    """Оценка (б): чистый дебит x фактическая наработка против календарного факта."""
    out = res.copy()
    out["y_pred"] = [
        float(r.y_pred) * float(uptime.at[r.date, r.well])
        if pd.notna(uptime.at[r.date, r.well]) else float("nan")
        for r in out.itertuples()
    ]
    out["y_true"] = [
        float(cal_mat.at[r.date, r.well]) if pd.notna(cal_mat.at[r.date, r.well]) else float("nan")
        for r in out.itertuples()
    ]
    return out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m = load_volve()
    winj, uptime = m["winj"], m["uptime"]

    results: dict[tuple[str, str], pd.DataFrame] = {}

    for target in ("oil", "liq"):
        eff, cal = m[f"{target}_eff"], m[target]
        # эталоны на чистом дебите
        for name, fc in {"naive": B.forecast_naive, "arps36": partial(B.forecast_arps, k=36)}.items():
            res = run_pointwise(eff, fc, cutoffs=CUTOFFS_V).dropna(subset=["y_true"])
            results[(name, target)] = res.reset_index(drop=True)
        # TiRex-2 на чистом дебите
        from tirex2 import load_model

        model = load_model("NX-AI/TiRex-2", device="cpu")
        for name, kw in {
            "tirex_u": dict(variant="u", groups=[[w] for w in PRODUCERS_V]),
            "tirex_cov": dict(variant="blocks_cov", groups=[list(PRODUCERS_V)],
                              group_inj=[list(INJECTORS_V)]),
        }.items():
            parts = []
            for cutoff in CUTOFFS_V:
                fc = forecast_tirex(model, eff, cutoff, HORIZON,
                                    inj_mat=winj, pres_mat=None, inj_future=winj, **kw)
                fc["cutoff"] = cutoff
                fc["y_true"] = [eff.at[d, w] for d, w in zip(fc.date, fc.well)]
                parts.append(fc)
            res = pd.concat(parts, ignore_index=True).dropna(subset=["y_true"])
            results[(name, target)] = res.reset_index(drop=True)

        for name in ("naive", "arps36", "tirex_u", "tirex_cov"):
            res = results[(name, target)]
            res.to_csv(OUT / f"volve_up_{name}_{target}.csv", index=False)
            cal_res = calendar_eval(res, cal, uptime)
            a = summarize(res).round(4).tail(1)
            b = summarize(cal_res).round(4).tail(1)
            print(f"=== volve-uptime {target} | {name} ===")
            print("  (а) чистый дебит:   ", a.to_string(index=False, header=(name == 'naive')))
            print("  (б) календарный (план работ известен):", b.to_string(index=False, header=False), flush=True)


if __name__ == "__main__":
    main()
