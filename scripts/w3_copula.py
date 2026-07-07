"""Волна 3, секция 3: копула по остаткам ансамбля — диагностика для отчёта.

По остаткам ансамбля NNLS (results/ext_ens_nnls_oil_tpd.csv и
..._liq_tpd.csv, сопоставленным по cutoff/well/date) оценивается гауссова
копула связи относительных ошибок нефти и жидкости:
- корреляции: Пирсон (сырая и winsorized 1%), Спирмен, корреляция
  нормальных скоров (это и есть параметр гауссовой копулы);
- по скважинам: распределение Спирмена, доля скважин с rho > 0.3;
- покрытие совместного прямоугольника из маргинальных 80%-интервалов:
  эмпирическое vs независимость (0.64) vs гауссова копула с оценённой rho.

Вывод: нужна ли копула для согласованных сценариев нефть+жидкость.

Запуск: uv run python scripts/w3_copula.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "results"
EPS = 1e-9


def normal_scores(v: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v) / (len(v) + 1.0)
    return stats.norm.ppf(r)


def rect_prob_gauss(rho: float, p_lo: float = 0.1, p_hi: float = 0.9) -> float:
    """P(оба нормальных скора в [z_lo, z_hi]) для гауссовой копулы с rho."""
    z_lo, z_hi = stats.norm.ppf(p_lo), stats.norm.ppf(p_hi)
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return float(mvn.cdf([z_hi, z_hi]) - mvn.cdf([z_lo, z_hi])
                 - mvn.cdf([z_hi, z_lo]) + mvn.cdf([z_lo, z_lo]))


def main() -> None:
    o = pd.read_csv(OUT / "ext_ens_nnls_oil_tpd.csv", parse_dates=["cutoff", "date"])
    l = pd.read_csv(OUT / "ext_ens_nnls_liq_tpd.csv", parse_dates=["cutoff", "date"])
    m = o.merge(l, on=["cutoff", "well", "date", "step"], suffixes=("_oil", "_liq"))
    print(f"сопоставлено пар нефть/жидкость: {len(m)} "
          f"(нефть {len(o)}, жидкость {len(l)})")

    m["rel_oil"] = (m.y_true_oil - m.y_pred_oil) / np.maximum(np.abs(m.y_pred_oil), EPS)
    m["rel_liq"] = (m.y_true_liq - m.y_pred_liq) / np.maximum(np.abs(m.y_pred_liq), EPS)

    ro, rl = m.rel_oil.to_numpy(), m.rel_liq.to_numpy()

    # --- корреляции (в целом) ---
    pearson = float(np.corrcoef(ro, rl)[0, 1])
    lo_o, hi_o = np.quantile(ro, [0.01, 0.99])
    lo_l, hi_l = np.quantile(rl, [0.01, 0.99])
    row, rlw = np.clip(ro, lo_o, hi_o), np.clip(rl, lo_l, hi_l)
    pearson_w = float(np.corrcoef(row, rlw)[0, 1])
    spearman = float(stats.spearmanr(ro, rl).statistic)
    zo, zl = normal_scores(ro), normal_scores(rl)
    rho_gauss = float(np.corrcoef(zo, zl)[0, 1])

    print("\nкорреляция относительных ошибок нефть vs жидкость (в целом):")
    print(f"  Пирсон (сырая)        : {pearson:+.3f}")
    print(f"  Пирсон (winsor 1%)    : {pearson_w:+.3f}")
    print(f"  Спирмен               : {spearman:+.3f}")
    print(f"  нормальные скоры (rho гауссовой копулы): {rho_gauss:+.3f}")

    # --- по скважинам ---
    per_well = m.groupby("well").apply(
        lambda g: float(stats.spearmanr(g.rel_oil, g.rel_liq).statistic),
        include_groups=False)
    print("\nСпирмен по скважинам:")
    q = per_well.quantile([0.25, 0.5, 0.75])
    print(f"  медиана {q[0.5]:+.3f}, квартели [{q[0.25]:+.3f}, {q[0.75]:+.3f}], "
          f"min {per_well.min():+.3f}, max {per_well.max():+.3f}")
    print(f"  скважин с rho > 0.3: {(per_well > 0.3).sum()} из {len(per_well)}")
    print("  топ-5 по |rho|:", {int(k): round(v, 3) for k, v in
          per_well.reindex(per_well.abs().sort_values(ascending=False).index)
          .head(5).items()})

    # --- совместный прямоугольник 80% x 80% ---
    q_o = np.quantile(ro, [0.1, 0.9])
    q_l = np.quantile(rl, [0.1, 0.9])
    in_o = (ro >= q_o[0]) & (ro <= q_o[1])
    in_l = (rl >= q_l[0]) & (rl <= q_l[1])
    emp_joint = float((in_o & in_l).mean())
    indep = float(in_o.mean() * in_l.mean())  # ~0.8*0.8 = 0.64
    copula = rect_prob_gauss(rho_gauss)

    print("\nсовместное покрытие прямоугольника из маргинальных 80%-интервалов:")
    print(f"  маргинальные покрытия  : нефть {in_o.mean():.3f}, жидкость {in_l.mean():.3f}")
    print(f"  независимость (до)     : {indep:.3f}")
    print(f"  гауссова копула (после): {copula:.3f}")
    print(f"  эмпирическое           : {emp_joint:.3f}")

    # --- вердикт ---
    print("\nвердикт:")
    if rho_gauss < 0.3:
        print(f"  rho = {rho_gauss:.2f} < 0.3 — связь слабая, копула почти не "
              "нужна: независимые маргинальные интервалы недооценивают "
              f"совместное покрытие лишь на {emp_joint - indep:+.3f}.")
    else:
        print(f"  rho = {rho_gauss:.2f} >= 0.3 — связь существенная: для "
              "согласованных сценариев нефть+жидкость стоит сэмплировать "
              "остатки через гауссову копулу; независимость занижает "
              f"совместное покрытие на {emp_joint - indep:.3f} "
              f"({indep:.3f} -> {emp_joint:.3f}, копула даёт {copula:.3f}).")


if __name__ == "__main__":
    main()
