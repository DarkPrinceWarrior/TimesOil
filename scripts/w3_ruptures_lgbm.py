"""Волна 3, секция 1: признаки смены режима закачки (ruptures) для LightGBM.

PELT (модель rbf) по суммарной закачке блока и по адресной закачке скважины.
Признаки: «месяцев с последней смены режима» и «величина последнего скачка (%)»
для обоих рядов. Без утечки: для обучающих дат детекция каузальная (только по
истории до самой даты), для тестовых дат — заморожена на срезе.

Логика LightGBM скопирована из scripts/run_lgbm.py (оригинал не меняется);
вместо одного cross_validation(n_windows=3) — три отдельных окна (n_windows=1
на усечённой до «срез+6 мес» таблице), что даёт те же обучающие выборки
(refit=True), но позволяет подставлять признаки своего среза.

Запуск: uv run --with ruptures python scripts/w3_ruptures_lgbm.py
Выходы: results/w3_lgbm_cp_<цель>.csv (вариант с признаками),
        results/w3_lgbm_base_<цель>.csv (база для честного сравнения).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import ruptures as rpt
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.data import (
    injection_matrix, load_monthly, producer_matrices,
    static_features, well_coords,
)
from timesoil.metrics import wape
from timesoil.mlprep import combined_crm, long_frame
from timesoil.wells import PRODUCERS, WELL_BLOCK, block_wells

OUT = Path(__file__).resolve().parents[1] / "results"

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
    random_state=0, verbosity=-1,
)

PEN = 3.0        # штраф PELT (rbf)
MIN_SIZE = 3     # минимальная длина сегмента, мес
MIN_START = 12   # до этой длины истории детекция не запускается
CP_COLS = ["cp_months_blk", "cp_jump_blk", "cp_months_alloc", "cp_jump_alloc"]


def _last_change(prefix: np.ndarray) -> int | None:
    """Позиция начала последнего режима по PELT(rbf) на префиксе ряда."""
    n = len(prefix)
    if n < 2 * MIN_SIZE + 2 or float(np.std(prefix)) < 1e-12:
        return None
    algo = rpt.Pelt(model="rbf", min_size=MIN_SIZE, jump=1).fit(
        prefix.reshape(-1, 1).astype(float))
    bkps = algo.predict(pen=PEN)
    cps = [b for b in bkps if b < n]
    return cps[-1] if cps else None


def _jump_pct(prefix: np.ndarray, cp: int) -> float:
    """Скачок среднего уровня в последней смене, % от среднего |уровня| ряда."""
    before = prefix[max(0, cp - 6):cp]
    after = prefix[cp:cp + 6]
    denom = max(float(np.abs(prefix).mean()), 1e-9)
    return 100.0 * (float(after.mean()) - float(before.mean())) / denom


def causal_cp_arrays(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Каузальные массивы по позициям t: старт последнего режима (-1 если нет)
    и скачок (%), детекция только по данным до t включительно."""
    vals = series.to_numpy(float)
    n = len(vals)
    last_cp = np.full(n, -1, dtype=int)
    jump = np.zeros(n)
    prev = None
    for t in range(MIN_START, n):
        cp = _last_change(vals[:t + 1])
        if cp is not None:
            last_cp[t] = cp
            # скачок пересчитывается по мере накопления «после»-окна
            jump[t] = _jump_pct(vals[:t + 1], cp)
        elif prev is not None:
            last_cp[t] = prev[0]
            jump[t] = prev[1]
        if last_cp[t] >= 0:
            prev = (last_cp[t], jump[t])
    return last_cp, jump


def freeze_features(last_cp: np.ndarray, jump: np.ndarray,
                    cut_pos: int, end_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Признаки по позициям 0..end_pos: до среза — каузальные, после — детекция
    заморожена на срезе (месяцы продолжают тикать, скачок константен)."""
    ms = np.empty(end_pos + 1)
    jm = np.empty(end_pos + 1)
    for t in range(end_pos + 1):
        tt = min(t, cut_pos)
        base = last_cp[tt] if last_cp[tt] >= 0 else 0
        ms[t] = t - base
        jm[t] = jump[tt]
    return ms, jm


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)
    alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
    blk_sum = pd.DataFrame({
        w: inj[block_wells(WELL_BLOCK[w], injectors=True)].sum(axis=1)
        for w in PRODUCERS
    })

    crm_covs = {}
    for cutoff in CUTOFFS:
        p = OUT / f"crm_cov_{cutoff:%Y%m}.csv"
        crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)
    crm_mat = combined_crm(crm_covs, CUTOFFS)

    st = static_features().reset_index()
    st = st[st.well.isin(PRODUCERS)]
    static_df = pd.DataFrame(dict(
        unique_id=st.well.astype(str), perm=st.perm_md, poro=st.poro,
        h_eff=st.h_eff, block=st.block.astype("category").cat.codes,
    ))
    static_cols = list(static_df.columns.drop("unique_id"))

    frames = long_frame({"oil_tpd": mats["oil_tpd"], "liq_tpd": mats["liq_tpd"]},
                        alloc, blk_sum, crm_mat, mats["p_res"])

    # --- каузальные массивы смен режима: блоки + адресная закачка скважин ---
    idx = inj.index
    pos = {d: i for i, d in enumerate(idx)}
    blocks = sorted({WELL_BLOCK[w] for w in PRODUCERS})
    blk_series = {b: inj[block_wells(b, injectors=True)].sum(axis=1) for b in blocks}
    print("детекция смен режима (каузальная, PELT rbf, pen=%.1f)..." % PEN, flush=True)
    blk_cp = {b: causal_cp_arrays(s) for b, s in blk_series.items()}
    well_cp = {w: causal_cp_arrays(alloc[w]) for w in PRODUCERS}
    n_ch = {b: len(set(blk_cp[b][0][blk_cp[b][0] >= 0])) for b in blocks}
    print("  найдено различных смен по блокам:", n_ch, flush=True)

    # --- таблица признаков смен на срез ---
    def cp_frame(cutoff: pd.Timestamp) -> pd.DataFrame:
        cut_pos = pos[cutoff]
        end_pos = pos[cutoff + pd.DateOffset(months=HORIZON)]
        rows = []
        for w in PRODUCERS:
            mb, jb = freeze_features(*blk_cp[WELL_BLOCK[w]], cut_pos, end_pos)
            ma, ja = freeze_features(*well_cp[w], cut_pos, end_pos)
            rows.append(pd.DataFrame(dict(
                unique_id=str(w), ds=idx[:end_pos + 1],
                cp_months_blk=mb, cp_jump_blk=jb,
                cp_months_alloc=ma, cp_jump_alloc=ja,
            )))
        return pd.concat(rows, ignore_index=True)

    cp_frames = {c: cp_frame(c) for c in CUTOFFS}

    # --- прогоны: база и база+признаки, окно на окно ---
    for tname, df_long in frames.items():
        df_long = df_long.merge(static_df, on="unique_id", how="left")
        results = {}
        for variant in ("base", "cp"):
            parts = []
            for cutoff in CUTOFFS:
                end = cutoff + pd.DateOffset(months=HORIZON)
                dfc = df_long[df_long.ds <= end].copy()
                if variant == "cp":
                    dfc = dfc.merge(cp_frames[cutoff], on=["unique_id", "ds"], how="left")
                mlf = MLForecast(
                    models={"lgbm": lgb.LGBMRegressor(**LGB_PARAMS)},
                    freq="MS",
                    lags=[1, 2, 3, 4, 5, 6, 12],
                    lag_transforms={1: [RollingMean(3), RollingMean(6)]},
                )
                cv = mlf.cross_validation(
                    dfc, n_windows=1, h=HORIZON, step_size=HORIZON, refit=True,
                    static_features=static_cols,
                )
                cv = cv.dropna(subset=["y"]).reset_index(drop=True)
                part = pd.DataFrame(dict(
                    cutoff=cutoff, well=cv["unique_id"].astype(int), date=cv["ds"],
                    y_true=cv["y"].astype(float),
                    y_pred=np.maximum(cv["lgbm"].astype(float), 0.0),
                ))
                parts.append(part)
            res = pd.concat(parts, ignore_index=True)
            res["step"] = res.groupby(["cutoff", "well"]).cumcount() + 1
            res.to_csv(OUT / f"w3_lgbm_{variant}_{tname}.csv", index=False)
            results[variant] = res
            print(f"\n=== {tname} | lgbm {variant} ===")
            print(summarize(res).round(4).to_string(index=False), flush=True)

        b, c = results["base"], results["cp"]
        print(f"\n--- {tname}: WAPE база vs +ruptures ---")
        for cutoff in CUTOFFS:
            wb = wape(b[b.cutoff == cutoff].y_true, b[b.cutoff == cutoff].y_pred)
            wc = wape(c[c.cutoff == cutoff].y_true, c[c.cutoff == cutoff].y_pred)
            print(f"  {cutoff:%Y-%m}: base={wb:.4f}  cp={wc:.4f}  d={wc - wb:+.4f}")
        wb, wc = wape(b.y_true, b.y_pred), wape(c.y_true, c.y_pred)
        print(f"  ALL    : base={wb:.4f}  cp={wc:.4f}  d={wc - wb:+.4f}", flush=True)


if __name__ == "__main__":
    main()
