"""Обученный стекинг второго уровня на расширенной проверке (14 срезов x 6 мес).

Протокол тот же, что в run_ensemble.py, без утечки: для каждого среза
параметры схемы подбираются на остальных 13, оценка — на исключённом.

Схемы:
  nnls_base    — глобальный NNLS (воспроизведение базы run_ensemble.py);
  greedy_well  — жадный отбор моделей с усреднением (Caruana) отдельно
                 по каждой скважине, по её ошибкам на обучающих срезах;
  nnls_step    — NNLS отдельно на каждый шаг горизонта 1..6;
  lgbm_meta    — мета-модель LightGBM: признаки = прогнозы компонентов +
                 шаг горизонта, блок скважины, обводнённость последнего
                 месяца контекста, возраст скважины в месяцах;
  greedy_step  — жадный по скважинам поверх компонентов + nnls_step
                 (для обучающих срезов nnls_step считается с двойным
                 исключением срезов — без утечки).

Выход: сводная таблица по обеим целям; лучшая схема стекинга сохраняется
как results/ext_stack_<схема>_<цель>.csv (формат как у ext_*-файлов).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import nnls

from timesoil import metrics as M
from timesoil.backtest import CUTOFFS
from timesoil.data import load_monthly, producer_matrices
from timesoil.wells import WELL_BLOCK

SEED = 42
GREEDY_ITER = 30          # максимум добавлений в жадном отборе
GREEDY_MIN_ROWS = 12      # меньше строк по скважине -> глобальные NNLS-веса
OUT = Path(__file__).resolve().parents[1] / "results"

COMPONENTS = {
    "oil_tpd": {
        "frac_crm": "ext_frac_crm_oil_tpd.csv",
        "crm2p": "ext_crm2p_oil_tpd.csv",              # двухфазная CRM (этап 7)
        "chronos": "ext_chronos_oil_tpd.csv",
        "chronos_lora": "chronos_lora_oil_tpd.csv",    # дообученный (этап 7)
        "tirex": "ext_tirex_oil_tpd.csv",
        "lgbm": "ext_lgbm_oil_tpd.csv",
        "tide": "ext_nf_tide_oil_tpd.csv",
    },
    "liq_tpd": {
        "crm": "ext_crm_liq_tpd.csv",
        "crm2p": "ext_crm2p_liq_tpd.csv",
        "tirex": "ext_tirex_liq_tpd.csv",
        "chronos": "ext_chronos_liq_tpd.csv",
        "chronos_lora": "chronos_lora_liq_tpd.csv",
        "lgbm": "ext_lgbm_liq_tpd.csv",
    },
}

BLOCK_CODE = {b: i for i, b in enumerate(sorted(set(WELL_BLOCK.values())))}


# ---------------------------------------------------------------- данные

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
        cols = df[["cutoff", "well", "step", "date", "y_true", "y_pred"]].rename(
            columns={"y_pred": name})
        if base is None:
            base = cols
        else:
            base = base.merge(cols.drop(columns=["y_true", "step"]),
                              on=["cutoff", "well", "date"], how="inner")
    return base.dropna().reset_index(drop=True)


def meta_features() -> pd.DataFrame:
    """Признаки (cutoff, well): обводнённость последнего месяца контекста,
    возраст скважины в месяцах, блок."""
    dfm = load_monthly()
    mats = producer_matrices(dfm)
    liq_m, oil_m = mats["liq_tpd"], mats["oil_tpd"]
    wct_m = (1.0 - oil_m / liq_m.where(liq_m > 0)).clip(0.0, 1.0).ffill()
    age_m = liq_m.notna().cumsum()

    cutoffs = pd.date_range("2013-03-01", "2015-05-01", freq="2MS")
    rows = []
    for c in cutoffs:
        wct_row = wct_m.loc[:c].iloc[-1]
        age_row = age_m.loc[:c].iloc[-1]
        for w in liq_m.columns:
            rows.append(dict(
                cutoff=c, well=str(w),
                wct_last=float(wct_row[w]) if np.isfinite(wct_row[w]) else 0.0,
                age_m=float(age_row[w]),
                block_code=BLOCK_CODE[WELL_BLOCK[int(w)]],
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- веса

def nnls_weights(train: pd.DataFrame, models: list[str]) -> np.ndarray:
    A = train[models].to_numpy(float)
    w, _ = nnls(A, train.y_true.to_numpy(float))
    return w / w.sum() if w.sum() > 0 else np.full(len(models), 1 / len(models))


def greedy_weights(P: np.ndarray, y: np.ndarray, n_iter: int = GREEDY_ITER) -> np.ndarray:
    """Жадный отбор с возвращением (Caruana): усреднение выбранного мешка."""
    errs = np.abs(P - y[:, None]).sum(axis=0)
    counts = np.zeros(P.shape[1])
    j0 = int(np.argmin(errs))
    counts[j0] = 1
    bag_sum, cur_err = P[:, j0].copy(), errs[j0]
    for k in range(2, n_iter + 1):
        cand = np.abs((bag_sum[:, None] + P) / k - y[:, None]).sum(axis=0)
        j = int(np.argmin(cand))
        if cand[j] >= cur_err - 1e-12:
            break
        counts[j] += 1
        bag_sum += P[:, j]
        cur_err = cand[j]
    return counts / counts.sum()


# ---------------------------------------------------------------- схемы
# каждая схема: mat -> DataFrame с y_pred, собранным по leave-one-cutoff-out

def scheme_nnls_base(mat: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    parts = []
    for c in sorted(mat.cutoff.unique()):
        train, test = mat[mat.cutoff != c], mat[mat.cutoff == c].copy()
        w = nnls_weights(train, models)
        test["y_pred"] = np.maximum(test[models].to_numpy(float) @ w, 0.0)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def _greedy_predict(train: pd.DataFrame, test: pd.DataFrame,
                    models: list[str], cand: list[str]) -> np.ndarray:
    """Пер-скважинный жадный отбор на train, применение к test.

    cand — кандидаты для жадного отбора (модели + возможные доп. колонки);
    запасной вариант при нехватке строк — глобальные NNLS-веса моделей.
    """
    w_glob = nnls_weights(train, models)
    w_glob = np.concatenate([w_glob, np.zeros(len(cand) - len(models))])
    preds = np.empty(len(test))
    for well, idx in test.groupby("well").indices.items():
        tr = train[train.well == well]
        if len(tr) < GREEDY_MIN_ROWS:
            w = w_glob
        else:
            w = greedy_weights(tr[cand].to_numpy(float),
                               tr.y_true.to_numpy(float))
        preds[idx] = test.iloc[idx][cand].to_numpy(float) @ w
    return np.maximum(preds, 0.0)


def scheme_greedy_well(mat: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    parts = []
    for c in sorted(mat.cutoff.unique()):
        train, test = mat[mat.cutoff != c], mat[mat.cutoff == c].copy()
        test["y_pred"] = _greedy_predict(train, test, models, models)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def _apply_step_weights(df: pd.DataFrame, models: list[str],
                        w_by_step: dict[int, np.ndarray]) -> np.ndarray:
    out = np.empty(len(df))
    for s, idx in df.groupby("step").indices.items():
        out[idx] = df.iloc[idx][models].to_numpy(float) @ w_by_step[int(s)]
    return out


def _fit_step_weights(train: pd.DataFrame, models: list[str]) -> dict[int, np.ndarray]:
    return {int(s): nnls_weights(g, models) for s, g in train.groupby("step")}


def scheme_nnls_step(mat: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    parts = []
    for c in sorted(mat.cutoff.unique()):
        train, test = mat[mat.cutoff != c], mat[mat.cutoff == c].copy()
        w_by_step = _fit_step_weights(train, models)
        test["y_pred"] = np.maximum(_apply_step_weights(test, models, w_by_step), 0.0)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def scheme_greedy_step(mat: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Жадный по скважинам поверх компонентов + nnls_step.

    Для тестового среза c колонка nnls_step считается на срезах без c;
    для обучающего среза c' — на срезах без c и без c' (двойное исключение),
    т.е. кандидат всюду вне обучающей выборки своих весов.
    """
    parts = []
    for c in sorted(mat.cutoff.unique()):
        train = mat[mat.cutoff != c].copy()
        test = mat[mat.cutoff == c].copy()
        col = np.empty(len(train))
        for c2, idx in train.groupby("cutoff").indices.items():
            w2 = _fit_step_weights(mat[~mat.cutoff.isin([c, c2])], models)
            col[idx] = _apply_step_weights(train.iloc[idx], models, w2)
        train["nnls_step"] = np.maximum(col, 0.0)
        w_test = _fit_step_weights(train, models)
        test["nnls_step"] = np.maximum(
            _apply_step_weights(test, models, w_test), 0.0)
        test["y_pred"] = _greedy_predict(train, test, models,
                                         models + ["nnls_step"])
        parts.append(test.drop(columns="nnls_step"))
    return pd.concat(parts, ignore_index=True)


def scheme_lgbm_meta(mat: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    feats = models + ["step", "block_code", "wct_last", "age_m"]
    parts = []
    for c in sorted(mat.cutoff.unique()):
        train, test = mat[mat.cutoff != c], mat[mat.cutoff == c].copy()
        est = LGBMRegressor(
            objective="l1", n_estimators=300, num_leaves=15,
            learning_rate=0.05, min_child_samples=20,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            random_state=SEED, n_jobs=1, verbose=-1,
        )
        est.fit(train[feats], train.y_true)
        test["y_pred"] = np.maximum(est.predict(test[feats]), 0.0)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- сводка

def evaluate(ens: pd.DataFrame) -> tuple[float, float, float]:
    canon = ens[ens.cutoff.isin(CUTOFFS)]
    return (M.wape(ens.y_true, ens.y_pred),
            M.wape(canon.y_true, canon.y_pred),
            M.cum_error_pct(ens.y_true.to_numpy(), ens.y_pred.to_numpy()))


def main() -> None:
    meta = meta_features()
    summary = []
    for target, files in COMPONENTS.items():
        mat = load_matrix(files)
        models = [m for m in files if m in mat.columns]
        mat = mat.merge(meta, on=["cutoff", "well"], how="left")
        assert mat[["wct_last", "age_m", "block_code"]].notna().all().all()
        print(f"\n##### {target}: {len(models)} моделей, "
              f"{mat.cutoff.nunique()} срезов, {len(mat)} строк #####")

        schemes = {
            "nnls_base": scheme_nnls_base(mat, models),
            "greedy_well": scheme_greedy_well(mat, models),
            "nnls_step": scheme_nnls_step(mat, models),
            "lgbm_meta": scheme_lgbm_meta(mat, models),
            "greedy_step": scheme_greedy_step(mat, models),
        }
        best_name, best_wape = None, np.inf
        for name, ens in schemes.items():
            w14, w3, bias = evaluate(ens)
            summary.append(dict(target=target, scheme=name, wape14=w14,
                                wape3=w3, bias_pct=bias))
            print(f"  {name:12s} WAPE(14)={w14:.4f}  WAPE(3)={w3:.4f}  "
                  f"смещение={bias:+.2f}%")
            if name != "nnls_base" and w14 < best_wape:
                best_name, best_wape = name, w14
        best = schemes[best_name].copy()
        best["_w"] = best.well.astype(int)
        best = best.sort_values(["cutoff", "_w", "step"])
        path = OUT / f"ext_stack_{best_name}_{target}.csv"
        best[["cutoff", "well", "step", "date", "y_true", "y_pred"]].to_csv(
            path, index=False)
        print(f"  -> лучшая схема стекинга: {best_name} "
              f"(WAPE14={best_wape:.4f}), сохранено: {path.name}")

    df = pd.DataFrame(summary)
    print("\n================= СВОДКА =================")
    print(df.to_string(index=False, formatters={
        "wape14": "{:.4f}".format, "wape3": "{:.4f}".format,
        "bias_pct": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
