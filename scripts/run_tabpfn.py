"""TabPFN-TS (PriorLabs) на панели поля — ИССЛЕДОВАТЕЛЬСКОЕ сравнение.

ЛИЦЕНЗИЯ: веса TabPFN — некоммерческая лицензия Prior Labs; прогон только
для проверки тезиса «модель сильна на коротких рядах с ковариатами»,
в продуктив не идёт.

ПРИВАТНОСТЬ: движок строго ЛОКАЛЬНЫЙ — TabPFNMode.LOCAL (пакет `tabpfn`,
чекпойнт TabPFN-TS-3 скачивается один раз в локальный кэш). Облачный
tabpfn-client НЕ используется, телеметрия отключена
(TABPFN_DISABLE_TELEMETRY=1) — данные заказчика машину не покидают.

Ковариаты — как у лидеров (см. run_lgbm / timesoil.mlprep):
известные наперёд — адресная закачка (inj_alloc), закачка блока (inj_block),
ряд CRM (crm, склейка по срезам без утечки); давление лагом 6 мес
(pres_lag6 — прошлое, но известно на весь горизонт h<=6). Календарные и
сезонные признаки пакет генерирует сам (TABPFN_TS_DEFAULT_FEATURES).

Запуск:
  uv run --with tabpfn-time-series python scripts/run_tabpfn.py
      [--targets oil_tpd liq_tpd] [--ext] [--smoke]

  --smoke: только последний срез (замер скорости на CPU);
  --ext:   14 срезов EXT_CUTOFFS (ряды CRM для них не строились -> без crm,
           как в ext_lgbm).

Выход: results/[ext_]tabpfn_<цель>.csv
       (cutoff, well, step, date, y_true, y_pred, q10, q90)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# до импорта tabpfn_*: телеметрия PostHog выключена, данные не уходят
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, EXT_CUTOFFS, HORIZON, summarize
from timesoil.mlprep import field_dataset

OUT = Path(__file__).resolve().parents[1] / "results"
MIN_HISTORY = 12  # месяцев, как в run_pointwise

QUANTILES = [0.1, 0.5, 0.9]


def _quantile_col(pred: pd.DataFrame, q: float) -> pd.Series:
    """Колонка квантиля в выдаче predict_df: имя может быть float или str."""
    for c in pred.columns:
        if c == q or str(c) == str(q):
            return pred[c]
    raise KeyError(f"квантиль {q} не найден среди колонок {list(pred.columns)}")


def forecast_cutoff(pipeline, df_long: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Один срез: контекст до cutoff, будущее — 6 мес с известными ковариатами."""
    ctx = df_long[df_long.ds <= cutoff]
    fut = df_long[df_long.ds > cutoff].groupby("unique_id", sort=False).head(HORIZON)

    hist_len = ctx.groupby("unique_id").size()
    fut_len = fut.groupby("unique_id").size()
    keep = hist_len[hist_len >= MIN_HISTORY].index.intersection(fut_len[fut_len == HORIZON].index)
    ctx = ctx[ctx.unique_id.isin(keep)]
    fut = fut[fut.unique_id.isin(keep)]

    context_df = ctx.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    future_df = fut.drop(columns="y").rename(columns={"unique_id": "item_id", "ds": "timestamp"})

    pred = pipeline.predict_df(context_df, future_df=future_df, quantiles=QUANTILES).reset_index()
    pred["y_pred"] = np.maximum(_quantile_col(pred, 0.5).to_numpy(float), 0.0)
    pred["q10"] = np.maximum(_quantile_col(pred, 0.1).to_numpy(float), 0.0)
    pred["q90"] = np.maximum(_quantile_col(pred, 0.9).to_numpy(float), 0.0)

    truth = fut.rename(columns={"unique_id": "item_id", "ds": "timestamp"})[
        ["item_id", "timestamp", "y"]
    ]
    pred = pred.merge(truth, on=["item_id", "timestamp"], how="left", validate="1:1")
    assert pred.y.notna().all(), f"нет факта для части прогнозов на срезе {cutoff.date()}"

    res = pd.DataFrame(dict(
        cutoff=cutoff,
        well=pred.item_id.astype(int),
        date=pred.timestamp,
        y_true=pred.y.astype(float),
        y_pred=pred.y_pred,
        q10=pred.q10,
        q90=pred.q90,
    )).sort_values(["well", "date"], ignore_index=True)
    res["step"] = res.groupby("well").cumcount() + 1
    return res[["cutoff", "well", "step", "date", "y_true", "y_pred", "q10", "q90"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="*", default=["oil_tpd", "liq_tpd"])
    ap.add_argument("--ext", action="store_true", help="14 срезов EXT_CUTOFFS")
    ap.add_argument("--smoke", action="store_true", help="только последний срез (замер скорости)")
    args = ap.parse_args()

    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    cutoffs = tuple(EXT_CUTOFFS if args.ext else CUTOFFS)
    if args.smoke:
        cutoffs = cutoffs[-1:]
    prefix = "ext_" if args.ext else ""

    OUT.mkdir(exist_ok=True)
    frames, _static, _mats = field_dataset(OUT, cutoffs)

    # ЛОКАЛЬНЫЙ движок — облачный клиент не задействуется
    pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)

    for target in args.targets:
        df_long = frames[target]
        cov = [c for c in df_long.columns if c not in ("unique_id", "ds", "y")]
        print(f"\n=== {target} | tabpfn-ts (local) | ковариаты: {cov} ===", flush=True)
        parts = []
        for cutoff in cutoffs:
            t0 = time.time()
            parts.append(forecast_cutoff(pipeline, df_long, cutoff))
            print(f"  срез {cutoff.date()}: {len(parts[-1])} строк за {time.time() - t0:.1f}с",
                  flush=True)
        res = pd.concat(parts, ignore_index=True)
        out_path = OUT / f"{prefix}tabpfn_{target}.csv"
        res.to_csv(out_path, index=False)
        print(f"-> {out_path}")
        print(summarize(res).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
