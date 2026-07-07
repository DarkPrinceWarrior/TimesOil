"""Лёгкие сети neuralforecast на нашем поле: 3 среза x 6 мес.

Модели: BiTCN, NHITS, TiDE (компактные конфигурации под 33 ряда x
~100 точек). Признаки: futr = адресная/блочная закачка + ряд CRM,
hist = давление лагом 6, static = проницаемость/пористость/толщина/блок.
Квантильный лосс (уровень 80 %) — интервалы из коробки.

Запуск в ОТДЕЛЬНОМ окружении (torch>=2.9): external/nf/.venv.
Выходы: results/nf_<модель>_<цель>.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.mlprep import field_dataset

OUT = Path(__file__).resolve().parents[1] / "results"
FUTR = ["inj_alloc", "inj_block", "crm"]
HIST = ["pres_lag6"]
STAT = ["perm", "poro", "h_eff", "block"]


def build_models():
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, BiTCN, TiDE

    common = dict(
        h=HORIZON, input_size=24, loss=MQLoss(level=[80]),
        futr_exog_list=FUTR, hist_exog_list=HIST, stat_exog_list=STAT,
        scaler_type="robust", max_steps=1200, val_check_steps=50,
        early_stop_patience_steps=5, batch_size=32, random_seed=0,
    )
    return [
        BiTCN(hidden_size=16, dropout=0.3, **common),
        NHITS(mlp_units=[[128, 128]] * 3, dropout_prob_theta=0.3, **common),
        TiDE(hidden_size=64, decoder_output_dim=8, dropout=0.3, **common),
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frames, static_df, mats = field_dataset(OUT, CUTOFFS, fill_na=True)

    from neuralforecast import NeuralForecast

    for tname, df_long in frames.items():
        nf = NeuralForecast(models=build_models(), freq="MS")
        cv = nf.cross_validation(
            df=df_long, static_df=static_df,
            n_windows=3, step_size=HORIZON, val_size=12, refit=True,
        )
        cv = cv.reset_index() if "unique_id" not in cv.columns else cv
        for model in ("BiTCN", "NHITS", "TiDE"):
            med = f"{model}-median"
            if med not in cv.columns:
                med = model
            res = pd.DataFrame(dict(
                cutoff=cv["cutoff"], well=cv["unique_id"], date=cv["ds"],
                y_true=cv["y"].astype(float),
                y_pred=np.maximum(cv[med].astype(float), 0.0),
                q10=np.maximum(cv.get(f"{model}-lo-80", np.nan), 0.0),
                q90=np.maximum(cv.get(f"{model}-hi-80", np.nan), 0.0),
            )).dropna(subset=["y_true"])
            res["step"] = res.groupby(["cutoff", "well"]).cumcount() + 1
            res.to_csv(OUT / f"nf_{model.lower()}_{tname}.csv", index=False)
            print(f"=== field {tname} | nf-{model} ===")
            print(summarize(res).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
