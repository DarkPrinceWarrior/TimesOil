"""Открытый эталон UNISIM-I-H (CEPETRO/UNICAMP) — проверка переносимости.

Помесячная «наблюдённая» история с добавленным шумом: 14 добывающих +
11 нагнетательных, 2013-06..2024-05, дебиты в м3/сут, давления в кгс/см2.
Запечатывающий разлом f3 отделяет восточный блок (PROD023A/24A/25A,
INJ007/INJ010) от главного. Файлы: data/unisim/unisim_ih_*.csv
(конвертация из CMG *.fhf, в git не входят — передавать scp).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "unisim"

BLOCKS_U: dict[str, dict[str, list[str]]] = {
    "MAIN": {
        "producers": ["NA1A", "NA2", "NA3D", "RJS19", "PROD005", "PROD008",
                      "PROD009", "PROD010", "PROD012", "PROD014", "PROD021"],
        "injectors": ["INJ003", "INJ005", "INJ006", "INJ015", "INJ017",
                      "INJ019", "INJ021", "INJ022", "INJ023"],
    },
    "EAST": {
        "producers": ["PROD023A", "PROD024A", "PROD025A"],
        "injectors": ["INJ007", "INJ010"],
    },
}
PRODUCERS_U: tuple[str, ...] = tuple(
    w for b in BLOCKS_U.values() for w in b["producers"]
)
INJECTORS_U: tuple[str, ...] = tuple(
    w for b in BLOCKS_U.values() for w in b["injectors"]
)


def load_unisim(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Матрицы дата x скважина: oil, liq, wct-совместимые ряды, закачка, Pзаб.

    Месяцы до первого ненулевого дебита скважины -> NaN («скважины ещё нет»);
    закачка до старта нагнетательной -> 0.
    """
    prod = pd.read_csv(Path(data_dir) / "unisim_ih_production_monthly.csv",
                       parse_dates=["date"])
    inj = pd.read_csv(Path(data_dir) / "unisim_ih_injection_monthly.csv",
                      parse_dates=["date"])

    def matrix(df: pd.DataFrame, col: str, wells: tuple[str, ...]) -> pd.DataFrame:
        m = df.pivot(index="date", columns="well", values=col).reindex(columns=list(wells))
        return m

    oil = matrix(prod, "oil_rate_m3d", PRODUCERS_U)
    liq = matrix(prod, "liquid_rate_m3d", PRODUCERS_U)
    bhp = matrix(prod, "bhp_kgf_cm2", PRODUCERS_U)
    started = liq.fillna(0.0).cumsum() > 0
    oil, liq, bhp = oil.where(started), liq.where(started), bhp.where(started)

    winj = matrix(inj, "water_inj_rate_m3d", INJECTORS_U).fillna(0.0)
    winj = winj.reindex(liq.index).fillna(0.0)
    return {"oil": oil, "liq": liq, "bhp": bhp, "winj": winj}


def coords_unisim(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    w = pd.read_csv(Path(data_dir) / "unisim_ih_wells.csv")
    return w.set_index("well")[["x_utm_m", "y_utm_m"]].rename(
        columns={"x_utm_m": "x", "y_utm_m": "y"}
    )


def distance_weights(coords: pd.DataFrame, eta: float = 2.0) -> pd.DataFrame:
    """Веса 1/d^eta внутри блока (проницаемость по скважинам недоступна),
    нормировка по нагнетательной — доли её закачки."""
    W = pd.DataFrame(0.0, index=list(PRODUCERS_U), columns=list(INJECTORS_U))
    for b in BLOCKS_U.values():
        for j in b["producers"]:
            for i in b["injectors"]:
                d = float(np.hypot(coords.at[j, "x"] - coords.at[i, "x"],
                                   coords.at[j, "y"] - coords.at[i, "y"]))
                W.at[j, i] = 1.0 / d**eta
    col = W.sum(axis=0).replace(0.0, np.nan)
    return W.div(col, axis=1).fillna(0.0)
