"""Volve (Equinor, 2008-2016) — реальные промысловые данные, третий полигон.

Источник: parquet-выгрузка sumpalabs/petrodb (данные под открытой лицензией
Equinor). 5 добывающих (F-12, F-14 — вся история; F-11, F-15D, F-1C —
молодые, 2013-2014+) и 2 нагнетательные (F-4; F-5 — переведена из добычи).
F-1C остановлена в 2016-04. Один блок; координат в выгрузке нет —
адресное распределение закачки равномерное. Объёмы Sm3/мес -> м3/сут.
Файлы: data/volve/*.parquet (в git не входят — передавать scp).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "volve"

NAMES = {
    5351: "F-14", 5599: "F-12", 5693: "F-4", 5769: "F-5",
    7078: "F-11", 7289: "F-15D", 7405: "F-1C",
}
PRODUCERS_V: tuple[str, ...] = ("F-12", "F-14", "F-11", "F-15D", "F-1C")
INJECTORS_V: tuple[str, ...] = ("F-4", "F-5")


def load_volve(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Матрицы дата x скважина (м3/сут): oil, liq, закачка воды."""
    m = pd.read_parquet(Path(data_dir) / "monthly_production.parquet")
    m["well"] = m["npd_wellbore_code"].map(NAMES)
    m["days"] = m["date"].dt.days_in_month
    m["oil_m3d"] = m["oil_volume_sm3"] / m["days"]
    m["liq_m3d"] = (m["oil_volume_sm3"] + m["water_volume_sm3"]) / m["days"]
    m["winj_m3d"] = m["water_injection_sm3"] / m["days"]

    idx = pd.date_range(m["date"].min(), m["date"].max(), freq="MS")

    def matrix(col: str, wells: tuple[str, ...]) -> pd.DataFrame:
        p = m.pivot(index="date", columns="well", values=col)
        return p.reindex(index=idx, columns=list(wells))

    oil = matrix("oil_m3d", PRODUCERS_V)
    liq = matrix("liq_m3d", PRODUCERS_V)
    winj = matrix("winj_m3d", INJECTORS_V).fillna(0.0)
    return {"oil": oil, "liq": liq, "winj": winj}


def uniform_weights() -> pd.DataFrame:
    """Равномерное распределение закачки по добывающим (координат нет)."""
    W = pd.DataFrame(1.0, index=list(PRODUCERS_V), columns=list(INJECTORS_V))
    return W / len(PRODUCERS_V)
