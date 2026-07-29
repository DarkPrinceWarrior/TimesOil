"""Загрузка и очистка данных месторождения.

Особенности исходных файлов (проверено сверкой обоих Excel, значения идентичны):
- история 2007-05..2015-12, но последний месяц (2015-12) — артефакт выгрузки
  (отрицательные разности накопленных, нулевые давления) + одна полностью
  пустая строка в MODEL_Y -> обрезаем по LAST_VALID;
- колонка "THP" в MODEL_Y на самом деле пластовое давление (совпадает с листом
  Ppl широкого файла; у добывающих "THP" > BHP, что для устья невозможно);
- "Добыча жидкости/нефти, т." = точные разности накопленных WLPT/WOMT;
- нули до первого месяца работы скважины означают "скважины ещё нет"
  (WEFF=0), а не нулевую добычу;
- месячные тонны содержат календарный эффект (28..31 день) -> используем
  среднесуточные величины *_tpd (т/сут) и winj_m3pd (м3/сут).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .wells import INJECTORS, PRODUCERS, WELL_BLOCK

RAW_DIR = Path(__file__).resolve().parents[2] / "raw_data"
DATASET_XLSX = "Dataset.xlsx"
WIDE_XLSX = "Dataset Шутову АА+.xlsx"
SCENARIO_CSVS = {
    "alenkin_space": "Модель MODEL_Y_AlenkinDA.csv",
    "lrat5000": "Модель MODEL_Y_TEST_LRAT5000_AlenkinDA.csv",
    "alenkin_underscore": "Модель_MODEL_Y_AlenkinDA.csv",
}
LAST_VALID = pd.Timestamp("2015-11-01")

RENAME = {
    "DATA": "date",
    "Добыча жидкости, т.": "liq_t",
    "Добыча нефти т.": "oil_t",
    "Закачка воды, м3": "winj_m3",
    "THP": "p_res",  # пластовое давление (см. докстринг)
    "BHP": "p_bhp",  # забойное давление
    "WEFF": "weff",
}

SCENARIO_RENAME = {
    "DATA": "date",
    "WLPT_Diff": "liq_t",
    "WOMT_Diff": "oil_t",
    "WWIT_Diff": "winj_m3",
    "THP": "p_res",
    "BHP": "p_bhp",
    "WEFF": "weff",
}


def _clean_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Общая нормализация MODEL_Y после сопоставления исходных колонок."""
    required = {
        "date", "well", "oil_t", "liq_t", "winj_m3",
        "p_res", "p_bhp", "weff",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"в MODEL_Y отсутствуют колонки: {sorted(missing)}")

    df = df.dropna(subset=["well"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["well"] = df["well"].astype(int)
    df = df[df["date"] <= LAST_VALID].copy()
    if df.duplicated(["date", "well"]).any():
        duplicates = df.loc[
            df.duplicated(["date", "well"], keep=False), ["date", "well"]
        ]
        raise ValueError(
            "дубли date × well в MODEL_Y: "
            f"{duplicates.drop_duplicates().to_dict('records')[:5]}"
        )

    df["days"] = df["date"].dt.days_in_month
    df["oil_tpd"] = df["oil_t"] / df["days"]
    df["liq_tpd"] = df["liq_t"] / df["days"]
    df["winj_m3pd"] = df["winj_m3"] / df["days"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["wct"] = np.where(
            df["liq_t"] > 0, 1.0 - df["oil_t"] / df["liq_t"], np.nan
        )
    df["block"] = df["well"].map(WELL_BLOCK)
    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_monthly(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Помесячные данные всех 49 скважин в длинном формате, без артефактов.

    Колонки: date, well, oil_t, liq_t, winj_m3, p_res, p_bhp, weff,
    days, oil_tpd, liq_tpd, winj_m3pd, wct (массовая обводнённость), block.
    """
    df = pd.read_excel(Path(raw_dir) / DATASET_XLSX, sheet_name="MODEL_Y")
    return _clean_monthly(df.rename(columns=RENAME))


def load_scenario_csv(path: Path | str) -> pd.DataFrame:
    """Загрузить альтернативный MODEL_Y из CSV в канонической схеме проекта."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return _clean_monthly(df.rename(columns=SCENARIO_RENAME))


def load_scenarios(
    raw_dir: Path | str = RAW_DIR,
    include_reference: bool = True,
) -> dict[str, pd.DataFrame]:
    """Сценарии одной модели с одинаковым зерном date × well.

    `reference` — исходный Dataset.xlsx; остальные сценарии перечислены явно
    в :data:`SCENARIO_CSVS`, чтобы случайный CSV из raw_data не попал в обучение.
    """
    raw_dir = Path(raw_dir)
    scenarios = {
        scenario: load_scenario_csv(raw_dir / filename)
        for scenario, filename in SCENARIO_CSVS.items()
    }
    if include_reference:
        scenarios = {"reference": load_monthly(raw_dir), **scenarios}

    expected_keys: pd.MultiIndex | None = None
    for scenario, df in scenarios.items():
        keys = pd.MultiIndex.from_frame(
            df[["date", "well"]].sort_values(["date", "well"])
        )
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise ValueError(
                f"сценарий {scenario!r} не совпадает по ключам date × well"
            )
    return scenarios


def pivot(df: pd.DataFrame, value: str, wells: list[int] | tuple[int, ...] | None = None) -> pd.DataFrame:
    """Матрица date x well для одной величины; месяцы до старта скважины -> NaN."""
    m = df.pivot(index="date", columns="well", values=value)
    if wells is not None:
        m = m[list(wells)]
    started = df.pivot(index="date", columns="well", values="weff").cumsum() > 0
    return m.where(started[m.columns])


def producer_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """oil/liq т/сут и p_res по 33 действующим добывающим."""
    return {v: pivot(df, v, PRODUCERS) for v in ("oil_tpd", "liq_tpd", "p_res")}


def injection_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Закачка м3/сут по 16 нагнетательным (до перевода под нагнетание — 0)."""
    inj = sorted(INJECTORS)
    m = df.pivot(index="date", columns="well", values="winj_m3pd")[inj]
    return m.fillna(0.0)


def well_coords(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Координаты всех 49 скважин (лист coords)."""
    c = pd.read_excel(Path(raw_dir) / DATASET_XLSX, sheet_name="coords")
    c["well"] = c["Well"].str.strip("'").astype(int)
    return (
        c.set_index("well")[["X", "Y"]]
        .rename(columns={"X": "x", "Y": "y"})
        .sort_index()
    )


def static_features(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Статика добывающих из DobXY: проницаемость, пористость, насыщенность, толщина."""
    d = pd.read_excel(Path(raw_dir) / WIDE_XLSX, sheet_name="DobXY")
    d = d.rename(
        columns={
            "skw": "well",
            "Dob_X": "x",
            "Dob_Y": "y",
            "Проницаемость абсолютная, мД": "perm_md",
            "Пористость, %": "poro",
            "Начальная нефтенасыщенность, доли": "so_init",
            "Начальная эффективная нефтенасыщенная толщина, м": "h_eff",
        }
    )
    cols = ["well", "x", "y", "perm_md", "poro", "so_init", "h_eff"]
    d = d[cols].copy()
    d["well"] = d["well"].astype(int)
    d["block"] = d["well"].map(WELL_BLOCK)
    return d.set_index("well").sort_index()
