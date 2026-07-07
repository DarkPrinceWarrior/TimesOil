"""Подготовка длинных таблиц с признаками для обучаемых моделей
(LightGBM через mlforecast, сети через neuralforecast).

Формат: unique_id (скважина), ds (дата), y (цель, т/сут) + признаки:
- inj_alloc — адресная закачка скважины (веса гидропроводности), известна наперёд;
- inj_block — суммарная закачка блока, известна наперёд;
- crm       — ряд ёмкостно-резистивной модели (склейка по срезам без утечки);
- pres_lag6 — пластовое давление лагом 6 мес (известно на весь горизонт h<=6).
"""

from __future__ import annotations

import pandas as pd


def combined_crm(crm_covs: dict, cutoffs) -> pd.DataFrame | None:
    """Склейка рядов CRM разных срезов в одну матрицу дата x скважина без
    утечки: значение на дату d берётся из CRM, подогнанного на самом раннем
    срезе, чей ряд «история+горизонт» покрывает d."""
    parts, prev_end = [], None
    for c in cutoffs:
        crm = crm_covs.get(c)
        if crm is None:
            return None
        seg = crm if prev_end is None else crm.loc[crm.index > prev_end]
        parts.append(seg)
        prev_end = crm.index.max()
    return pd.concat(parts).sort_index()


def long_frame(
    targets: dict[str, pd.DataFrame],
    alloc: pd.DataFrame,
    blk_inj_sum: pd.DataFrame,
    crm_mat: pd.DataFrame | None,
    pres: pd.DataFrame | None,
    fill_na: bool = False,
) -> dict[str, pd.DataFrame]:
    """Длинные таблицы по целям. fill_na=True — заполнение пропусков в
    признаках (для сетей, не терпящих NaN): назад/вперёд по скважине."""
    out = {}
    for tname, mat in targets.items():
        rows = []
        end = mat.dropna(how="all").index.max()
        for w in mat.columns:
            s = mat[w].dropna()
            # скважины, остановленные раньше общего конца, исключаются:
            # окна кросс-валидации строятся от конца каждого ряда
            if len(s) < 24 or s.index.max() != end:
                continue
            df = pd.DataFrame({"unique_id": str(w), "ds": s.index, "y": s.values})
            df["inj_alloc"] = alloc.reindex(s.index)[w].to_numpy(float)
            df["inj_block"] = blk_inj_sum.reindex(s.index)[w].to_numpy(float)
            if crm_mat is not None and w in crm_mat.columns:
                df["crm"] = crm_mat.reindex(s.index)[w].to_numpy(float)
            if pres is not None and w in pres.columns:
                df["pres_lag6"] = pres[w].shift(6).reindex(s.index).to_numpy(float)
            if fill_na:
                df = df.bfill().ffill()
            rows.append(df)
        out[tname] = pd.concat(rows, ignore_index=True)
    return out


def field_dataset(results_dir, cutoffs, fill_na: bool = False):
    """Готовый набор нашего поля: длинные таблицы, статика, матрицы целей."""
    from .allocation import allocate, hydro_weights
    from .data import (
        injection_matrix, load_monthly, producer_matrices,
        static_features, well_coords,
    )
    from .wells import PRODUCERS, WELL_BLOCK, block_wells

    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)
    alloc = allocate(inj, hydro_weights(static_features(), well_coords()))
    blk_sum = pd.DataFrame({
        w: inj[block_wells(WELL_BLOCK[w], injectors=True)].sum(axis=1)
        for w in PRODUCERS
    })
    crm_covs = {}
    for cutoff in cutoffs:
        p = results_dir / f"crm_cov_{cutoff:%Y%m}.csv"
        if p.exists():
            crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)
    crm_mat = combined_crm(crm_covs, cutoffs)
    st = static_features().reset_index()
    st = st[st.well.isin(PRODUCERS)]
    static_df = pd.DataFrame(dict(
        unique_id=st.well.astype(str), perm=st.perm_md, poro=st.poro,
        h_eff=st.h_eff, block=st.block.astype("category").cat.codes,
    ))
    frames = long_frame(
        {"oil_tpd": mats["oil_tpd"], "liq_tpd": mats["liq_tpd"]},
        alloc, blk_sum, crm_mat, mats["p_res"], fill_na=fill_na,
    )
    return frames, static_df, mats
