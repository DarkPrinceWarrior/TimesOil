"""Адресное распределение закачки по добывающим скважинам.

Вместо блочных сумм каждая добывающая получает свою взвешенную закачку
$\\tilde I_j(t) = \\sum_i w_{ij} I_i(t)$. Веса:
- гидропроводность: $w_{ij} \\propto \\bar k_{ij} / d_{ij}^{\\eta}$, где
  $\\bar k$ — гармоническое среднее проницаемостей пары, $d$ — расстояние;
  связи через разломы — нули; нормировка по нагнетательной (доли закачки);
- связность CRM: готовая матрица $f_{ij}$ блочной ёмкостно-резистивной
  модели (уже нормирована ограничением «до единицы»).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .wells import INJECTORS, PRODUCERS, WELL_BLOCK


def hydro_weights(static: pd.DataFrame, coords: pd.DataFrame, eta: float = 2.0) -> pd.DataFrame:
    """Матрица весов [добывающие x нагнетательные] по гидропроводности.

    static: индекс well, колонка perm_md (нагнетательная 19 без статики —
    берётся среднее по её блоку); coords: индекс well, колонки x, y.
    """
    inj = sorted(INJECTORS)
    perm = static["perm_md"].copy()
    for w in inj:
        if w not in perm.index or pd.isna(perm.get(w)):
            blk = WELL_BLOCK[w]
            blk_perm = [perm[p] for p in perm.index if WELL_BLOCK.get(p) == blk]
            perm.loc[w] = float(np.mean(blk_perm))

    W = pd.DataFrame(0.0, index=list(PRODUCERS), columns=inj)
    for j in PRODUCERS:
        for i in inj:
            if WELL_BLOCK[j] != WELL_BLOCK[i]:
                continue  # экран разлома
            d = float(np.hypot(coords.at[j, "x"] - coords.at[i, "x"],
                               coords.at[j, "y"] - coords.at[i, "y"]))
            k_h = 2.0 * perm[j] * perm[i] / (perm[j] + perm[i])
            W.at[j, i] = k_h / d**eta
    # нормировка по нагнетательной: доли её закачки по добывающим блока
    col_sums = W.sum(axis=0).replace(0.0, np.nan)
    return W.div(col_sums, axis=1).fillna(0.0)


def allocate(inj_mat: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Закачка на добывающую: [даты x добывающие] = inj [даты x нагн] @ W.T."""
    cols = weights.columns
    out = inj_mat[cols].to_numpy() @ weights.to_numpy().T
    return pd.DataFrame(out, index=inj_mat.index, columns=weights.index)
