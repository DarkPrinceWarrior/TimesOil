"""Двухфазная ёмкостно-резистивная модель (CRM2P) с переменной продуктивностью.

Источники: Галеев, Синицына (Вестник ТюмГУ, 2025, 11(4):70-92) — CRMP с ОФП
Кори и нестационарной tau; Zhanabayeva, Pourafshary (Sci. Rep., 2026) —
adapted two-phase CRM, ур. (11)-(17). Ключевые соотношения:

    J_j(t) = J'_j * M_j(t),      tau_j(t) = tau'_j / M_j(t),

где M(S_w) = k_ro(S_w)/mu_o + k_rw(S_w)/mu_w — суммарная подвижность фаз;
рекуррентное решение CRMP (Sci. Rep., ур. 16):

    q_t^k = q_t^{k-1} exp(-dt_k M^k / tau') + (1 - exp(...)) * sum_i f_ij I_i^k;

насыщенность дренажного объёма каждой добывающей — из материального баланса
нефти (Sci. Rep., ур. 17, без сжимаемостного члена):

    V_p,j dS_o,j/dt = -q_o,j(t),   q_o = f_o q_t,   f_o = M_o / (M_o + M_w).

ОФП — по Кори (Галеев, ур. 7-8): k_rw = F_w S_wd^n_w, k_ro = F_o (1-S_wd)^n_o,
S_wd = (S_w - S_wc) / (1 - S_wc - S_or).

Принятые упрощения (адаптация к нашим данным):
- жидкость/нефть в т/сут, закачка в м3/сут: рассогласование масс и объёмов
  поглощается связностями f_ij (как в crm.py) и дренажными объёмами V_p
  (тонно-эквивалент), плотности отдельно не вводятся;
- в f_o и M идентифицируемо только произведение kappa = (F_w/F_o)*(mu_o/mu_w)
  (F_o/mu_o уходит в tau'): mu_o/mu_w — ОДИН параметр поля (пулинговая
  подгонка по всем добывающим), F_b = F_w/F_o — концевая поправка блока;
- параметры Кори — НА БЛОК (S_wc, S_or, n_o, n_w, F_b — 5 шт.), не на
  скважину; на скважину — только V_p,j плюс стандартные CRMP-параметры
  tau'_j и f_ij (внутри блока; межблочные связи — структурные нули);
- начальная насыщенность S_wd0,j — обращением наблюдённой стартовой
  обводнённости через кривую f_w(S_wd) (Галеев, ур. 9-10), не параметр;
- сжимаемостный член в балансе насыщенности отброшен (несжимаемое
  приближение; пласт недонасыщенный, режим вытеснения водой);
- при обучении насыщенность движется по ФАКТИЧЕСКОЙ накопленной нефти
  (teacher forcing), при прогнозе — по прогнозной нефти;
- прогноз стартует с наблюдённого дебита жидкости на срезе (фильтрация
  состояния) и наблюдённой накопленной нефти;
- ограничение sum_j f_ij <= 1 — мягким штрафом.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from .crm import BLOCKS, FULL_START
from .wells import PRODUCERS, block_wells

EPS = 1e-9
SWD_MAX = 0.999
WCT_MIN, WCT_MAX = 1e-4, 0.9995
HALF_LIFE_M = 24.0  # период полураспада весов давности в стадии A, мес.


@dataclass(frozen=True)
class CoreyBlock:
    """Параметры Кори блока; kappa = (F_w/F_o) * (mu_o/mu_w)."""

    swc: float
    sor: float
    n_o: float
    n_w: float
    kappa: float

    @property
    def dsw(self) -> float:
        """Подвижный диапазон водонасыщенности 1 - S_wc - S_or."""
        return max(1.0 - self.swc - self.sor, 0.05)


def _lam(swd: np.ndarray | float, p: CoreyBlock) -> tuple[np.ndarray, np.ndarray]:
    """Относительные подвижности (в единицах F_o/mu_o): (M_o, M_w)."""
    swd = np.clip(swd, 0.0, SWD_MAX)
    return (1.0 - swd) ** p.n_o, p.kappa * swd**p.n_w


def mobility(swd: np.ndarray | float, p: CoreyBlock) -> np.ndarray:
    """Суммарная относительная подвижность M(S_wd)."""
    lam_o, lam_w = _lam(swd, p)
    return lam_o + lam_w


def frac_oil(swd: np.ndarray | float, p: CoreyBlock) -> np.ndarray:
    """Доля нефти в потоке f_o = M_o / (M_o + M_w)."""
    lam_o, lam_w = _lam(swd, p)
    return lam_o / (lam_o + lam_w + EPS)


def invert_fw(fw: float, p: CoreyBlock) -> float:
    """S_wd по заданной обводнённости: f_w(S_wd) = fw (монотонно растёт)."""
    fw = float(np.clip(fw, WCT_MIN, WCT_MAX))
    lo, hi = 1e-9, SWD_MAX

    def g(s: float) -> float:
        return float(1.0 - frac_oil(s, p)) - fw

    if g(lo) >= 0.0:
        return lo
    if g(hi) <= 0.0:
        return hi
    return float(brentq(g, lo, hi, xtol=1e-10))


# ---------------------------------------------------------------------------
# Стадия A: насыщенность и кривая обводнённости (Кори на блок, V_p на скважину)
# ---------------------------------------------------------------------------


@dataclass
class SaturationFit:
    """Итог стадии A для группы скважин."""

    corey: CoreyBlock
    vp: pd.Series        # дренажный поровый объём, тонно-эквивалент
    swd0: pd.Series      # начальная нормированная водонасыщенность
    theta: float         # kappa (пулинг) либо F_b (блок)


def _wct_inputs(
    oil: pd.DataFrame, liq: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(накопленная нефть до начала месяца [T,P], wct [T,P], маска, веса [T])."""
    days = oil.index.days_in_month.to_numpy(float)
    oil_m = np.nan_to_num(oil.to_numpy(float)) * days[:, None]
    cum_prev = np.cumsum(oil_m, axis=0) - oil_m  # тонны ДО начала месяца
    liq_v = np.nan_to_num(liq.to_numpy(float))
    oil_v = np.nan_to_num(oil.to_numpy(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        wct = np.where(liq_v > 1e-6, 1.0 - oil_v / np.maximum(liq_v, EPS), np.nan)
    mask = np.isfinite(wct)
    t = np.arange(len(oil), dtype=float)
    weights = 0.5 ** ((t[-1] - t) / HALF_LIFE_M)
    return cum_prev, wct, mask, weights


def _fw_start(wct: np.ndarray, mask: np.ndarray, j: int, k: int = 6) -> float:
    """Стартовая обводнённость скважины: среднее первых k валидных месяцев."""
    vals = wct[mask[:, j], j][:k]
    return float(np.clip(np.mean(vals), WCT_MIN, WCT_MAX)) if len(vals) else 0.02


def fit_saturation(
    oil: pd.DataFrame,
    liq: pd.DataFrame,
    *,
    mu_ratio: float | None = None,
) -> SaturationFit:
    """Подгонка кривой обводнённости через Кори + материальный баланс.

    mu_ratio=None — пулинговый режим: theta = kappa = mu_o/mu_w (F=1),
    оценивает отношение вязкостей на поле. mu_ratio задан — блочный режим:
    theta = F_b (концевая поправка), kappa = F_b * mu_ratio.
    """
    wells = list(oil.columns)
    n = len(wells)
    cum_prev, wct, mask, w_t = _wct_inputs(oil, liq)
    cum_total = np.maximum(cum_prev[-1] + 1.0, 1e3)

    # идентифицируемо только произведение kappa = F * mu_ratio:
    # пулинг — оценка mu_o/mu_w в физичном диапазоне при типовом F = 0.3;
    # блок — концевая поправка F_b при замороженном mu_ratio поля.
    pooled = mu_ratio is None
    th0, th_lo, th_hi = (4.0, 1.5, 10.0) if pooled else (0.30, 0.02, 2.0)
    x0 = np.concatenate([[0.20, 0.30, 2.0, 2.0, th0], np.log10(cum_total / 0.2)])
    lo = np.concatenate([[0.02, 0.05, 1.0, 1.0, th_lo], np.log10(cum_total / 0.8)])
    hi = np.concatenate([[0.40, 0.45, 4.5, 4.5, th_hi], np.log10(cum_total * 1e3)])

    fw0 = np.array([_fw_start(wct, mask, j) for j in range(n)])

    def unpack(x: np.ndarray) -> tuple[CoreyBlock, np.ndarray, float]:
        swc, sor, n_o, n_w, theta = x[:5]
        kappa = 0.3 * theta if pooled else theta * mu_ratio
        return CoreyBlock(swc, sor, n_o, n_w, kappa), 10.0 ** x[5:], theta

    def residuals(x: np.ndarray) -> np.ndarray:
        p, vp, _ = unpack(x)
        res = []
        for j in range(n):
            m = mask[:, j]
            if not m.any():
                continue
            swd0 = invert_fw(fw0[j], p)
            swd = np.clip(swd0 + cum_prev[m, j] / (vp[j] * p.dsw), 0.0, SWD_MAX)
            fw_model = 1.0 - frac_oil(swd, p)
            res.append((fw_model - wct[m, j]) * w_t[m])
        res.append([50.0 * max(0.0, x[0] + x[1] - 0.8)])  # мягко: swc+sor < 0.8
        return np.concatenate(res)

    sol = least_squares(residuals, x0, bounds=(lo, hi), method="trf", max_nfev=400)
    p, vp, theta = unpack(sol.x)
    swd0 = np.array([invert_fw(fw0[j], p) for j in range(n)])
    return SaturationFit(
        corey=p,
        vp=pd.Series(vp, index=wells),
        swd0=pd.Series(swd0, index=wells),
        theta=float(theta),
    )


def swd_history(
    fit: SaturationFit, oil: pd.DataFrame
) -> pd.DataFrame:
    """Путь S_wd по фактической накопленной нефти (на начало каждого месяца)."""
    days = oil.index.days_in_month.to_numpy(float)
    oil_m = np.nan_to_num(oil.to_numpy(float)) * days[:, None]
    cum_prev = np.cumsum(oil_m, axis=0) - oil_m
    p = fit.corey
    vp = fit.vp[oil.columns].to_numpy(float)
    swd = fit.swd0[oil.columns].to_numpy(float)[None, :] + cum_prev / (vp[None, :] * p.dsw)
    return pd.DataFrame(np.clip(swd, 0.0, SWD_MAX), index=oil.index, columns=oil.columns)


# ---------------------------------------------------------------------------
# Стадия B: CRMP жидкости с tau(t) = tau' / M(t)
# ---------------------------------------------------------------------------


@dataclass
class CrmBlockFit:
    """Итог стадии B: связности и базовые постоянные времени блока."""

    gains: pd.DataFrame   # [добывающие x нагнетательные]
    tau_p: pd.Series      # tau'_j, сут * (единица M)


def _simulate(
    q0: np.ndarray,
    inj_term: np.ndarray,
    m_path: np.ndarray,
    tau_p: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Рекурсия CRMP (Sci. Rep., ур. 16): свободный прогон от q0."""
    a = np.exp(-(dt[:, None] * m_path) / tau_p[None, :])  # [T, P]
    q = np.empty_like(inj_term)
    q[0] = q0
    for k in range(1, len(q)):
        q[k] = q[k - 1] * a[k] + (1.0 - a[k]) * inj_term[k]
    return q


def fit_crm_liquid(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    m_path: pd.DataFrame,
) -> CrmBlockFit:
    """Подгонка f_ij (внутри блока) и tau'_j по истории жидкости.

    Свободная симуляция от первого месяца (как в pywaterflood), мягкий штраф
    за sum_j f_ij > 1 по каждой нагнетательной.
    """
    prods, injs = list(liq.columns), list(inj.columns)
    n_p, n_i = len(prods), len(injs)
    q_obs = np.nan_to_num(liq.to_numpy(float))
    inj_v = np.nan_to_num(inj.to_numpy(float))
    m_v = m_path[prods].to_numpy(float)
    dt = liq.index.days_in_month.to_numpy(float)

    tail = min(24, len(liq))
    g0 = float(np.clip(q_obs[-tail:].sum() / max(inj_v[-tail:].sum(), 1.0), 0.05, 1.0))
    m_med = float(np.median(m_v))
    x0 = np.concatenate([np.full(n_p * n_i, g0 / n_p), [np.log10(120.0 * m_med)] * n_p])
    lo = np.concatenate([np.zeros(n_p * n_i), np.full(n_p, 0.3)])
    hi = np.concatenate([np.ones(n_p * n_i), np.full(n_p, 4.5)])

    pen_scale = 3.0 * float(np.mean(np.abs(q_obs))) * np.sqrt(len(q_obs))

    def residuals(x: np.ndarray) -> np.ndarray:
        f = x[: n_p * n_i].reshape(n_p, n_i)
        tau_p = 10.0 ** x[n_p * n_i :]
        sim = _simulate(q_obs[0], inj_v @ f.T, m_v, tau_p, dt)
        pen = pen_scale * np.maximum(0.0, f.sum(axis=0) - 1.0)
        return np.concatenate([(sim - q_obs).ravel(), pen])

    sol = least_squares(
        residuals, x0, bounds=(lo, hi), method="trf",
        max_nfev=200, x_scale="jac",
    )
    f = sol.x[: n_p * n_i].reshape(n_p, n_i)
    tau_p = 10.0 ** sol.x[n_p * n_i :]
    return CrmBlockFit(
        gains=pd.DataFrame(f, index=prods, columns=injs),
        tau_p=pd.Series(tau_p, index=prods),
    )


# ---------------------------------------------------------------------------
# Стадия C: связный прогноз жидкости и нефти
# ---------------------------------------------------------------------------


def forecast_block(
    sat: SaturationFit,
    crm: CrmBlockFit,
    liq: pd.DataFrame,
    oil: pd.DataFrame,
    inj_future: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогон вперёд: жидкость по CRMP(tau(t)), нефть по f_o(S), S — баланс.

    inj_future — закачка нагнетательных блока на месяцы прогноза (план ППД
    считается известным). Старт (фильтрация состояния): наблюдённый дебит
    жидкости на срезе; насыщенность — обращением наблюдённой обводнённости
    последних месяцев через f_w(S_wd) (Галеев, ур. 9-10), при отсутствии
    валидной обводнённости — по накопленной нефти от начала истории.
    """
    prods = list(liq.columns)
    p = sat.corey
    vp = sat.vp[prods].to_numpy(float)
    days_hist = liq.loc[:cutoff].index.days_in_month.to_numpy(float)
    cum_oil = (np.nan_to_num(oil.loc[:cutoff].to_numpy(float)) * days_hist[:, None]).sum(axis=0)
    swd = np.clip(
        sat.swd0[prods].to_numpy(float) + cum_oil / (vp * p.dsw), 0.0, SWD_MAX
    )
    wct_tail = 1.0 - (oil.loc[:cutoff] / liq.loc[:cutoff].where(liq.loc[:cutoff] > 1e-6))
    for j, w in enumerate(prods):
        recent = wct_tail[w].dropna().tail(3)
        if len(recent):
            swd[j] = invert_fw(float(recent.mean()), p)
    q_prev = np.nan_to_num(liq.loc[cutoff].to_numpy(float))
    f = crm.gains.to_numpy(float)
    tau_p = crm.tau_p[prods].to_numpy(float)

    liq_rows, oil_rows = [], []
    for dt_month in inj_future.index:
        dt = float(dt_month.days_in_month)
        m = mobility(swd, p)
        a = np.exp(-dt * m / tau_p)
        q_t = np.maximum(q_prev * a + (1.0 - a) * (f @ inj_future.loc[dt_month].to_numpy(float)), 0.0)
        f_o = frac_oil(swd, p)
        q_o = f_o * q_t
        liq_rows.append(q_t)
        oil_rows.append(q_o)
        swd = np.clip(swd + q_o * dt / (vp * p.dsw), 0.0, SWD_MAX)
        q_prev = q_t
    idx = inj_future.index
    return (
        pd.DataFrame(liq_rows, index=idx, columns=prods),
        pd.DataFrame(oil_rows, index=idx, columns=prods),
    )


def crm2p_forecast(
    liq: pd.DataFrame,
    oil: pd.DataFrame,
    inj: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """CRM2P по всем блокам: (жидкость, нефть) на горизонте + диагностика.

    Обучение на [FULL_START, cutoff]; mu_o/mu_w — один параметр поля
    (пулинговая стадия A по всем 33 добывающим), затем по блокам: Кори (5),
    V_p на скважину, CRMP-параметры; межблочные связности — нули.
    """
    full_idx = liq.loc[FULL_START:].index
    pos = full_idx.get_loc(cutoff)
    fut_idx = full_idx[pos + 1 : pos + 1 + horizon]

    oil_h = oil.loc[FULL_START:cutoff, list(PRODUCERS)]
    liq_h = liq.loc[FULL_START:cutoff, list(PRODUCERS)]
    pooled = fit_saturation(oil_h, liq_h, mu_ratio=None)
    mu_ratio = pooled.theta

    liq_parts, oil_parts, info = [], [], {"mu_ratio": mu_ratio, "blocks": {}}
    for b in BLOCKS:
        prods = [w for w in block_wells(b, injectors=False) if w in liq.columns]
        injs = block_wells(b, injectors=True)
        oil_b = oil.loc[FULL_START:cutoff, prods]
        liq_b = liq.loc[FULL_START:cutoff, prods]
        inj_b = inj.loc[FULL_START:cutoff, injs]

        sat = fit_saturation(oil_b, liq_b, mu_ratio=mu_ratio)
        m_path = mobility(swd_history(sat, oil_b), sat.corey)
        crm_fit = fit_crm_liquid(liq_b, inj_b, m_path)
        liq_p, oil_p = forecast_block(
            sat, crm_fit, liq_b, oil_b, inj.loc[fut_idx, injs], cutoff
        )
        liq_parts.append(liq_p)
        oil_parts.append(oil_p)
        info["blocks"][b] = {"corey": sat.corey, "F_b": sat.theta,
                             "vp": sat.vp, "gains": crm_fit.gains, "tau_p": crm_fit.tau_p}
    cols = list(PRODUCERS)
    return (
        pd.concat(liq_parts, axis=1)[cols],
        pd.concat(oil_parts, axis=1)[cols],
        info,
    )
