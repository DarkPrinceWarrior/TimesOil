"""CRMP с настройкой на пластовое давление (по мотивам семейства CRMP-TM).

Первоисточники идеи:
- Степанов С.В., Ручкин А.А., Бекман А.Д. и др. Обзор оригинальных методов...
  на основе прокси-моделей семейства CRM // PROНЕФТЬ. 2025;10(3):44-59;
- Бекман А.Д. Автореферат (ИПНГ РАН): целевые функции с невязкой по давлению —
  формулы (9)-(11) (CRMP-ML6: невязка по дебиту + рассогласование давлений
  Дарси/матбаланс) и (13)-(16) (CRMP-TM: матбаланс дренажного объёма с
  проводимостями T_j, настройка на разности пластовых давлений).

Отличие наших данных: помесячное ПЛАСТОВОЕ давление известно по каждой
добывающей (p_res), поэтому давление входит в адаптацию не косвенно
(через рассогласование двух модельных оценок), а прямо — как данные.

Модель (по блокам; экраны разломов = структурные нули связностей):
    c_t V_pj dP_j/dt = sum_i f_ij I_i(t) + e_j - q_j(t) + sum_k T_jk (P_k - P_j)
    q_j(t) = J_j(t) (P_j(t) - P_wf),   P_wf = 60 атм = const (факт данных)
где f_ij — связности (sum_j f_ij <= 1, штраф), c_t V_pj — упругоёмкость
дренажного объёма (м3/атм), J_j(t) = J_j0 exp(g_j t) — коэффициент
продуктивности (м3/сут/атм) с медленным дрейфом (в данных наблюдаемое
q/(p_res - P_wf) растёт ~6% за 2 года — рост подвижности с обводнением;
|g_j| <= 0.3 1/год), e_j >= 0 — приток извне (аквифер/несмоделированный
источник; блок B2 добывает заметно больше закачки), T_jk >= 0 —
межскважинные проводимости внутри блока (опция). Интегрирование — неявный
Эйлер по месячной сетке (сутки месяца).

Адаптация: scipy.optimize.least_squares (TRF, bounds), невязки по q И по P_j
с балансирующим весом w_p; f_ij в [0,1] + штраф на sum_j f_ij > 1; c_t V_p и
J — в лог-параметризации (положительность + сравнимые масштабы).

Прогноз: закачка на горизонте известна (план ППД), стартовое давление —
фактическое p_res на месяц среза (ассимиляция данных) -> интегрируем P_j ->
q_j = J_j (P_j - P_wf).

Единицы: q в т/сут, закачка в м3/сут — рассогласование масштабов поглощается
f_ij (жидкость преимущественно вода, плотность ~1), как в базовом CRM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .crm import BLOCKS, FULL_START
from .wells import INJECTORS, PRODUCERS, block_wells

P_WF = 60.0  # забойное давление добывающих, атм (в данных строго константа)

# границы параметров (лог-пространство для ctv и J)
_LOG_CTV_BOUNDS = (np.log(1e0), np.log(1e8))   # c_t V_p, м3/атм
_LOG_J_BOUNDS = (np.log(1e-3), np.log(1e3))    # J, м3/сут/атм
_E_FRAC_MAX = 1.0    # e_j <= доля от среднего дебита скважины
_T_HAT_MAX = 3.0     # T_jk <= 3 * медианный J блока
_TAU0_DAYS = 45.0    # стартовая постоянная времени ctv/J
_G_MAX = 0.3         # |g_j| <= 0.3 1/год (дрейф продуктивности)


def _simulate(
    p0: np.ndarray,        # (NP,) давление на стартовый месяц
    inj: np.ndarray,       # (K, NI) закачка на месяцы 1..K, м3/сут
    dt: np.ndarray,        # (K,) длительности месяцев, сут
    f: np.ndarray,         # (NP, NI)
    ctv: np.ndarray,       # (NP,)
    Jt: np.ndarray,        # (K, NP) продуктивность по месяцам 1..K
    e: np.ndarray,         # (NP,)
    T: np.ndarray | None,  # (NP, NP) симметричная, нулевая диагональ
    p_wf: float = P_WF,
) -> np.ndarray:
    """Неявный Эйлер: давления P (K, NP) на месяцы 1..K."""
    K, NP_ = len(dt), len(p0)
    P = np.empty((K, NP_))
    p_prev = p0.astype(float).copy()
    if T is not None:
        lap = np.diag(T.sum(axis=1)) - T
        eye = np.eye(NP_)
    for t in range(K):
        a = dt[t] / ctv
        b = p_prev + a * (f @ inj[t] + e + Jt[t] * p_wf)
        if T is None:
            p_prev = b / (1.0 + a * Jt[t])
        else:
            A = eye + a[:, None] * (np.diag(Jt[t]) + lap)
            p_prev = np.linalg.solve(A, b)
        P[t] = p_prev
    return P


def _rates(P: np.ndarray, Jt: np.ndarray, p_wf: float = P_WF) -> np.ndarray:
    return Jt * (P - p_wf)


@dataclass
class CrmpPressure:
    """Подогнанная модель одного блока (или всего поля)."""

    producers: list[int]
    injectors: list[int]
    f: np.ndarray            # (NP, NI) связности
    ctv: np.ndarray          # (NP,) c_t V_p, м3/атм
    J0: np.ndarray           # (NP,) продуктивность на месяц ref, м3/сут/атм
    g: np.ndarray            # (NP,) дрейф продуктивности, 1/год
    e: np.ndarray            # (NP,) внешний приток, м3(т)/сут
    T: np.ndarray | None     # (NP, NP) межскважинные проводимости
    ref: pd.Timestamp        # начало окна адаптации (отсчёт дрейфа J)
    p_wf: float = P_WF

    @property
    def gains(self) -> pd.DataFrame:
        return pd.DataFrame(self.f, index=self.producers, columns=self.injectors)

    def J_at(self, index: pd.DatetimeIndex) -> np.ndarray:
        """(K, NP) продуктивность на месяцы index: J0 exp(g * годы от ref)."""
        yrs = _years(index, self.ref)
        return self.J0[None, :] * np.exp(np.outer(yrs, self.g))

    def simulate(
        self, p0: np.ndarray, inj: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(давления, дебиты >= 0) на месяцы индекса inj от старта p0."""
        Jt = self.J_at(inj.index)
        P = _simulate(
            p0, inj[self.injectors].to_numpy(), _days(inj.index),
            self.f, self.ctv, Jt, self.e, self.T, self.p_wf,
        )
        q = np.maximum(_rates(P, Jt, self.p_wf), 0.0)
        cols = self.producers
        return (
            pd.DataFrame(P, index=inj.index, columns=cols),
            pd.DataFrame(q, index=inj.index, columns=cols),
        )


def _days(index: pd.DatetimeIndex) -> np.ndarray:
    return index.days_in_month.to_numpy(float)


def _years(index: pd.DatetimeIndex, ref: pd.Timestamp) -> np.ndarray:
    return (index - ref).days.to_numpy(float) / 365.25


def _unpack(
    x: np.ndarray, NP_: int, NI: int, qbar: np.ndarray, t_scale: float, use_T: bool
) -> tuple[np.ndarray, ...]:
    f = x[: NP_ * NI].reshape(NP_, NI)
    ctv = np.exp(x[NP_ * NI : NP_ * NI + NP_])
    J0 = np.exp(x[NP_ * NI + NP_ : NP_ * NI + 2 * NP_])
    g = x[NP_ * NI + 2 * NP_ : NP_ * NI + 3 * NP_]
    e = x[NP_ * NI + 3 * NP_ : NP_ * NI + 4 * NP_] * qbar
    T = None
    if use_T:
        iu = np.triu_indices(NP_, k=1)
        T = np.zeros((NP_, NP_))
        T[iu] = x[NP_ * NI + 4 * NP_ :] * t_scale
        T += T.T
    return f, ctv, J0, g, e, T


def fit_block(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    pres: pd.DataFrame,
    producers: list[int],
    injectors: list[int],
    cutoff: pd.Timestamp,
    start: pd.Timestamp = FULL_START,
    w_p: float = 1.0,
    use_T: bool = False,
    lam_f: float = 3.0,
    lam_e: float = 0.3,
    lam_t: float = 0.1,
    max_nfev: int | None = None,
    tr_solver: str | None = None,
) -> CrmpPressure:
    """Совместная адаптация блока на [start, cutoff] по дебитам И давлениям."""
    q_obs = liq.loc[start:cutoff, producers].to_numpy()
    p_obs = pres.loc[start:cutoff, producers].to_numpy()
    inj_w = inj.loc[start:cutoff, injectors]
    dt = _days(inj_w.index)[1:]
    inj_steps = inj_w.to_numpy()[1:]
    yrs = _years(inj_w.index, start)                 # годы от начала окна

    NT, NP_ = q_obs.shape
    NI = len(injectors)
    qbar = q_obs.mean(axis=0)                        # средний дебит скважины
    q_scale = float(np.abs(q_obs).mean())            # масштаб невязки по q
    p_scale = max(float(p_obs.std()), 2.0)           # масштаб невязки по P
    J_init = qbar / np.maximum(p_obs.mean(axis=0) - P_WF, 5.0)
    t_scale = float(np.median(J_init))

    # x = [f | log ctv | log J0 | g | e/qbar | T/t_scale?]
    n_t = NP_ * (NP_ - 1) // 2 if use_T else 0
    x0 = np.concatenate([
        np.full(NP_ * NI, 0.5 / NP_),
        np.log(J_init * _TAU0_DAYS),
        np.log(J_init),
        np.zeros(NP_),
        np.full(NP_, 1e-3),
        np.zeros(n_t),
    ])
    lo = np.concatenate([
        np.zeros(NP_ * NI),
        np.full(NP_, _LOG_CTV_BOUNDS[0]), np.full(NP_, _LOG_J_BOUNDS[0]),
        np.full(NP_, -_G_MAX),
        np.zeros(NP_),
        np.zeros(n_t),
    ])
    hi = np.concatenate([
        np.ones(NP_ * NI),
        np.full(NP_, _LOG_CTV_BOUNDS[1]), np.full(NP_, _LOG_J_BOUNDS[1]),
        np.full(NP_, _G_MAX),
        np.full(NP_, _E_FRAC_MAX),
        np.full(n_t, _T_HAT_MAX),
    ])

    w_reg_f = lam_f * np.sqrt(NT * NP_)   # штраф sum_j f_ij > 1
    w_reg_e = lam_e * np.sqrt(NT)         # мягкий L2 на приток извне
    w_reg_t = lam_t * np.sqrt(NT)         # мягкий L2 на проводимости

    def residuals(x: np.ndarray) -> np.ndarray:
        f, ctv, J0, g, e, T = _unpack(x, NP_, NI, qbar, t_scale, use_T)
        Jt = J0[None, :] * np.exp(np.outer(yrs, g))  # (NT, NP)
        P = _simulate(p_obs[0], inj_steps, dt, f, ctv, Jt[1:], e, T)
        q = _rates(P, Jt[1:])
        rq = ((q - q_obs[1:]) / q_scale).ravel()
        rq0 = (Jt[0] * (p_obs[0] - P_WF) - q_obs[0]) / q_scale  # месяц 0: калибровка J
        rp = (w_p * (P - p_obs[1:]) / p_scale).ravel()
        reg_f = w_reg_f * np.maximum(f.sum(axis=0) - 1.0, 0.0)
        reg_e = w_reg_e * (e / q_scale)
        parts = [rq, rq0, rp, reg_f, reg_e]
        if use_T:
            iu = np.triu_indices(NP_, k=1)
            parts.append(w_reg_t * (T[iu] / t_scale))
        return np.concatenate(parts)

    sol = least_squares(
        residuals, x0, bounds=(lo, hi), method="trf",
        x_scale="jac", max_nfev=max_nfev, tr_solver=tr_solver,
    )
    f, ctv, J0, g, e, T = _unpack(sol.x, NP_, NI, qbar, t_scale, use_T)
    return CrmpPressure(list(producers), list(injectors), f, ctv, J0, g, e, T, ref=start)


def forecast_block(
    model: CrmpPressure,
    inj: pd.DataFrame,
    pres: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    start: pd.Timestamp = FULL_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ряд «история + горизонт»: (дебиты, давления).

    История — модельная симуляция от start (диагностика); горизонт —
    интегрирование от фактического p_res на месяц среза при известной закачке.
    """
    idx = inj.loc[start:].index
    end = idx[idx.get_loc(cutoff) + horizon]

    hist_inj = inj.loc[start:cutoff, model.injectors].iloc[1:]
    p_hist, q_hist = model.simulate(
        pres.loc[start, model.producers].to_numpy(), hist_inj
    )
    fut_inj = inj.loc[cutoff:end, model.injectors].iloc[1:]
    p_fut, q_fut = model.simulate(
        pres.loc[cutoff, model.producers].to_numpy(), fut_inj
    )
    return pd.concat([q_hist, q_fut]), pd.concat([p_hist, p_fut])


def crmp_pressure_forecast(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    pres: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    w_p: float = 1.0,
    use_T: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CRMP-давление по всем блокам.

    Возвращает (дебиты история+горизонт [даты x 33 скв.],
    давления история+горизонт, связности f_ij [добывающие x нагнетательные];
    межблоковые связности — структурные нули).
    """
    preds_q, preds_p, gains_parts = [], [], []
    for b in BLOCKS:
        prods = [w for w in block_wells(b, injectors=False) if w in liq.columns]
        injs = block_wells(b, injectors=True)
        model = fit_block(liq, inj, pres, prods, injs, cutoff, w_p=w_p, use_T=use_T)
        q, p = forecast_block(model, inj, pres, cutoff, horizon)
        preds_q.append(q)
        preds_p.append(p)
        gains_parts.append(model.gains)
    pred_q = pd.concat(preds_q, axis=1)[list(PRODUCERS)]
    pred_p = pd.concat(preds_p, axis=1)[list(PRODUCERS)]
    gains = pd.concat(gains_parts).reindex(
        index=list(PRODUCERS), columns=sorted(INJECTORS)
    )
    return pred_q, pred_p, gains.fillna(0.0)


def crmp_pressure_full_field_gains(
    liq: pd.DataFrame,
    inj: pd.DataFrame,
    pres: pd.DataFrame,
    cutoff: pd.Timestamp,
    w_p: float = 1.0,
    max_nfev: int | None = 150,
) -> pd.DataFrame:
    """Адаптация всего поля без блоковых ограничений — проверка блоков данными.

    Без межскважинных проводимостей (T) — 528 связностей и так на грани
    идентифицируемости; max_nfev и tr_solver='lsmr' держат время в минутах
    (структурный вывод — концентрация связности по блокам — складывается
    задолго до полной сходимости параметров).
    """
    model = fit_block(
        liq, inj, pres, list(PRODUCERS), sorted(INJECTORS), cutoff,
        w_p=w_p, use_T=False, max_nfev=max_nfev, tr_solver="lsmr",
    )
    return model.gains
