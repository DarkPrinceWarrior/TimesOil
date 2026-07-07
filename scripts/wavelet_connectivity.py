"""Вейвлет-когерентность «закачка -> добыча жидкости»: независимая
диагностика межскважинной связности и маска-регуляризатор для CRM.

Метод (по Gabry et al., Energies 2026, 19(9):2211): кросс-вейвлет-когерентность
(комплексный вейвлет Морле) между рядом закачки нагнетательной i и рядом
жидкости добывающей j. Мера связности c_ij — средняя когерентность в полосе
периодов 3-12 месяцев внутри конуса влияния (данные месячные). Значимость —
перестановочный тест: циклические сдвиги ряда закачки (сохраняют
автокорреляцию), p = доля перестановок с когерентностью не ниже наблюдённой.

Окно пары: пересечение периодов работы скважин до среза 2015-05 включительно
(тот же срез, что у results/crm_gains_201505.csv). Пары с < 36 общих месяцев
помечаются ненадёжными и в маску не входят.

Выходы:
- results/wavelet_coherence.csv — длинный формат: пара, c_ij, p, окно, блоки
  (полная матрица восстанавливается pivot'ом injector x producer);
- results/wavelet_mask.csv — бинарная маска значимых пар
  (добывающие x нагнетательные, формат как crm_gains_*.csv);
- results/wavelet_heatmap.png — карта когерентности, сортировка по блокам.

Запуск: uv run --with pycwt python scripts/wavelet_connectivity.py
        [--nperm 150] [--alpha 0.05] [--crm-test]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt
from scipy.signal import detrend
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from timesoil.data import injection_matrix, load_monthly, producer_matrices
from timesoil.wells import INJECTORS, PRODUCERS, WELL_BLOCK

OUT = Path(__file__).resolve().parents[1] / "results"
CUTOFF = pd.Timestamp("2015-05-01")
BAND = (3.0, 12.0)  # полоса периодов, месяцы
MIN_RELIABLE = 36  # минимум общих месяцев для надёжной оценки
MIN_COMPUTE = 24  # ниже — c_ij не считаем вовсе
BLOCK_ORDER = ("A", "B", "B2", "C", "D", "E")
# параметры вейвлет-разложения: шаг 1 мес, масштабы 2..16 мес (12 суб-октав)
WCT_KW = dict(dt=1.0, dj=1 / 12, s0=2.0, J=36, sig=False, normalize=True)


def band_coherence(x: np.ndarray, y: np.ndarray) -> float:
    """Средняя вейвлет-когерентность x<->y в полосе BAND внутри конуса влияния."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wct, _, coi, freq, _ = pycwt.wct(x, y, **WCT_KW)
    period = 1.0 / freq
    in_band = (period >= BAND[0]) & (period <= BAND[1])
    valid = in_band[:, None] & (period[:, None] <= coi[None, :])
    vals = wct[valid]
    return float(vals.mean()) if vals.size else np.nan


def pair_stats(
    inj_series: pd.Series, liq_series: pd.Series, nperm: int, rng: np.random.Generator
) -> tuple[float, float, int]:
    """(c_ij, p-value, длина общего окна) для одной пары нагнетательная-добывающая."""
    inj_start = inj_series[inj_series > 0].index.min()
    prod_start = liq_series.first_valid_index()
    if pd.isna(inj_start) or pd.isna(prod_start):
        return np.nan, np.nan, 0
    start = max(inj_start, prod_start)
    x = inj_series.loc[start:CUTOFF]
    y = liq_series.loc[start:CUTOFF].interpolate(limit_direction="both")
    n = len(x)
    if n < MIN_COMPUTE:
        return np.nan, np.nan, n
    x = detrend(x.to_numpy(float))
    y = detrend(y.to_numpy(float))
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan, np.nan, n
    obs = band_coherence(x, y)
    if not np.isfinite(obs):
        return np.nan, np.nan, n
    # нулевая гипотеза: связи нет; циклический сдвиг закачки сохраняет её
    # автокорреляцию, но разрушает синхронность с добычей
    exceed = 0
    for _ in range(nperm):
        k = int(rng.integers(3, n - 3))
        c = band_coherence(np.roll(x, k), y)
        if np.isfinite(c) and c >= obs:
            exceed += 1
    p = (1 + exceed) / (1 + nperm)
    return obs, p, n


def compute_all(nperm: int) -> pd.DataFrame:
    df = load_monthly()
    liq = producer_matrices(df)["liq_tpd"]
    inj = injection_matrix(df)
    rows = []
    for i in sorted(INJECTORS):
        for p in PRODUCERS:
            rng = np.random.default_rng([42, i, p])
            c, pval, n = pair_stats(inj[i], liq[p], nperm, rng)
            rows.append(
                dict(
                    injector=i,
                    producer=p,
                    block_inj=WELL_BLOCK[i],
                    block_prod=WELL_BLOCK[p],
                    same_block=WELL_BLOCK[i] == WELL_BLOCK[p],
                    n_months=n,
                    reliable=n >= MIN_RELIABLE,
                    coherence=c,
                    p_value=pval,
                )
            )
        print(f"нагнетательная {i}: готово")
    return pd.DataFrame(rows)


def analyze(res: pd.DataFrame, alpha: float) -> pd.DataFrame:
    ok = res.dropna(subset=["coherence"]).query("reliable")
    ok = ok.assign(significant=ok["p_value"] < alpha)
    within = ok[ok["same_block"]]
    across = ok[~ok["same_block"]]

    print("\n=== Вейвлеты против блоков ===")
    print(f"надёжных пар: {len(ok)} из {len(res)} "
          f"(ненадёжных/непосчитанных: {len(res) - len(ok)})")
    print(f"средняя c_ij внутри блоков:  {within['coherence'].mean():.3f} "
          f"(n={len(within)})")
    print(f"средняя c_ij через разломы:  {across['coherence'].mean():.3f} "
          f"(n={len(across)})")
    u = mannwhitneyu(within["coherence"], across["coherence"], alternative="greater")
    print(f"Манн-Уитни (внутри > поперёк): p = {u.pvalue:.2e}")
    print(f"значимых (p<{alpha}) внутри блоков: {within['significant'].sum()}"
          f"/{len(within)} = {within['significant'].mean():.1%}")
    print(f"значимых (p<{alpha}) через разломы: {across['significant'].sum()}"
          f"/{len(across)} = {across['significant'].mean():.1%}")
    n_sig = ok["significant"].sum()
    if n_sig:
        conc = within["significant"].sum() / n_sig
        print(f"доля внутриблочных среди значимых: {conc:.1%} "
              f"(доля внутриблочных среди всех пар: {ok['same_block'].mean():.1%})")
    tot = ok["coherence"].sum()
    print(f"доля суммарной когерентности внутри блоков: "
          f"{within['coherence'].sum() / tot:.1%} "
          f"(у полевого CRM было 66.7% при 21.4% пар)")

    print("\n=== Корреляция с CRM-связностями ===")
    for name, path in [
        ("полевой CRM (без ограничений)", OUT / "crm_fullfield_gains.csv"),
        ("блочный CRM (срез 2015-05)", OUT / "crm_gains_201505.csv"),
    ]:
        if not path.exists():
            print(f"{name}: файл {path.name} не найден, пропуск")
            continue
        g = pd.read_csv(path, index_col=0)
        g.columns = g.columns.astype(int)
        f_ij = ok.apply(lambda r: g.at[r["producer"], r["injector"]], axis=1)
        sub = ok[(f_ij.notna())]
        f = f_ij.dropna().to_numpy()
        c = sub["coherence"].to_numpy()
        pr, sr = pearsonr(c, f), spearmanr(c, f)
        print(f"{name}: Пирсон r={pr[0]:.3f} (p={pr[1]:.1e}), "
              f"Спирмен rho={sr[0]:.3f} (p={sr[1]:.1e}), n={len(c)}")
        active = f > 1e-6
        if active.any() and (~active).any():
            print(f"  средняя c_ij при f_ij>0: {c[active].mean():.3f}; "
                  f"при f_ij=0: {c[~active].mean():.3f}")

    print("\n=== Потенциал маски для блочного CRM ===")
    intra = ok[ok["same_block"]]
    insig = intra[~intra["significant"]]
    print(f"внутриблочных надёжных пар: {len(intra)}; "
          f"незначимых из них: {len(insig)} ({len(insig) / len(intra):.1%})")
    gpath = OUT / "crm_gains_201505.csv"
    if gpath.exists():
        g = pd.read_csv(gpath, index_col=0)
        g.columns = g.columns.astype(int)
        gains = insig.apply(lambda r: g.at[r["producer"], r["injector"]], axis=1)
        nz = insig[gains > 1e-6].assign(crm_gain=gains[gains > 1e-6])
        print(f"из них с ненулевой связностью в блочном CRM: {len(nz)} пар, "
              f"суммарный f_ij = {nz['crm_gain'].sum():.3f} "
              f"(вся матрица: {g.to_numpy().sum():.3f})")
        if len(nz):
            top = nz.nlargest(min(8, len(nz)), "crm_gain")
            print("кандидаты на зануление (нагн.-доб., f_ij, c_ij, p):")
            for _, r in top.iterrows():
                print(f"  {r['injector']:>3}->{r['producer']:<3} блок {r['block_inj']}: "
                      f"f={r['crm_gain']:.3f}, c={r['coherence']:.3f}, p={r['p_value']:.2f}")
    return ok


def common_signal_diagnostics() -> None:
    """Когерентность нагн-нагн и доб-доб: оценка общего полевого сигнала.

    Высокая когерентность закачки между блоками означает, что бивариантная
    (не частная) когерентность закачка-добыча завышает связность через
    разломы: добывающая, реагирующая на «свою» нагнетательную, когерентна
    и с чужими, у которых похож спектр графика закачки.
    """
    import itertools

    df = load_monthly()
    inj = injection_matrix(df).loc["2008-07":CUTOFF]
    liq = producer_matrices(df)["liq_tpd"].loc[:CUTOFF]

    def cross(cols: list[int], mat: pd.DataFrame) -> tuple[float, float]:
        within, across = [], []
        for a, b in itertools.combinations(cols, 2):
            idx = mat[a].dropna().index.intersection(mat[b].dropna().index)
            if len(idx) < MIN_RELIABLE:
                continue
            x = detrend(mat.loc[idx, a].to_numpy(float))
            y = detrend(mat.loc[idx, b].to_numpy(float))
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            (within if WELL_BLOCK[a] == WELL_BLOCK[b] else across).append(
                band_coherence(x, y)
            )
        return float(np.mean(within)), float(np.mean(across))

    print("\n=== Диагностика общего сигнала (та же полоса 3-12 мес) ===")
    w, a = cross(sorted(INJECTORS), inj)
    print(f"нагн-нагн: внутри блока {w:.3f}, поперёк {a:.3f} "
          "(поперёк ~ внутри -> общий спектр графиков закачки завышает c_ij через разломы)")
    w, a = cross(list(PRODUCERS), liq)
    print(f"доб-доб:   внутри блока {w:.3f}, поперёк {a:.3f}")


def crm_mask_test(res: pd.DataFrame, alpha: float) -> None:
    """Блочный CRM без нагнетательных, незначимых для всех добывающих блока.

    Точечное per-pair зануление в pywaterflood недоступно; проверяем более
    грубый вариант — исключение целых нагнетательных из блока. Сравнение —
    WAPE жидкости на контроле 2015-06..2015-11 (6 мес после среза).
    """
    from timesoil.crm import FULL_START, fit_block, predict_block
    from timesoil.wells import block_wells

    df = load_monthly()
    liq = producer_matrices(df)["liq_tpd"]
    inj = injection_matrix(df)
    end = pd.Timestamp("2015-11-01")
    sig = {
        (r["injector"], r["producer"])
        for _, r in res.iterrows()
        if r["reliable"] and np.isfinite(r["p_value"]) and r["p_value"] < alpha
    }

    def run(block: str, injectors: list[int]) -> pd.DataFrame:
        prods = [w for w in block_wells(block, injectors=False) if w in liq.columns]
        model = fit_block(liq, inj, prods, injectors, CUTOFF)
        return predict_block(model, inj.loc[FULL_START:end, injectors], prods)

    print("\n=== Блочный CRM с вейвлет-маской (исключение нагнетательных) ===")
    err_base = err_mask = true_sum = 0.0
    changed = False
    for b in BLOCK_ORDER:
        injs = block_wells(b, injectors=True)
        prods = [w for w in block_wells(b, injectors=False) if w in liq.columns]
        keep = [i for i in injs if any((i, p) in sig for p in prods)]
        if not keep:
            keep = injs  # блок без значимых связей — оставляем как есть
        note = "без изменений" if keep == injs else f"исключены {sorted(set(injs) - set(keep))}"
        print(f"блок {b}: нагнетательных {len(injs)} -> {len(keep)} ({note})")
        pred_b = run(b, injs)
        pred_m = pred_b if keep == injs else run(b, keep)
        changed |= keep != injs
        test = liq.loc[CUTOFF + pd.offsets.MonthBegin(1):end, prods]
        err_base += (pred_b.loc[test.index] - test).abs().to_numpy().sum()
        err_mask += (pred_m.loc[test.index] - test).abs().to_numpy().sum()
        true_sum += test.to_numpy().sum()
    if not changed:
        print("маска не исключает ни одной нагнетательной целиком — прогон совпадает с базовым")
    print(f"WAPE жидкости 2015-06..2015-11: базовый блочный CRM {err_base / true_sum:.4f}, "
          f"с вейвлет-маской {err_mask / true_sum:.4f}")


def heatmap(res: pd.DataFrame, alpha: float) -> None:
    order_p = sorted(PRODUCERS, key=lambda w: (BLOCK_ORDER.index(WELL_BLOCK[w]), w))
    order_i = sorted(INJECTORS, key=lambda w: (BLOCK_ORDER.index(WELL_BLOCK[w]), w))
    c = res.pivot(index="producer", columns="injector", values="coherence")
    pv = res.pivot(index="producer", columns="injector", values="p_value")
    c = c.loc[order_p, order_i]
    pv = pv.loc[order_p, order_i]

    fig, ax = plt.subplots(figsize=(7.5, 12), dpi=150)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(c.to_numpy(), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    # значимые пары — точка в ячейке
    ys, xs = np.where(pv.to_numpy() < alpha)
    ax.scatter(xs, ys, s=9, c="#1a1a1a", marker="o", linewidths=0)
    # границы блоков
    for wells_axis, axis in ((order_p, "y"), (order_i, "x")):
        blocks = [WELL_BLOCK[w] for w in wells_axis]
        for k in range(1, len(blocks)):
            if blocks[k] != blocks[k - 1]:
                if axis == "y":
                    ax.axhline(k - 0.5, color="#555555", lw=1.0)
                else:
                    ax.axvline(k - 0.5, color="#555555", lw=1.0)
    ax.set_xticks(range(len(order_i)))
    ax.set_xticklabels([f"{w}\n{WELL_BLOCK[w]}" for w in order_i], fontsize=7)
    ax.set_yticks(range(len(order_p)))
    ax.set_yticklabels([f"{w} ({WELL_BLOCK[w]})" for w in order_p], fontsize=7)
    ax.set_xlabel("нагнетательные (по блокам)")
    ax.set_ylabel("добывающие (по блокам)")
    ax.set_title(
        "Вейвлет-когерентность закачка-жидкость, полоса 3-12 мес\n"
        f"точка — пара значима (перестановочный тест, p<{alpha}); "
        "серый — окно < 24 мес",
        fontsize=9,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("средняя когерентность $c_{ij}$")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "wavelet_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nкарта: {OUT / 'wavelet_heatmap.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nperm", type=int, default=150, help="число перестановок")
    ap.add_argument("--alpha", type=float, default=0.05, help="порог значимости")
    ap.add_argument("--crm-test", action="store_true",
                    help="прогнать блочный CRM с исключением незначимых нагнетательных")
    ap.add_argument("--reuse", action="store_true",
                    help="не пересчитывать, взять results/wavelet_coherence.csv")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.reuse and (OUT / "wavelet_coherence.csv").exists():
        res = pd.read_csv(OUT / "wavelet_coherence.csv")
    else:
        res = compute_all(args.nperm)
        res.to_csv(OUT / "wavelet_coherence.csv", index=False)

    sig = (
        res.assign(m=(res["reliable"] & (res["p_value"] < args.alpha)).astype(int))
        .pivot(index="producer", columns="injector", values="m")
        .reindex(index=list(PRODUCERS), columns=sorted(INJECTORS))
        .fillna(0)
        .astype(int)
    )
    sig.to_csv(OUT / "wavelet_mask.csv")
    print(f"сохранено: {OUT / 'wavelet_coherence.csv'}, {OUT / 'wavelet_mask.csv'}")

    analyze(res, args.alpha)
    common_signal_diagnostics()
    heatmap(res, args.alpha)
    if args.crm_test:
        crm_mask_test(res, args.alpha)


if __name__ == "__main__":
    main()
