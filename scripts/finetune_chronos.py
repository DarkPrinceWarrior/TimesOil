"""Дообучение Chronos-2 (LoRA) на панели поля и честная оценка против zero-shot.

Для каждой пары (цель, срез) модель amazon/chronos-2 дообучается методом LoRA
ТОЛЬКО на истории до среза (утечки нет: обучающие и валидационные данные не
выходят за срез; ряд CRM берётся из results/crm_cov_<YYYYMM>.csv — он посчитан
на данных до того же среза). Схема ковариат — та же, что в zero-shot варианте
cov_crm (`timesoil.chronos_runner`): закачка блока и ряд CRM — известные
наперёд, пластовое давление — только прошлое. Прогноз дообученной моделью идёт
тем же кодом `forecast_chronos(..., variant="cov_crm")`, что и zero-shot, —
сравнение честное по построению.

Валидация (по умолчанию включена): последние 6 месяцев истории до среза —
holdout; обучающие ряды обрезаются на 6 месяцев раньше среза, ранняя остановка
по eval_loss (EarlyStoppingCallback, отбор лучшего чекпойнта). --no-val — учить
на всей истории до среза без валидации.

Запуск (peft не в проектном окружении — добавляется через --with):
  смоук (CPU):  uv run --with peft python scripts/finetune_chronos.py \
                    --targets oil_tpd --cutoffs 2015-05-01 --steps 30 --eval-every 10
  канон (a100): uv run --with peft python scripts/finetune_chronos.py --steps 300

Выходы: results/chronos_lora_<цель>.csv (cutoff, well, step, date, y_true,
y_pred, q10..q90); с --with-zeroshot дополнительно пересчитывается zero-shot
тем же скриптом -> results/chronos_lora_zs_<цель>.csv; иначе база сравнения —
существующий results/chronos_cov_crm_<цель>.csv. Чекпойнты — в
results/chronos_ft/ (вне git).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.backtest import CUTOFFS, HORIZON, summarize
from timesoil.chronos_runner import forecast_chronos
from timesoil.metrics import wape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BLOCKS = ("A", "B", "B2", "C", "D", "E")


def build_train_inputs(
    target_mat: pd.DataFrame,
    train_end: pd.Timestamp,
    groups: list[list],
    group_inj: list[list],
    inj_mat: pd.DataFrame,
    pres_mat: pd.DataFrame | None,
    crm_mat: pd.DataFrame | None,
    horizon: int,
) -> list:
    """PreparedInput'ы для fit: история до train_end, схема ковариат как в cov_crm.

    Построение записей зеркалит `forecast_chronos` (та же обработка NaN, тот же
    bfill/ffill ряда CRM), только без будущих строк: известность закачки и CRM
    наперёд помечается через known_covariates_names — при обучении «будущие»
    значения нарезаются из самой истории (окна внутри Chronos2Dataset).
    """
    from chronos.chronos2.preprocess import from_data_frame

    prepared: list = []
    for gi, wells in enumerate(groups):
        blk_inj = group_inj[gi]
        recs = []
        for w in wells:
            s = target_mat.loc[:train_end, w].dropna()
            if s.empty:
                continue
            crm_w = None
            if crm_mat is not None and w in crm_mat.columns:
                crm_w = crm_mat[w].reindex(s.index).bfill().ffill()
            for t, y in s.items():
                r = {"item_id": str(w), "timestamp": t, "target": float(y)}
                for i in blk_inj:
                    r[f"inj_{i}"] = float(inj_mat.at[t, i])
                if pres_mat is not None and w in pres_mat.columns:
                    pv = pres_mat.at[t, w]
                    r["pres"] = float(pv) if pd.notna(pv) else np.nan
                if crm_w is not None:
                    r["crm"] = float(crm_w.at[t])
                recs.append(r)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        known = [c for c in df.columns if c.startswith("inj_")]
        if "crm" in df.columns:
            known.append("crm")
        prepared.extend(
            from_data_frame(
                df,
                target_columns=["target"],
                prediction_length=horizon,
                known_covariates_names=known or None,
            )
        )
    return prepared


def finetune_one(
    base_pipeline,
    target_mat: pd.DataFrame,
    cutoff: pd.Timestamp,
    groups: list[list],
    group_inj: list[list],
    inj_mat: pd.DataFrame,
    pres_mat: pd.DataFrame | None,
    crm_mat: pd.DataFrame | None,
    args: argparse.Namespace,
    tag: str,
):
    """LoRA-дообучение копии базовой модели на истории одного среза."""
    use_val = not args.no_val
    train_end = cutoff - pd.DateOffset(months=HORIZON) if use_val else cutoff
    train_inputs = build_train_inputs(
        target_mat, train_end, groups, group_inj, inj_mat, pres_mat, crm_mat, HORIZON
    )
    val_inputs = None
    if use_val:
        # полная история до среза; Chronos2Dataset(VALIDATION) сам берёт
        # последние 6 месяцев каждого ряда как валидационное окно
        val_inputs = build_train_inputs(
            target_mat, cutoff, groups, group_inj, inj_mat, pres_mat, crm_mat, HORIZON
        )

    fit_kwargs: dict = {}
    callbacks = []
    if use_val:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        fit_kwargs.update(
            eval_steps=args.eval_every,
            save_steps=args.eval_every,
            logging_steps=args.eval_every,
        )

    t0 = time.time()
    ft = base_pipeline.fit(
        inputs=train_inputs,
        prediction_length=HORIZON,
        validation_inputs=val_inputs,
        finetune_mode="lora",
        lora_config={"r": args.lora_r, "lora_alpha": args.lora_alpha,
                     "target_modules": ["self_attention.q", "self_attention.v",
                                        "self_attention.k", "self_attention.o",
                                        "output_patch_embedding.output_layer"]},
        learning_rate=args.lr,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=OUT / "chronos_ft" / tag,
        callbacks=callbacks,
        remove_printer_callback=True,
        seed=args.seed,
        **fit_kwargs,
    )
    print(f"[{tag}] дообучение: {len(train_inputs)} рядов (история <= {train_end.date()}), "
          f"{time.time() - t0:.0f} с", flush=True)
    return ft


def forecast_with_truth(
    pipeline,
    mat: pd.DataFrame,
    cutoff: pd.Timestamp,
    groups: list[list],
    group_inj: list[list],
    inj_mat: pd.DataFrame,
    pres_mat: pd.DataFrame | None,
    crm_mat: pd.DataFrame | None,
) -> pd.DataFrame:
    """Прогноз cov_crm тем же кодом, что zero-shot, + фактические значения."""
    fc = forecast_chronos(
        pipeline, mat, cutoff, HORIZON, "cov_crm",
        groups=groups, group_inj=group_inj,
        inj_mat=inj_mat, pres_mat=pres_mat, inj_future=inj_mat, crm_mat=crm_mat,
    )
    fc["cutoff"] = cutoff
    fc["y_true"] = [mat.at[d, w] for d, w in zip(fc.date, fc.well)]
    lead = ["cutoff", "well", "step", "date", "y_true", "y_pred"]
    return fc[lead + [c for c in fc.columns if c not in lead]]


def compare_table(res_lora: pd.DataFrame, res_zs: pd.DataFrame) -> pd.DataFrame:
    """WAPE per-срез и ALL: LoRA против zero-shot."""
    rows = []
    for cutoff, g in res_lora.groupby("cutoff"):
        z = res_zs[res_zs.cutoff == cutoff]
        rows.append(dict(cutoff=str(pd.Timestamp(cutoff).date()),
                         wape_lora=wape(g.y_true, g.y_pred),
                         wape_zs=wape(z.y_true, z.y_pred)))
    rows.append(dict(cutoff="ALL",
                     wape_lora=wape(res_lora.y_true, res_lora.y_pred),
                     wape_zs=wape(res_zs.y_true, res_zs.y_pred)))
    out = pd.DataFrame(rows)
    out["delta"] = out.wape_lora - out.wape_zs
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", nargs="*", default=["oil_tpd", "liq_tpd"])
    ap.add_argument("--cutoffs", nargs="*", default=[str(c.date()) for c in CUTOFFS],
                    help="срезы (YYYY-MM-DD); по умолчанию канонические 3")
    ap.add_argument("--steps", type=int, default=300, help="число шагов дообучения")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="рядов в батче (цели + ковариаты вместе)")
    ap.add_argument("--eval-every", type=int, default=50,
                    help="частота валидации/сохранения, шагов")
    ap.add_argument("--patience", type=int, default=3,
                    help="ранняя остановка: валидаций без улучшения eval_loss")
    ap.add_argument("--no-val", action="store_true",
                    help="без валидации/ранней остановки — учить на всей истории до среза")
    ap.add_argument("--with-zeroshot", action="store_true",
                    help="пересчитать zero-shot тем же скриптом (иначе база — chronos_cov_crm_*.csv)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--device", default=None, help="cpu | cuda (по умолчанию — авто)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import peft  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Нужен peft (иначе chronos молча откатится на полное дообучение). "
            "Запуск: uv run --with peft python scripts/finetune_chronos.py ..."
        )

    import torch
    from chronos import Chronos2Pipeline

    from timesoil.data import injection_matrix, load_monthly, producer_matrices
    from timesoil.wells import block_wells

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cutoffs = [pd.Timestamp(c) for c in args.cutoffs]

    df = load_monthly()
    mats = producer_matrices(df)
    inj = injection_matrix(df)
    groups = [block_wells(b, injectors=False) for b in BLOCKS]
    group_inj = [block_wells(b, injectors=True) for b in BLOCKS]
    crm_covs = {}
    for cutoff in cutoffs:
        p = OUT / f"crm_cov_{cutoff:%Y%m}.csv"
        if not p.exists():
            raise SystemExit(f"нет {p} — ряд CRM обязателен для варианта cov_crm")
        crm_covs[cutoff] = pd.read_csv(p, index_col=0, parse_dates=True).rename(columns=int)

    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    OUT.mkdir(exist_ok=True)

    for tname in args.targets:
        mat = mats[tname]
        parts_lora, parts_zs = [], []
        for cutoff in cutoffs:
            tag = f"{tname}_{cutoff:%Y%m}"
            ft = finetune_one(base, mat, cutoff, groups, group_inj, inj,
                              mats["p_res"], crm_covs[cutoff], args, tag)
            parts_lora.append(forecast_with_truth(
                ft, mat, cutoff, groups, group_inj, inj, mats["p_res"], crm_covs[cutoff]))
            del ft
            if args.with_zeroshot:
                parts_zs.append(forecast_with_truth(
                    base, mat, cutoff, groups, group_inj, inj, mats["p_res"], crm_covs[cutoff]))

        res_lora = pd.concat(parts_lora, ignore_index=True)
        res_lora.to_csv(OUT / f"chronos_lora_{tname}.csv", index=False)

        if args.with_zeroshot:
            res_zs = pd.concat(parts_zs, ignore_index=True)
            res_zs.to_csv(OUT / f"chronos_lora_zs_{tname}.csv", index=False)
            zs_src = "пересчитан этим скриптом"
        else:
            ref = OUT / f"chronos_cov_crm_{tname}.csv"
            if ref.exists():
                res_zs = pd.read_csv(ref, parse_dates=["date", "cutoff"])
                res_zs = res_zs[res_zs.cutoff.isin(cutoffs)]
                zs_src = f"из {ref.name}"
            else:
                res_zs = None
                zs_src = "нет (файл zero-shot не найден)"

        print(f"\n=== {tname} | chronos-2 LoRA (steps={args.steps}, lr={args.lr}) ===")
        print(summarize(res_lora).round(4).to_string(index=False))
        if res_zs is not None and len(res_zs):
            print(f"--- сравнение с zero-shot cov_crm ({zs_src}) ---")
            print(compare_table(res_lora, res_zs).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
