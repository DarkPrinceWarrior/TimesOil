"""A/B-проверка обучения LightGBM на нескольких режимах одной модели.

Сценарии разделяются ключом ``scenario|well``. Для каждого скользящего среза
MLForecast видит по всем сценариям только историю до cutoff; будущие цели
соседних сценариев в обучение не попадают. Основная метрика считается на
исходном сценарии ``reference`` для сопоставимости с прежними результатами.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler

from timesoil import metrics as M
from timesoil.allocation import allocate, hydro_weights
from timesoil.backtest import CUTOFFS, EXT_CUTOFFS, HORIZON
from timesoil.crm import crm_forecast
from timesoil.data import (
    injection_matrix,
    load_scenarios,
    producer_matrices,
    static_features,
    well_coords,
)
from timesoil.mlprep import combined_crm, long_frame
from timesoil.wells import PRODUCERS, WELL_BLOCK, block_wells

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
REFERENCE = "reference"
LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=0,
    verbosity=-1,
)


@dataclass
class ScenarioData:
    targets: dict[str, pd.DataFrame]
    injection: pd.DataFrame
    allocated_injection: pd.DataFrame
    block_injection: pd.DataFrame
    pressure: pd.DataFrame
    crm: pd.DataFrame


def _crm_cache_path(scenario: str, cutoff: pd.Timestamp) -> Path:
    return OUT / "scenario_crm" / f"{scenario}_{cutoff:%Y%m}.csv"


def _crm_covariates(
    scenario: str,
    liquid: pd.DataFrame,
    injection: pd.DataFrame,
    cutoffs: tuple[pd.Timestamp, ...],
    reuse: bool,
) -> pd.DataFrame:
    covariates: dict[pd.Timestamp, pd.DataFrame] = {}
    for cutoff in cutoffs:
        path = _crm_cache_path(scenario, cutoff)
        if reuse and path.exists():
            pred = pd.read_csv(
                path, index_col=0, parse_dates=True
            ).rename(columns=int)
        else:
            pred, _ = crm_forecast(liquid, injection, cutoff, HORIZON)
            path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(path)
        covariates[cutoff] = pred
    crm = combined_crm(covariates, cutoffs)
    if crm is None:
        raise RuntimeError(f"не удалось собрать CRM-ковариату для {scenario}")
    return crm


def prepare_scenarios(
    cutoffs: tuple[pd.Timestamp, ...],
    reuse_crm: bool,
) -> dict[str, ScenarioData]:
    weights = hydro_weights(static_features(), well_coords())
    prepared: dict[str, ScenarioData] = {}
    for scenario, monthly in load_scenarios().items():
        mats = producer_matrices(monthly)
        injection = injection_matrix(monthly)
        allocated = allocate(injection, weights)
        block_sum = pd.DataFrame(
            {
                well: injection[
                    block_wells(WELL_BLOCK[well], injectors=True)
                ].sum(axis=1)
                for well in PRODUCERS
            }
        )
        crm = _crm_covariates(
            scenario, mats["liq_tpd"], injection, cutoffs, reuse=reuse_crm
        )
        prepared[scenario] = ScenarioData(
            targets={
                "oil_tpd": mats["oil_tpd"],
                "liq_tpd": mats["liq_tpd"],
            },
            injection=injection,
            allocated_injection=allocated,
            block_injection=block_sum,
            pressure=mats["p_res"],
            crm=crm,
        )
    return prepared


def static_table() -> pd.DataFrame:
    static = static_features().reset_index()
    static = static[static["well"].isin(PRODUCERS)]
    return pd.DataFrame(
        {
            "well_id": static["well"].astype(int),
            "perm": static["perm_md"],
            "poro": static["poro"],
            "h_eff": static["h_eff"],
            "block": static["block"].astype("category").cat.codes,
        }
    )


def pooled_frame(
    scenarios: dict[str, ScenarioData],
    target: str,
    selected: tuple[str, ...],
    with_scenario_feature: bool,
) -> tuple[pd.DataFrame, list[str]]:
    static = static_table()
    scenario_codes = {scenario: code for code, scenario in enumerate(scenarios)}
    rows: list[pd.DataFrame] = []
    for scenario in selected:
        data = scenarios[scenario]
        frame = long_frame(
            {target: data.targets[target]},
            data.allocated_injection,
            data.block_injection,
            data.crm,
            data.pressure,
        )[target]
        frame["well_id"] = frame["unique_id"].astype(int)
        frame["unique_id"] = scenario + "|" + frame["unique_id"].astype(str)
        frame = frame.merge(static, on="well_id", how="left", validate="many_to_one")
        if with_scenario_feature:
            frame["scenario_code"] = scenario_codes[scenario]
        rows.append(frame)

    pooled = pd.concat(rows, ignore_index=True)
    static_cols = ["well_id", "perm", "poro", "h_eff", "block"]
    if with_scenario_feature:
        static_cols.append("scenario_code")
    if pooled[static_cols].isna().any().any():
        raise ValueError("пропуски в статических признаках сценарного набора")
    return pooled, static_cols


def run_variant(
    scenarios: dict[str, ScenarioData],
    target: str,
    variant: str,
    selected: tuple[str, ...],
    with_scenario_feature: bool,
    reference_weight: float,
    scale_target: bool,
    cutoffs: tuple[pd.Timestamp, ...],
    step_size: int,
) -> tuple[pd.DataFrame, int]:
    frame, static_cols = pooled_frame(
        scenarios, target, selected, with_scenario_feature
    )
    model = MLForecast(
        models={"lgbm": lgb.LGBMRegressor(**LGB_PARAMS)},
        freq="MS",
        lags=[1, 2, 3, 4, 5, 6, 12],
        lag_transforms={1: [RollingMean(3), RollingMean(6)]},
        target_transforms=[LocalStandardScaler()] if scale_target else None,
    )
    weight_col = None
    if reference_weight != 1.0:
        frame["sample_weight"] = np.where(
            frame["unique_id"].str.startswith(f"{REFERENCE}|"),
            reference_weight,
            1.0,
        )
        weight_col = "sample_weight"
    cv = model.cross_validation(
        frame,
        n_windows=len(cutoffs),
        h=HORIZON,
        step_size=step_size,
        refit=True,
        static_features=static_cols,
        weight_col=weight_col,
    ).dropna(subset=["y"])
    actual_cutoffs = tuple(pd.Timestamp(c) for c in sorted(cv["cutoff"].unique()))
    if actual_cutoffs != cutoffs:
        raise RuntimeError(
            f"MLForecast построил срезы {actual_cutoffs}, ожидались {cutoffs}"
        )

    ids = cv["unique_id"].str.split("|", n=1, expand=True)
    result = pd.DataFrame(
        {
            "variant": variant,
            "scenario": ids[0],
            "cutoff": cv["cutoff"],
            "well": ids[1].astype(int),
            "date": cv["ds"],
            "y_true": cv["y"].astype(float),
            "y_pred": np.maximum(cv["lgbm"].astype(float), 0.0),
        }
    )
    result["step"] = (
        result.groupby(["scenario", "cutoff", "well"]).cumcount() + 1
    )
    return result, len(frame)


def nearest_scenario(
    scenarios: dict[str, ScenarioData],
    target: str,
    through: pd.Timestamp,
) -> tuple[str, dict[str, float]]:
    """Ближайший режим по WAPE-расстоянию только на доступной ранней истории."""
    reference = scenarios[REFERENCE].targets[target].loc[:through].to_numpy()
    distances = {}
    for scenario, data in scenarios.items():
        if scenario == REFERENCE:
            continue
        candidate = data.targets[target].loc[:through].to_numpy()
        distances[scenario] = float(
            np.abs(reference - candidate).sum()
            / max(np.abs(reference).sum(), 1e-12)
        )
    return min(distances, key=lambda scenario: distances[scenario]), distances


def metrics_rows(
    result: pd.DataFrame,
    target: str,
    variant: str,
    training_rows: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, data in result.groupby("scenario"):
        rows.append(
            {
                "target": target,
                "variant": variant,
                "scenario": scenario,
                "training_rows": training_rows,
                "n_predictions": len(data),
                "wape": M.wape(data["y_true"], data["y_pred"]),
                "smape": M.smape(data["y_true"], data["y_pred"]),
                "rmse": M.rmse(data["y_true"], data["y_pred"]),
                "cum_err_pct": M.cum_error_pct(
                    data["y_true"].to_numpy(), data["y_pred"].to_numpy()
                ),
            }
        )
    return rows


def scenario_profile(scenarios: dict[str, ScenarioData]) -> pd.DataFrame:
    rows = []
    for scenario, data in scenarios.items():
        index = data.targets["liq_tpd"].index
        rows.append(
            {
                "scenario": scenario,
                "start": index.min().date(),
                "end": index.max().date(),
                "months": len(index),
                "producer_series": data.targets["liq_tpd"].shape[1],
                "mean_field_liq_tpd": data.targets["liq_tpd"].sum(axis=1).mean(),
                "mean_field_oil_tpd": data.targets["oil_tpd"].sum(axis=1).mean(),
                "mean_field_inj_m3pd": data.injection.sum(axis=1).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--ext",
        action="store_true",
        help="14 срезов с шагом 2 месяца вместо трёх канонических",
    )
    parser.add_argument(
        "--reuse-crm",
        action="store_true",
        help="переиспользовать сохранённые сценарные CRM-ковариаты",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["oil_tpd", "liq_tpd"],
        choices=["oil_tpd", "liq_tpd"],
    )
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    cutoffs = EXT_CUTOFFS if args.ext else CUTOFFS
    step_size = 2 if args.ext else HORIZON
    scenarios = prepare_scenarios(cutoffs, reuse_crm=args.reuse_crm)
    names = tuple(scenarios)
    if names[0] != REFERENCE:
        raise RuntimeError(f"первым сценарием должен быть {REFERENCE!r}: {names}")

    profile = scenario_profile(scenarios)
    prefix = "ext_" if args.ext else ""
    profile.to_csv(OUT / f"{prefix}scenario_data_profile.csv", index=False)
    print("=== Профиль сценариев ===")
    print(profile.round(2).to_string(index=False), flush=True)

    summaries: list[dict[str, object]] = []
    for target in args.targets:
        nearest, distances = nearest_scenario(
            scenarios, target, through=cutoffs[0]
        )
        print(
            f"\n{target}: расстояния до reference на истории <= "
            f"{cutoffs[0].date()}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in distances.items())
            + f"; ближайший={nearest}",
            flush=True,
        )
        variants = (
            ("reference_only", (REFERENCE,), False, 1.0, False),
            ("reference_scaled", (REFERENCE,), False, 1.0, True),
            ("nearest", (REFERENCE, nearest), False, 1.0, False),
            ("nearest_scaled", (REFERENCE, nearest), False, 1.0, True),
            ("pooled", names, False, 1.0, False),
            ("pooled_weighted", names, False, 3.0, False),
            ("pooled_scaled", names, False, 1.0, True),
            ("pooled_weighted_scaled", names, False, 3.0, True),
            ("pooled_with_id", names, True, 1.0, False),
        )
        for (
            variant,
            selected,
            with_scenario_feature,
            reference_weight,
            scale_target,
        ) in variants:
            result, training_rows = run_variant(
                scenarios,
                target,
                variant,
                selected,
                with_scenario_feature,
                reference_weight,
                scale_target,
                cutoffs,
                step_size,
            )
            result.to_csv(
                OUT / f"{prefix}scenario_lgbm_{variant}_{target}.csv",
                index=False,
            )
            summaries.extend(
                metrics_rows(result, target, variant, training_rows)
            )
            reference = result[result["scenario"] == REFERENCE]
            print(
                f"{target:8s} {variant:14s} "
                f"rows={training_rows:5d} "
                f"WAPE(reference)={M.wape(reference.y_true, reference.y_pred):.4f}",
                flush=True,
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(
        OUT / f"{prefix}scenario_augmentation_summary.csv", index=False
    )
    print("\n=== Итог ===")
    print(
        summary[summary["scenario"] == REFERENCE]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
