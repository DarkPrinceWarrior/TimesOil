"""Сводная таблица всех моделей + квантильная калибровка TiRex-2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from timesoil import metrics as M

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    rows = []
    for path in sorted(RES.glob("*.csv")):
        name = path.stem
        if name.startswith(("forward_", "spdm_run")):
            continue
        try:
            model, target = name.rsplit("_", 2)[0], "_".join(name.rsplit("_", 2)[1:])
        except ValueError:
            continue
        if target not in ("oil_tpd", "liq_tpd"):
            continue
        r = pd.read_csv(path)
        if "y_true" not in r or "y_pred" not in r:
            continue
        row = dict(
            model=model,
            target=target,
            wape=M.wape(r.y_true, r.y_pred),
            smape=M.smape(r.y_true, r.y_pred),
            rmse=M.rmse(r.y_true, r.y_pred),
            cum_err_pct=M.cum_error_pct(r.y_true.to_numpy(), r.y_pred.to_numpy()),
        )
        if "q10" in r.columns:
            row["coverage_80"] = M.coverage_80(r.y_true, r.q10, r.q90)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["target", "wape"])
    summary.to_csv(RES / "summary.csv", index=False)
    for target, g in summary.groupby("target"):
        print(f"\n===== {target} =====")
        print(g.drop(columns="target").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
