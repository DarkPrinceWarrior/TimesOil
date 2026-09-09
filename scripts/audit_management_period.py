"""Recalculate official CHDD over the complete available management period."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

from timesoil.aios.economics import CHDDEconomicsAdapter, opm_management_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    args = parser.parse_args()
    root = args.run.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_bytes).hexdigest()
    if (root / "manifest.sha256").read_text().split()[0] != manifest_hash:
        raise ValueError("OPM manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    export_bytes = (root / "canonical/manifest.json").read_bytes()
    export = json.loads(export_bytes)
    raw = (root / "canonical/chdd.csv").read_bytes()
    if (manifest["status"] != "success"
            or export["provenance"]["opm_run_manifest_sha256"] != manifest_hash
            or export["outputs"]["chdd_csv"]["sha256"] != sha256(raw).hexdigest()):
        raise ValueError("canonical export does not match the successful OPM run")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    end = max(date.fromisoformat(row["DATA"]) for row in rows)
    period = (args.start, end)
    shifted = opm_management_rows(rows, period)
    result = CHDDEconomicsAdapter().calculate(
        shifted, start_year=args.start.year, output_dir=args.output,
        charge_initial_pump=False, management_period=period,
    )
    managed = [row for row in shifted if args.start.isoformat() <= str(row["DATA"]) < end.isoformat()]
    months = sorted({str(row["DATA"]) for row in managed})
    wells = sorted({str(row["well"]) for row in managed})
    assert len(managed) == len(months) * len(wells)
    summary = {
        "schema": "timesoil.management-period-audit/v1",
        "source_run": str(root), "opm_manifest_sha256": manifest_hash,
        "export_manifest_sha256": sha256(export_bytes).hexdigest(),
        "canonical_csv_sha256": sha256(raw).hexdigest(),
        "start_inclusive": args.start.isoformat(), "end_exclusive": end.isoformat(),
        "months": len(months), "wells": len(wells), "well_months": len(managed),
        "total_chdd_m": result.total_chdd_m,
        "oil_t": sum(float(row["WOMT_Diff"]) for row in managed),
        "liquid_t": sum(float(row["WLPT_Diff"]) for row in managed),
        "injection_m3": sum(float(row["WWIT_Diff"]) for row in managed),
        "max_liquid_m3d": max(float(row["WLPR"]) for row in managed),
        "recomputed_official_economics": True, "new_opm_run": False,
        "scope": "Complete available archive period; validation-case constraints not yet supplied.",
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
