"""Export OPM SUMMARY CSV to official CHDD and Track 2 contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from timesoil.aios.opm_chdd import OpmChddError, export_opm_chdd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--opm-run-manifest", type=Path, required=True)
    parser.add_argument("--summary-extraction-manifest", type=Path, required=True)
    parser.add_argument("--deck-dir", type=Path)
    parser.add_argument(
        "--density-map",
        type=Path,
        help="JSON: {well: {oil_kg_m3: number, water_kg_m3: number}}",
    )
    parser.add_argument("--unit-system", choices=("METRIC", "FIELD"))
    parser.add_argument("--chdd-output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        export_opm_chdd(
            args.summary_csv,
            args.chdd_output,
            args.trajectory_output,
            args.manifest,
            scenario_id=args.scenario_id,
            source_model=args.source_model,
            opm_run_manifest=args.opm_run_manifest,
            summary_extraction_manifest=args.summary_extraction_manifest,
            deck_dir=args.deck_dir,
            density_map=args.density_map,
            unit_system=args.unit_system,
        )
    except (OpmChddError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
