"""New proposal after source-date/BHP preflight fixes; preserve failed prior run."""
import json
from hashlib import sha256
import sys

from timesoil.aios.opm import OPM_IMAGE
import run_z_transition_policy as driver


def run(*, conversion_search=False, output_name=None):
    driver.OUT = driver.R / (output_name or ('timesfm-conversion-z-policy-20260910' if conversion_search
                                           else 'timesfm-guarded-z-policy-20260910'))
    driver.OUT.mkdir(exist_ok=False)
    records = []
    for relative in ('timesfm-bhp-policy-20260909/cycles/baseline',
                     'physical-sweep-z-20260909/cycles/candidate-03'):
        root = driver.R / relative
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['status'] == 'success' and manifest['image_reference'] == OPM_IMAGE
        grid = root / 'input/Model_Z/Model_Z_grid.inc'
        assert sha256(grid.read_bytes()).hexdigest() == 'c2bc700bd6cba9ea6dceba3349618a1d7098cd91abc96bed0d57d84c6d3fb708'
        records.append(dict(run=str(root), manifest_sha256=sha256((root / 'manifest.json').read_bytes()).hexdigest()))
    (driver.OUT / 'physics-preflight.json').write_text(json.dumps(dict(
        image_reference=OPM_IMAGE, original_PINCHREG_preserved=True,
        experimental_PINCH_data_used=False, new_proposal_required=True,
        source_schedule_checked_before_forecast=True, receipts=records), indent=2) + '\n')
    try:
        driver.run(conversion_search=conversion_search)
    except Exception:
        (driver.OUT / 'exit').write_text('1\n')
        raise


if __name__ == '__main__':
    driver.self_check()
    if '--self-check' not in sys.argv:
        run(conversion_search='--conversion-search' in sys.argv)
