"""Run a bounded Model Z control experiment through the production full cycle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compare_track2_cycles import compare
from timesoil.aios.workflow import CycleRequest


def scaled_request(original, producer, injector):
    request = deepcopy(original)
    for action in request['controls']:
        if action['status'] == 'OPEN':
            factor = injector if action['role'] == 'injector' else producer
            action['value'] *= factor
            if action['target'] == 'LRAT':
                action['value'] = min(500.0, action['value'])
    request['context'] = {
        'track': 2,
        'objective': 'Verify a numerical control-search hypothesis using full-period OPM and official CHDD. Audit numerical validity; do not claim surrogate certification or improvement before paired comparison.',
        'facts': {'is_baseline': False, 'schedule_kind': 'bounded_physical_search',
                  'surrogate_used_for_candidate_selection': False,
                  'optimization_improvement_claimed': False},
        'relative_to_incumbent': {'producer_scale': producer, 'injector_scale': injector},
    }
    return request


def self_check():
    original = {'controls': [dict(status='OPEN', role='producer', target='LRAT', value=400),
                             dict(status='SHUT', role='injector', target='WRAT', value=0),
                             dict(status='OPEN', role='injector', target='WRAT', value=100)]}
    result = scaled_request(original, 2, .5)
    assert [a['value'] for a in result['controls']] == [500, 0, 50]
    assert original['controls'][0]['value'] == 400
    print('Bounded rates, shut status and source preservation checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('incumbent_request', type=Path)
    parser.add_argument('baseline_run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    self_check()
    original = json.loads(args.incumbent_request.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    # Fixed bounded design; all factors are relative to the supplied incumbent.
    factors = [(1, .5), (1, .8), (1, 1.5), (1, 2), (2, 1), (2, 1.5), (2, 2), (.7, 1)]
    requests = []
    for index, (producer, injector) in enumerate(factors):
        request = scaled_request(original, producer, injector)
        request['scenario_id'] = f'physical-sweep-{index:02d}'
        CycleRequest.from_mapping(request)
        path = args.output / f'request-{index:02d}.json'
        path.write_text(json.dumps(request, indent=2) + '\n')
        requests.append(path)

    def run(index):
        run_id = f'candidate-{index:02d}'
        command = [sys.executable, '-m', 'timesoil.aios.cli', 'full-cycle', str(requests[index]),
                   '--runs-dir', str(args.output / 'cycles'), '--run-id', run_id, '--timeout', '7200']
        affinity = '14-29' if index % 2 == 0 else '32-47'
        env = {**os.environ, 'OPM_MPI_PROCESSES': '16', 'OPM_THREADS_PER_PROCESS': '1',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'OPM_CPU_AFFINITY': affinity}
        started = time.monotonic()
        with (args.output / f'{run_id}.log').open('x') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result = {'index': index, 'factors': factors[index], 'returncode': code,
                  'seconds': time.monotonic() - started}
        if code in (0, 2) and (args.output / 'cycles' / run_id / 'full-cycle-receipt.json').is_file():
            try:
                result['comparison'] = compare(args.baseline_run, args.output / 'cycles' / run_id,
                                               original['horizon_months'])
            except Exception as error:
                result['comparison_error'] = str(error)
        (args.output / f'{run_id}-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(requests), 2):
            results.extend(pool.map(run, (start, start + 1)))
            (args.output / 'completion.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
