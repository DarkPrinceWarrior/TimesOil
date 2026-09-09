"""Compress authenticated OPM INIT/EGRID conductances into a well connectivity prior."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from compare_track2_cycles import verify
from timesoil.aios.opm import OPM_IMAGE
from timesoil.aios.opm_chdd import (
    _expanded_record, _keyword_offsets, _read_deck_densities_and_start, _single_record,
)


def array(text, keyword):
    value = np.asarray(_expanded_record(_single_record(text, keyword), keyword), dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f'non-finite {keyword}')
    return value


def resistance_graph(active, shape, faces, nnc):
    """Use exported conductances, including parallel and non-neighbor connections."""
    nz, ny, nx = shape
    if active.shape != (nz * ny * nx,) or len(faces) != 3:
        raise ValueError('invalid grid dimensions')
    index = np.arange(len(active)).reshape(shape)
    left, right, values = [], [], []
    for axis, conductance in zip((2, 1, 0), faces, strict=True):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = index[tuple(lower)].ravel(), index[tuple(upper)].ravel()
        if conductance.shape != active.shape:
            raise ValueError('face array size differs from the Cartesian grid')
        left.extend(a); right.extend(b); values.extend(conductance[a])
    a, b, conductance = nnc
    if not (len(a) == len(b) == len(conductance)):
        raise ValueError('NNC index and conductance sizes differ')
    left.extend(a); right.extend(b); values.extend(conductance)
    left, right, values = np.asarray(left, int), np.asarray(right, int), np.asarray(values, float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (left < 0).any() or (right < 0).any()
            or (left >= len(active)).any() or (right >= len(active)).any()):
        raise ValueError('invalid exported conductance or cell index')
    keep = active[left] & active[right] & (values > 0) & (left != right)
    left, right, values = left[keep], right[keep], values[keep]
    graph = coo_matrix((np.r_[values, values], (np.r_[left, right], np.r_[right, left])),
                       shape=(len(active), len(active))).tocsr()
    graph.sum_duplicates()
    graph.data = 1 / graph.data
    return graph


def self_check():
    active = np.ones(3, dtype=bool)
    zero = np.zeros(3)
    graph = resistance_graph(active, (1, 1, 3), (np.array([2., 0, 0]), zero, zero), ([], [], []))
    distance = dijkstra(graph, directed=False, indices=0)
    np.testing.assert_allclose(distance[:2], [0, .5]); assert np.isinf(distance[2])
    graph = resistance_graph(active, (1, 1, 3), (np.array([2., 0, 0]), zero, zero), ([1], [2], [4]))
    np.testing.assert_allclose(dijkstra(graph, directed=False, indices=0), [0, .5, .75])
    print('Zero-conductance barrier and NNC connection checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    self_check()
    root = args.run.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'success' or manifest['returncode'] != 0 or manifest['image_reference'] != OPM_IMAGE:
        raise ValueError('requires a completed pinned-image OPM run')
    inputs = [e for e in manifest['artifacts'] if e['path'].startswith('input/')]
    for entry in inputs:
        verify(root, entry)
    args.output.mkdir(parents=True, exist_ok=False)
    exported, sources = {}, []
    for name, suffix in [('init', '.INIT'), ('grid', '.EGRID')]:
        entries = [e for e in manifest['artifacts'] if Path(e['path']).suffix.upper() == suffix]
        if len(entries) != 1:
            raise ValueError(f'exactly one authenticated {suffix} is required')
        entry = entries[0]; verify(root, entry); sources.append(entry)
        output = args.output / f'{name}.inc'
        command = ['docker', 'run', '--rm', '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
                   '--mount', f'type=bind,src={root},dst=/run,readonly',
                   '--mount', f'type=bind,src={args.output.resolve()},dst=/export', OPM_IMAGE,
                   'convertECL', '-g', '-o', f'/export/{name}.inc', f"/run/{entry['path']}"]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError('convertECL did not write its output')
        exported[name] = output.read_text()
    init, grid = exported['init'], exported['grid']
    for keyword in ('LGR', 'LGRHEADI', 'NNCG', 'NNCL', 'NNA1', 'NNA2'):
        if _keyword_offsets(init + grid, keyword):
            raise ValueError(f'local-grid refinement {keyword} requires an explicit grid mapping')
    dimensions = array(init, 'INTEHEAD')[8:11]
    if not np.equal(dimensions, dimensions.astype(int)).all() or (dimensions <= 0).any():
        raise ValueError('invalid INIT dimensions')
    nx, ny, nz = map(int, dimensions)
    active_values = array(grid, 'ACTNUM')
    if not np.isin(active_values, (0, 1)).all():
        raise ValueError('unsupported ACTNUM values')
    active = active_values.astype(bool)

    def global_array(keyword):
        value = array(init, keyword)
        if len(value) == len(active):
            return value
        if len(value) != active.sum():
            raise ValueError(f'{keyword} has neither global nor active-cell size')
        full = np.zeros(len(active)); full[active] = value
        return full

    nnc = ([], [], [])
    present = [_keyword_offsets(text, name) for text, name in ((grid, 'NNC1'), (grid, 'NNC2'), (init, 'TRANNNC'))]
    if any(present):
        if not all(present):
            raise ValueError('incomplete exported NNC topology')
        a, b = array(grid, 'NNC1'), array(grid, 'NNC2')
        if not np.equal(a, a.astype(int)).all() or not np.equal(b, b.astype(int)).all():
            raise ValueError('non-integral NNC cell indices')
        nnc = (a.astype(int) - 1, b.astype(int) - 1, array(init, 'TRANNNC'))
    graph = resistance_graph(active, (nz, ny, nx), [global_array(k) for k in ('TRANX', 'TRANY', 'TRANZ')], nnc)
    unit, _, _, deck_hash, connections, _, deck_text = _read_deck_densities_and_start(root / 'input')
    if unit != 'METRIC':
        raise ValueError('connectivity currently requires METRIC units')
    np.testing.assert_array_equal(array(deck_text, 'DIMENS')[:3], dimensions)
    wells = tuple(sorted(connections))
    cells, static, extended_static = [], [], []
    permx, permy, poro, dz, ntg = [global_array(k) for k in ('PERMX', 'PERMY', 'PORO', 'DZ', 'NTG')]
    properties = {name: global_array(name) for name in
                  ('PERMX', 'PERMY', 'PERMZ', 'PORO', 'NTG', 'DEPTH', 'DX', 'DY', 'DZ', 'PORV')}
    regions = {name: global_array(name) for name in
               ('FIPNUM', 'FIP_ZONE', 'FIP_C1', 'EQLNUM', 'PVTNUM', 'SATNUM')
               if _keyword_offsets(init, name)}
    region_columns = [(name, value) for name, values in regions.items()
                      for value in sorted(set(values[active]))]
    feature_names = [f'{name}_net_weighted_mean' for name in properties] + [
        'completed_net_thickness_m', 'completed_pore_volume_m3', 'completion_cell_count',
        'completion_i_mean', 'completion_j_mean', 'completion_k_mean'] + [
        f'{name}_fraction_{value:g}' for name, value in region_columns]
    for well in wells:
        coordinates = np.array([list(map(int, key.split(','))) for key in connections[well]]) - 1
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or (coordinates < 0).any() or (coordinates >= dimensions).any():
            raise ValueError(f'invalid completion cells for {well}')
        indices = np.ravel_multi_index(coordinates[:, ::-1].T, (nz, ny, nx))
        indices = np.unique(indices[active[indices]])
        net = dz[indices] * ntg[indices]
        if not len(indices) or net.sum() <= 0 or (net < 0).any():
            raise ValueError(f'no conducting completed thickness for {well}')
        cells.append(indices)
        static.append([np.average(np.sqrt(permx[indices] * permy[indices]), weights=net),
                       np.average(poro[indices], weights=net), net.sum()])
        ijk = np.array(np.unravel_index(indices, (nz, ny, nx)))[::-1].T + 1
        extended_static.append([np.average(values[indices], weights=net) for values in properties.values()]
            + [net.sum(), properties['PORV'][indices].sum(), len(indices)]
            + np.average(ijk, weights=net, axis=0).tolist()
            + [np.average(regions[name][indices] == value, weights=net) for name, value in region_columns])
    if not graph.nnz:
        raise ValueError('grid contains no conducting connections')
    # ponytail: coincident completions are resolved only to one grid edge; use exported well connection factors for wellbore-scale resistance.
    resistance_floor = float(graph.data.min())
    shared_cells = []
    weights = np.zeros((len(wells), len(wells)))
    for i, indices in enumerate(cells):
        distance = dijkstra(graph, directed=False, indices=indices, min_only=True)
        for j, target in enumerate(cells):
            if i == j:
                continue
            resistance = float(distance[target].min())
            if resistance == 0:
                shared_cells.append([wells[i], wells[j]])
            if np.isfinite(resistance):
                weights[i, j] = 1 / max(resistance, resistance_floor)
    np.testing.assert_allclose(weights, weights.T, rtol=1e-10, atol=1e-12)
    value = {'well_ids': wells, 'weights': weights.tolist(), 'static': static,
             'provenance': {'source_sha256': manifest['source_sha256'], 'expanded_deck_sha256': deck_hash,
                'static_feature_names': feature_names, 'static_features': extended_static,
                'run_manifest_sha256': sha256((root / 'manifest.json').read_bytes()).hexdigest(),
                'image': OPM_IMAGE, 'raw_artifacts': sources,
                'converted_artifacts': {name: sha256((args.output / f'{name}.inc').read_bytes()).hexdigest() for name in exported},
                'method': 'inverse shortest resistance path through OPM TRANX/TRANY/TRANZ and exported NNC',
                'global_cells': len(active), 'active_cells': int(active.sum()), 'directed_graph_edges': graph.nnz,
                'nnc_count': len(nnc[0]), 'source_faults_present': bool(_keyword_offsets(deck_text, 'FAULTS')),
                'grid_resistance_floor': resistance_floor, 'shared_completed_cell_pairs': shared_cells,
                'threshold_pressure_present': bool(_keyword_offsets(deck_text, 'THPRES')),
                'limitations': 'Static connectivity prior, not a dynamic flow model: shortest paths omit parallel-path effective conductance; coincident completions use the minimum positive grid-edge resistance floor, not measured wellbore resistance; phase mobilities and threshold-pressure activation require OPM.'}}
    (args.output / 'connectivity.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'well_count': len(wells), 'nonzero_well_links': int(np.count_nonzero(weights)), **value['provenance']}), flush=True)


if __name__ == '__main__':
    main()
