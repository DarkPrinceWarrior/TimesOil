"""Qwen field policies with full-period TimesFM forecasts and OPM requests.

Forecast margins are screening estimates, never submitted CHDD. Every emitted
request retains source completions and is evaluated
by the production full-cycle command over the complete management period.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_timesfm3 import MODEL_REVISION, forecast_inputs
from timesoil.aios.agents import AgentRole, AgentWorkflow, ToolDefinition, ToolRegistry
from timesoil.aios.llm import ExternalQwenClient, LLMConfig
from timesoil.aios.surrogate import _project_physics
from timesoil.aios.track2 import load_trajectory_dataset
from timesoil.aios.workflow import CycleError, CycleRequest, _controls
from timesoil.aios.operating_constraints import check_controls, parse_constraints
from timesoil.aios.economics import CHDDEconomicsAdapter


def policy_controls(controls, policy):
    wells = {a["well"] for a in controls}
    shut = set(policy["shut_wells"])
    scales = {}
    for item in policy["well_scales"]:
        if item["well"] in scales:
            raise ValueError("duplicate well scale")
        scales[item["well"]] = item["scale"]
    if not shut <= wells or not scales.keys() <= wells:
        raise ValueError("unknown well in policy")
    values = [policy["producer_scale"], policy["injector_scale"], *scales.values()]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("policy scales must be finite and nonnegative")
    producer_bhp = policy.get('producer_bhp_add', 0)
    injector_bhp = policy.get('injector_bhp_factor', 1)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in (producer_bhp, injector_bhp)) or producer_bhp < 0 or not 0 < injector_bhp <= 1):
        raise ValueError('BHP changes must be finite and cannot relax the reference limits')
    output = []
    for action in controls:
        item = dict(action)
        if item["well"] in shut:
            item.update(status="SHUT", value=0.0)
        elif item["status"] == "OPEN":
            factor = scales.get(item["well"], policy["injector_scale" if item["role"] == "injector" else "producer_scale"])
            item["value"] *= factor
            if item["target"] == "LRAT":
                item["value"] = min(item["value"], 500.0)
            if producer_bhp or injector_bhp != 1:
                if not item.get('bhp_limit', 0) > 0:
                    raise ValueError('global BHP changes require explicit known reference limits')
                item['bhp_limit'] = (item['bhp_limit'] * injector_bhp if item['role'] == 'injector'
                                     else item['bhp_limit'] + producer_bhp)
        output.append(item)
    by_key = {(a['month'], a['well']): a for a in output}
    months = sorted({a['month'] for a in output})
    edited = set()
    for update in policy.get('well_updates', []):
        required = {'well', 'start', 'end'}
        fields = {'role', 'status', 'target', 'value', 'bhp_limit'}
        if (not isinstance(update, dict) or not required <= update.keys()
                or update.keys() - required - fields or not fields.intersection(update)):
            raise ValueError('well update fields are invalid')
        well, start, end = update['well'], update['start'], update['end']
        if well not in wells or start not in months or end not in months or start > end:
            raise ValueError('well update is outside the known well/month grid')
        changes = {key: value for key, value in update.items() if key in fields}
        for month in months:
            if not start <= month <= end:
                continue
            key = month, well
            if key in edited:
                raise ValueError('overlapping well updates')
            edited.add(key)
            action = by_key[key]
            if changes.get('role', action['role']) != action['role']:
                if not {'target', 'value', 'bhp_limit'} <= changes.keys():
                    raise ValueError('conversion requires explicit target, rate and BHP limit')
            action.update(changes)
            if action['status'] == 'SHUT':
                action['value'] = 0.0
    _controls(output)
    roles = {}
    for action in sorted(output, key=lambda a: (a['month'], a['well'])):
        if action['target'] == 'LRAT' and action['value'] > 500:
            raise ValueError('liquid target exceeds 500 m3/day')
        if roles.get(action['well']) == 'injector' and action['role'] == 'producer':
            raise ValueError('reverse conversion is not permitted')
        roles[action['well']] = action['role']
    return output


def reject_duplicate_controls(controls_sha256, candidates):
    if any(row['controls_sha256'] == controls_sha256 for row in candidates):
        raise ValueError('policy repeats an already evaluated control schedule; propose different controls')


def self_check():
    reject_duplicate_controls('new', [{'controls_sha256': 'old'}])
    try:
        reject_duplicate_controls('same', [{'controls_sha256': 'same'}])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate physical controls accepted as a new hypothesis')
    controls = [dict(month="2007-01-01", well="P", role="producer", status="OPEN", target="LRAT", value=300),
                dict(month="2007-01-01", well="I", role="injector", status="OPEN", target="WRAT", value=100),
                dict(month="2007-01-01", well="F", role="producer", status="SHUT", target="LRAT", value=0)]
    policy = dict(producer_scale=2, injector_scale=.8, shut_wells=[], well_scales=[])
    result = policy_controls(controls, policy)
    assert [r["value"] for r in result] == [500, 80, 0] and result[-1]["status"] == "SHUT"
    assert controls[0]["value"] == 300
    pressured = [dict(a, bhp_limit=300 if a['role'] == 'injector' else 50) for a in controls]
    changed = policy_controls(pressured, {**policy, 'producer_bhp_add': 5, 'injector_bhp_factor': .9})
    assert [a['bhp_limit'] for a in changed] == [55, 270, 50]
    assert pressured[0]['bhp_limit'] == 50
    assert policy_controls(controls, {**policy, "shut_wells": ["I"]})[1]["status"] == "SHUT"
    try:
        policy_controls(controls, {**policy, "well_scales": [{"well": "unknown", "scale": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown well accepted")
    two_months = controls + [{**a, 'month': '2007-02-01'} for a in controls]
    update = dict(well='P', start='2007-01-01', end='2007-02-01',
                  role='injector', target='WRAT', value=80, bhp_limit=280)
    converted = policy_controls(two_months, {**policy, 'well_updates': [update]})
    assert all(a['role'] == 'injector' and a['value'] == 80 and a['bhp_limit'] == 280
               for a in converted if a['well'] == 'P')
    stopped = policy_controls(two_months, {**policy, 'well_updates': [
        dict(well='P', start='2007-02-01', end='2007-02-01', status='SHUT')]})
    assert stopped[0]['status'] == 'OPEN' and stopped[3]['value'] == 0
    assert two_months[3]['status'] == 'OPEN'
    for updates in ([update, update], [{**update, 'end': '2007-01-01'}],
                    [{**update, 'start': '2006-12-01'}],
                    [{k: v for k, v in update.items() if k != 'bhp_limit'}],
                    [{**update, 'value': True}], [{**update, 'bhp_limit': -1}]):
        try:
            policy_controls(two_months, {**policy, 'well_updates': updates})
        except (ValueError, CycleError):
            pass
        else:
            raise AssertionError('invalid timed control update accepted')
    print('policy rates, dated status/BHP, conversion persistence and rejection checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument('--connectivity', type=Path)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--head-sha256')
    parser.add_argument('--skip-grid', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--reference-correction', type=Path)
    parser.add_argument('--reference-correction-sha256')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    self_check()
    if not 1 <= args.rounds <= 12:
        parser.error("rounds must be in [1, 12]")
    if bool(args.head) != bool(args.head_sha256) or (args.head and not args.connectivity):
        parser.error('trained head requires its SHA-256 and connectivity')
    if bool(args.reference) != bool(args.reference_sha256) or args.reference and not args.head:
        parser.error('physical reference requires its manifest hash and trained weights')
    if (bool(args.reference_correction) != bool(args.reference_correction_sha256)
            or args.reference_correction and (not args.reference or not args.skip_grid)):
        parser.error('local correction requires reference, correction hash and --skip-grid')
    request = json.loads(args.request.read_text())
    checked_request = CycleRequest.from_mapping(request)
    normative_profile = CHDDEconomicsAdapter.from_env().normative_profile(
        charge_initial_pump=checked_request.charge_initial_pump
    )
    econ = normative_profile["assumptions"]
    raw = (args.baseline_run / "canonical/trajectory.csv").read_bytes()
    manifest = json.loads((args.baseline_run / "canonical/manifest.json").read_text())
    if sha256(raw).hexdigest() != manifest["outputs"]["track2_csv"]["sha256"]:
        raise ValueError("baseline trajectory hash mismatch")
    dataset = load_trajectory_dataset(args.baseline_run / 'canonical/trajectory.csv',
        manifest=args.baseline_run / 'canonical/manifest.json')
    if len(dataset) != 1 or not dataset.model_z_identity:
        raise ValueError('one authenticated Model Z baseline is required')
    trajectory = dataset[0]
    pressure_semantics = {
        'forecast_vector': 'WBP9', 'forecast_definition': manifest['conversion']['THP'],
        'bottom_hole_vector': 'WBHP', 'bottom_hole_definition': manifest['conversion']['BHP'],
        'control_bhp_limit': 'Planned lower producer / upper injector BHP bound; not the forecast target.',
        'inactive_zero': 'A zero WBP9 report for an inactive well is not evidence of zero physical reservoir pressure.',
    }
    start = pd.Timestamp(min(a["month"] for a in request["controls"]))
    origin = int(trajectory.dates.get_loc(start))
    horizon, context = checked_request.horizon_months, 128
    if origin < context or origin + horizon >= len(trajectory.dates):
        raise ValueError('verified history and complete forecast horizon required')
    reference = None
    if args.reference:
        reference_manifest = args.reference / 'manifest.json'
        if sha256(reference_manifest.read_bytes()).hexdigest() != args.reference_sha256:
            raise ValueError('physical reference manifest hash mismatch')
        data = load_trajectory_dataset(args.reference / 'trajectory.csv', manifest=reference_manifest)
        if (len(data) != 1 or not data.model_z_identity or data[0].well_ids != trajectory.well_ids
                or not data[0].dates.equals(trajectory.dates) or data[0].actions.shape[-1] != 4
                or json.loads(reference_manifest.read_text())['provenance']['opm_source_sha256']
                != manifest['provenance']['opm_source_sha256']):
            raise ValueError('physical reference reservoir, BHP controls or temporal grid differs')
        reference = data[0]
        np.testing.assert_allclose(reference.states[:origin + 1], trajectory.states[:origin + 1], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(reference.actions[:origin, :, :trajectory.actions.shape[-1]], trajectory.actions[:origin])
        trajectory = reference
        raw = (args.reference / 'trajectory.csv').read_bytes()
        for a in request['controls']:
            if a['status'] == 'OPEN' and a.get('bhp_limit') is None:
                a['bhp_limit'] = float(reference.actions[reference.dates.get_loc(pd.Timestamp(a['month'])),
                                                         reference.well_ids.index(a['well']), 3])
        checked_request = CycleRequest.from_mapping(request)
    has_bhp = trajectory.actions.shape[-1] == 4
    if not has_bhp and any(a.bhp_limit is not None for a in checked_request.controls):
        raise ValueError('BHP screening requires an authenticated baseline with the BHP action channel')
    connectivity = None
    if args.connectivity:
        from timesoil.aios.interwell import WellConnectivity
        connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
        if (connectivity.well_ids != trajectory.well_ids or connectivity.provenance['source_sha256']
                != manifest['provenance']['opm_source_sha256']):
            raise ValueError('geology does not match the verified reservoir and well order')
    targets, _ = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon)
    target = targets.reshape(-1, context)
    well_index = {w: i for i, w in enumerate(trajectory.well_ids)}
    initial_controls = {a["well"]: a for a in request["controls"] if a["month"] == start.date().isoformat()}
    date_index = {d.date().isoformat(): i for i, d in enumerate(trajectory.dates)}
    operating_rules = parse_constraints(
        checked_request.context.get('operating_constraints', []), wells=trajectory.well_ids,
        start=min(a.month for a in checked_request.controls),
        end=max(a.month for a in checked_request.controls))
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A100 required")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(20260909)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
        revision=MODEL_REVISION, per_core_batch_size=1, device="cuda"))
    if args.head:
        from timesfm_geology import load_frozen_model
        if sha256(args.head.read_bytes()).hexdigest() != args.head_sha256:
            raise ValueError('trained head hash mismatch')
        selected = torch.load(args.head, map_location='cuda', weights_only=True)
        forecaster.model = load_frozen_model(forecaster.model, connectivity, selected,
                                             reference_sha256=args.reference_sha256)
    correction = None
    if reference is not None:
        from benchmark_timesfm_layouts import forecast_layout
        from evaluate_timesfm_scenarios import reference_delta
        reference_prediction = forecast_layout(forecaster, reference, origin, horizon, context,
                                               'joint', connectivity=connectivity)
        reference_truth = reference.states[origin + 1:origin + horizon + 1]
    if args.reference_correction:
        from fit_timesfm_reference import bhp_features, load_reference_correction
        correction, coefficients = load_reference_correction(args.reference_correction,
            args.reference_correction_sha256, args.head_sha256, args.reference_sha256,
            horizon, len(well_index))
    candidates = []
    days = np.array([d.days_in_month for d in trajectory.dates[origin:origin + horizon]])[:, None]

    def evaluate(policy):
        policy = {"producer_scale": 1.0, "injector_scale": 1.0,
                  "shut_wells": [], "well_scales": [], **policy}
        controls = policy_controls(request["controls"], policy)
        proposed = {**request, "scenario_id": f"timesfm-policy-{len(candidates):02d}", "controls": controls, "context": {
            **request.get("context", {}),
            "objective": "Verify this experimental TimesFM-screened field policy with full OPM and official CHDD. Improvement is unknown until paired comparison on the same period.",
            "facts": {"schedule_kind": "timesfm_policy_candidate", "is_baseline": False,
                      "surrogate_used_for_candidate_selection": False,
                      "surrogate_used_only_to_propose_hypotheses": True,
                      "simulated_reference_future_used": reference is not None,
                      "independent_surrogate_uq_calibrated": False,
                      "optimization_improvement_claimed": False},
            "policy": policy,
        }}
        checked = CycleRequest.from_mapping(proposed)
        reject_duplicate_controls(checked.controls_sha256, candidates)
        original_roles = {(a['month'], a['well']): a['role'] for a in request['controls']}
        if (not checked.context.get('constraints', {}).get('allow_conversion_to_injection', False)
                and any(a['role'] != original_roles[a['month'], a['well']] for a in controls)):
            raise ValueError('new role changes are not permitted by this case')
        check_controls(operating_rules, checked.controls)
        actions = trajectory.actions.copy()
        for a in controls:
            position = date_index[a['month']], well_index[a['well']]
            value = [a['value'], {'ORAT': 0, 'LRAT': 1, 'WRAT': 2}[a['target']], a['status'] == 'OPEN']
            if has_bhp:
                value.append(a.get('bhp_limit', actions[position][3]))
            elif 'bhp_limit' in a:
                raise ValueError('BHP policy requires the BHP action channel')
            actions[position] = value
        future = actions[origin:origin + horizon]
        if correction:
            local_features = bhp_features(future, reference.actions[origin:origin + horizon], correction['degree'])
        _, cov = forecast_inputs(trajectory.states, actions, origin, context, horizon)
        cov = cov[:, :-1].reshape(-1, context + horizon)
        if connectivity is not None:
            planned = actions[origin - context:origin + horizon]
            allocated = connectivity.features(np.zeros((len(planned) * len(well_index), 3)),
                planned.reshape(-1, planned.shape[-1]))[:, 0].reshape(len(planned), len(well_index)).T
            cov = np.concatenate([cov, allocated])
        begin = time.monotonic()
        forecast = next(forecaster.predict_batch([target], horizon=horizon,
            past_future_covariates=[cov],
            use_symmetric_averaging=False, make_positive=True, return_quantiles=False))
        prediction = forecast.forecast.reshape(len(well_index), 3, horizon).transpose(2, 0, 1)
        prediction = _project_physics(prediction, future, zero_injectors=True)[0]
        if reference is not None:
            prediction = reference_delta(reference_truth, reference_prediction, prediction, future)
        if correction:
            prediction = _project_physics(prediction + np.tensordot(local_features, coefficients, axes=(0, 0)),
                                          future, zero_injectors=True)[0]
        if not np.isfinite(prediction).all():
            raise ValueError("non-finite TimesFM forecast")
        oil = float((prediction[..., 0] * days).sum())
        liquid = float((prediction[..., 1] * days).sum())
        injection = float((np.where(future[..., 1] == 2, future[..., 0] * future[..., 2], 0) * days).sum())
        record = {"id": len(candidates), "policy": policy, "lookahead_months": horizon,
            "estimated_oil_t": oil, "estimated_liquid_t": liquid, "planned_injection_m3": injection,
            "screening_margin_m": (oil * (econ["oilPriceRubT"] - econ["deductionsRubT"] - econ["oilOpexRubT"])
                                   - liquid * econ["liquidOpexRubT"] - injection * econ["injectionOpexRubM3"]) / 1e6,
            "full_period_months": checked.horizon_months, "full_period_actions": len(controls),
            "bhp_channel": has_bhp,
            "pressure_semantics": pressure_semantics,
            "trained_head_sha256": args.head_sha256,
            "physical_reference_manifest_sha256": args.reference_sha256,
            "reference_correction_sha256": args.reference_correction_sha256,
            "forecast_by_well": [{"well": w,
                "oil_tonnes": float((prediction[:, i, 0] * days[:, 0]).sum()),
                "liquid_tonnes": float((prediction[:, i, 1] * days[:, 0]).sum()),
                "terminal_reservoir_pressure_bar": float(prediction[-1, i, 2])}
                for i, w in enumerate(trajectory.well_ids)],
            "controls_sha256": checked.controls_sha256, "inference_seconds": time.monotonic() - begin,
            "is_official_chdd": False}
        candidates.append(record)
        proposed["context"]["screening"] = record
        (args.output / f"request-{record['id']:02d}.json").write_text(json.dumps(proposed, ensure_ascii=False, indent=2))
        (args.output / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
        return record

    base_policy = dict(producer_scale=1.0, injector_scale=1.0, shut_wells=[], well_scales=[])
    grid = [(1, 1)] if args.skip_grid else [(1, 1), (1.25, 1), (1, .8), (1, 1.2), (.8, 1), (1.5, 1), (2, 1), (3, 1)]
    for production, injection in grid:
        print(json.dumps(evaluate({**base_policy, "producer_scale": production, "injector_scale": injection})), flush=True)
    schema = {"type": "object", "properties": {
        "producer_scale": {"type": "number"}, "injector_scale": {"type": "number"},
        "shut_wells": {"type": "array", "items": {"type": "string"}},
        "well_scales": {"type": "array", "items": {"type": "object", "properties": {
            "well": {"type": "string"}, "scale": {"type": "number"}},
            "required": ["well", "scale"], "additionalProperties": False}}},
        "required": [], "additionalProperties": False}
    schema['properties']['well_updates'] = {'type': 'array', 'items': {
        'type': 'object', 'properties': {
            'well': {'type': 'string'}, 'start': {'type': 'string'}, 'end': {'type': 'string'},
            'role': {'type': 'string', 'enum': ['producer', 'injector']},
            'status': {'type': 'string', 'enum': ['OPEN', 'SHUT']},
            'target': {'type': 'string', 'enum': ['ORAT', 'LRAT', 'WRAT']},
            'value': {'type': 'number', 'minimum': 0},
            **({'bhp_limit': {'type': 'number', 'exclusiveMinimum': 0}} if has_bhp else {})},
        'required': ['well', 'start', 'end'], 'additionalProperties': False}}
    if has_bhp:
        schema['properties'].update(producer_bhp_add={'type': 'number', 'minimum': 0},
                                    injector_bhp_factor={'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1})
    if correction:
        schema['properties'].pop('well_updates')
        for key in ('producer_scale', 'injector_scale'):
            schema['properties'][key] = {'type': 'number', 'const': 1}
        for key in ('shut_wells', 'well_scales'):
            schema['properties'][key]['maxItems'] = 0
        schema['properties']['producer_bhp_add']['maximum'] = 15
        schema['properties']['injector_bhp_factor']['minimum'] = .9
    proposed_ids = []

    async def propose_round(index):
        before = len(candidates)
        tool = ToolDefinition("propose_policy", "Propose a full-field policy. Optional producer_bhp_add (bar) and injector_bhp_factor tighten open-well BHP limits uniformly before individual updates. well_updates changes a well over inclusive monthly start/end dates after rate scaling: rate, status, target, role, BHP limit. Conversion requires explicit WRAT, value and BHP, must be permitted by the case, and cannot be reversed; extend its role to the end. Omitted scales default to 1, arrays to empty. Respect the explicit forecast_reference domain when present. Full-period TimesFM hypothesis forecast; every retained candidate requires full-period OPM, and official CHDD selects the winner.", schema,
            lambda policy, _: evaluate(policy))
        context_value = {"track": 2, "round": index,
            "pressure_semantics": pressure_semantics,
            "surrogate_evidence": {"model": "Google TimesFM 3.0", "revision": MODEL_REVISION,
                "adapted_checkpoint_loaded": args.head is not None, "verified_checkpoint_sha256": args.head_sha256,
                "full_period_forecast_already_completed": True, "accuracy_certified": False,
                "physical_reference_used": reference is not None,
                "local_correction_loaded": correction is not None,
                "interpretation": "Checkpoint loading and forecast execution are verified. Accuracy certification is a separate, unresolved property; it does not mean the model is untrained. The unchanged baseline is already evaluated."},
            "objective": f"Propose a new policy for maximum official CHDD over the request's {checked_request.horizon_months} management months. " + ("Use only uniform BHP changes inside forecast_reference.domain. " if correction else "Use per-well multipliers when useful; all wells are controllable. ") + "Call propose_policy exactly once. Propose an unexplored hypothesis for physical verification, even when its improvement is uncertain. Existing controls, including the unchanged baseline, are rejected as duplicates. Do not return an existing best policy as a new experiment.",
            "candidates": candidates,
            "verified_well_count": len(well_index),
            "forecast_reference": {'manifest_sha256': args.reference_sha256,
                'future_is_prior_physical_planning_information': reference is not None,
                'domain': correction['domain'] if correction else None,
                'local_policy_parameters': 'producer_bhp_add in bar, injector_bhp_factor; x=add/15, y=(1-factor)/0.1; 0<=x<=1, 0<=y<=1-x/2. Rates, roles and statuses stay fixed.' if correction else None},
            "geology": None if connectivity is None else {
                "static_feature_names": connectivity.provenance.get('static_feature_names', []),
                "static_by_well": [[well, values] for well, values in zip(connectivity.well_ids,
                    connectivity.provenance.get('static_features', connectivity.static.tolist()), strict=True)],
                "five_strongest_neighbors": [[well, [[connectivity.well_ids[j], float(connectivity.weights[i, j])]
                    for j in np.argsort(-connectivity.weights[i])[:5] if connectivity.weights[i, j] > 0]]
                    for i, well in enumerate(connectivity.well_ids)],
                "all_links_used_in_forecast": True, "limitations": connectivity.provenance['limitations']},
            "economics": normative_profile,
            "field_state": [{"well": w, "oil_tpd": float(trajectory.states[origin, i, 0]),
                "liquid_tpd": float(trajectory.states[origin, i, 1]), "pressure_bar": float(trajectory.states[origin, i, 2]),
                "initial_control": initial_controls[w]}
                for i, w in enumerate(trajectory.well_ids)],
            "management_period": {'first_control_month': start.date().isoformat(),
                'last_control_month': max(a['month'] for a in request['controls'])},
            "constraints": {"producer_liquid_max_m3d": 500, "source_bhp_limits_cannot_be_relaxed": True,
                "source_completions_preserved": True, 'bhp_channel': has_bhp,
                'case_constraints': checked_request.context.get('constraints', {}),
                'operating_constraints': checked_request.context.get('operating_constraints', []),
                "additional_water_quota": "not supplied in the current training archive"},
            "claim_limits": "Forecasts use observed pre-origin history and planned controls. When forecast_reference is present, its prior simulated future is also used; candidate future observations are excluded. Screening margin is a full-period undiscounted rate-integration estimate excluding pump CAPEX, state events and tax. It is NOT CHDD and cannot select a winning control policy. Every retained hypothesis must undergo full OPM plus official CHDD before selection. Candidate requires full-period OPM plus the official calculator. Request dates describe this experiment, not a confirmed competition horizon. No independently calibrated TimesFM uncertainty or improvement claim."}
        async with ExternalQwenClient(LLMConfig.from_env()) as client:
            plan = await AgentWorkflow(client, ToolRegistry((tool,)),
                role_tools={AgentRole.PLANNER: (tool.name,)}, required_tools={AgentRole.PLANNER: (tool.name,)}).run_plan(context_value)
        (args.output / f"agent-{index:02d}.json").write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
        if not all(d.approved for d in plan.decisions) or len(candidates) != before + 1:
            raise RuntimeError("Qwen must approve exactly one proposed policy")
        proposed_ids.append(candidates[-1]["id"])
        print(json.dumps(candidates[-1]), flush=True)

    for index in range(args.rounds):
        asyncio.run(propose_round(index))
    (args.output / "proposal-receipt.json").write_text(json.dumps({
        "source_trajectory_sha256": sha256(raw).hexdigest(), "request_sha256": sha256(args.request.read_bytes()).hexdigest(),
        "timesfm_revision": MODEL_REVISION, "agent_proposal_ids": proposed_ids,
        "horizon_months": horizon, "head_sha256": args.head_sha256,
        "reference_manifest_sha256": args.reference_sha256,
        "reference_correction_sha256": args.reference_correction_sha256,
        "connectivity_sha256": sha256(args.connectivity.read_bytes()).hexdigest() if args.connectivity else None,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "requires_full_period_opm": True,
        "final_chdd_computed": False}, indent=2))


if __name__ == "__main__":
    main()
