"""Feasible-regime bank: baseline arithmetic, family counts, control invariants, frozen split."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_feasible_bank as bank


START_YEAR = 2007
MONTHS = 24
PRODUCERS = tuple(str(number) for number in range(1, 20)) + ("104",)
INJECTORS = ("201", "202")
SOURCE = "/tmp/timesoil-kt2/model_z/Model_Z_final_OPM.zip"


def month_at(index: int) -> str:
    return f"{START_YEAR + index // 12:04d}-{index % 12 + 1:02d}-01"


def baseline_request() -> dict:
    controls = []
    for index in range(MONTHS):
        month = month_at(index)
        for position, well in enumerate(PRODUCERS):
            # Well "3" starts only in month 5: nothing may be produced before that.
            late = well == "3" and index < 5
            controls.append({"month": month, "well": well, "role": "producer",
                             "status": "SHUT" if late else "OPEN", "target": "LRAT",
                             "value": 0.0 if late else 300.0 + 10.0 * position,
                             "bhp_limit": 60.0})
        for well in INJECTORS:
            controls.append({"month": month, "well": well, "role": "injector", "status": "OPEN",
                             "target": "WRAT", "value": 300.0, "bhp_limit": 280.0})
    return {"context": {"constraints": {"allow_conversion_to_injection": False}},
            "controls": controls, "source": SOURCE, "deck": "Model_Z/Model_Z.data",
            "schedule_relative_path": "Model_Z/Model_Z_sch.inc", "scenario_id": "baseline",
            "source_model": "model_z_opm", "start_year": START_YEAR,
            "parsing_strictness": "low", "horizon_months": MONTHS}


def canonical_rows() -> list[dict[str, str]]:
    """Oil mass falls and water mass rises with time, so the water-cut ranking is well defined."""
    rows = []
    for index in range(MONTHS):
        month = month_at(index)
        for position, well in enumerate(PRODUCERS):
            oil_t = 0.0 if (well == "3" and index < 5) else 300.0 - 10.0 * position - 3.0 * index
            water_t = 0.0 if (well == "3" and index < 5) else 100.0 + 20.0 * position + 5.0 * index
            rows.append({"DATA": month, "well": well, "WLPT_Diff": f"{oil_t + water_t}",
                         "WOMT_Diff": f"{oil_t}", "WWIT_Diff": "0"})
        for well in INJECTORS:
            rows.append({"DATA": month, "well": well, "WLPT_Diff": "0", "WOMT_Diff": "0",
                         "WWIT_Diff": "9000"})
    return rows


def export_manifest() -> dict:
    return {"conversion": {"density_by_well": {
        well: {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}
        for well in PRODUCERS + INJECTORS}}}


def build(**overrides):
    options = dict(seed=20260909, blocks_path=None, block_count=6, shut_counts=(5, 10, 20),
                   shut_months=(6, 12), switch_months=(8, 16), conversion_anchor="104",
                   conversion_from_month=0, injection_cap_m3d=bank.INJECTION_CAP_M3D)
    options.update(overrides)
    request = baseline_request()
    return request, *bank.build_bank(request, canonical_rows(), export_manifest(), **options)


class ProducedWaterArithmetic(unittest.TestCase):
    def test_three_well_water_and_oil_volumes(self):
        densities = {"A": (800.0, 1000.0), "B": (800.0, 1250.0), "C": (800.0, 1000.0)}
        rows = [{"DATA": "2007-01-01", "well": "A", "WLPT_Diff": "300", "WOMT_Diff": "80"},
                {"DATA": "2007-01-01", "well": "B", "WLPT_Diff": "250", "WOMT_Diff": "0"},
                {"DATA": "2007-01-01", "well": "C", "WLPT_Diff": "0", "WOMT_Diff": "0"}]
        water, oil = bank.canonical_volumes(rows, densities)
        self.assertAlmostEqual(water["A"]["2007-01-01"], 220.0)
        self.assertAlmostEqual(oil["A"]["2007-01-01"], 100.0)
        self.assertAlmostEqual(water["B"]["2007-01-01"], 200.0)
        self.assertEqual(water["C"]["2007-01-01"], 0.0)
        # 420 m3 over a 31-day month.
        self.assertAlmostEqual(bank.field_water_m3d(water, ["2007-01-01"])["2007-01-01"],
                               420.0 / 31)

    def test_oil_mass_above_liquid_mass_is_rejected(self):
        rows = [{"DATA": "2007-01-01", "well": "A", "WLPT_Diff": "10", "WOMT_Diff": "20"}]
        with self.assertRaises(bank.BankError):
            bank.canonical_volumes(rows, {"A": (800.0, 1000.0)})


class ExportTolerances(unittest.TestCase):
    """Every tolerance the bank applies: inside it is clamped and counted, outside refuses."""

    def test_oil_above_liquid_within_tolerance_is_clamped_and_counted(self):
        densities = {"A": (800.0, 1000.0)}
        log = bank.ToleranceLog({"oil_above_liquid_rel_tol": bank.OIL_ABOVE_LIQUID_REL_TOL})
        # 1e-4 * 100 t = 0.01 t of slack; the row is 0.005 t over.
        rows = [{"DATA": "2007-01-01", "well": "A", "WLPT_Diff": "100.0",
                 "WOMT_Diff": "100.005"}]

        water, oil = bank.canonical_volumes(rows, densities, log)

        self.assertEqual(water["A"]["2007-01-01"], 0.0)
        self.assertAlmostEqual(oil["A"]["2007-01-01"], 100.005 * 1000.0 / 800.0)
        rule = log.as_manifest()["rules"]["oil_above_liquid_t"]
        self.assertEqual(rule["count"], 1)
        self.assertAlmostEqual(rule["max_magnitude"], 0.005)
        self.assertEqual((rule["first_well"], rule["first_date"]), ("A", "2007-01-01"))

    def test_oil_above_liquid_just_outside_tolerance_is_refused(self):
        rows = [{"DATA": "2007-01-01", "well": "A", "WLPT_Diff": "100.0",
                 "WOMT_Diff": "100.02"}]
        with self.assertRaisesRegex(bank.BankError, "oil mass exceeds liquid mass"):
            bank.canonical_volumes(rows, {"A": (800.0, 1000.0)})

    def test_connection_mean_density_is_used_and_counted(self):
        manifest = {"conversion": {
            "density_by_well": {"A": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}},
            "connection_density_by_well": {
                "A": {"oil_kg_m3": [700.0], "water_kg_m3": [900.0]},
                "B": {"oil_kg_m3": [800.0, 900.0], "water_kg_m3": [1000.0, 1100.0]},
                "C": {"oil_kg_m3": [], "water_kg_m3": []}}}}
        log = bank.ToleranceLog({})

        densities = bank.surface_densities(manifest, log)

        # An explicit well-level density always wins over the connection mean.
        self.assertEqual(densities["A"], (800.0, 1000.0))
        self.assertEqual(densities["B"], (850.0, 1050.0))
        self.assertNotIn("C", densities)
        rule = log.as_manifest()["rules"]["connection_mean_density_rel_spread"]
        self.assertEqual((rule["count"], rule["first_well"]), (1, "B"))
        self.assertAlmostEqual(rule["max_magnitude"], 100.0 / 850.0)

    def test_producing_well_without_a_usable_density_is_refused(self):
        densities = bank.surface_densities(
            {"conversion": {"density_by_well": {"A": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}},
                            "connection_density_by_well": {"C": {"oil_kg_m3": [],
                                                                 "water_kg_m3": []}}}})
        rows = [{"DATA": "2007-01-01", "well": "C", "WLPT_Diff": "5", "WOMT_Diff": "1"}]
        with self.assertRaisesRegex(bank.BankError, "no surface density"):
            bank.canonical_volumes(rows, densities)

    def test_bank_manifest_carries_the_tolerance_block(self):
        _, _, manifest, _ = build()
        applied = manifest["tolerances_applied"]
        self.assertEqual(applied["thresholds"]["oil_above_liquid_rel_tol"],
                         bank.OIL_ABOVE_LIQUID_REL_TOL)
        # The synthetic baseline needs no tolerance at all.
        self.assertEqual(applied["rules"], {})


class Families(unittest.TestCase):
    def test_counts_and_identifiers(self):
        _, scenarios, manifest, _ = build()
        counts = {family: sum(1 for s in scenarios if s["family"] == family)
                  for family in ("F1", "F2", "F3", "F4", "F5")}
        self.assertEqual(counts, {"F1": 9, "F2": 6, "F3": 5, "F4": 6, "F5": 8})
        self.assertEqual(len(scenarios), 34)
        self.assertEqual(len({s["id"] for s in scenarios}), 34)
        self.assertTrue(all(bank._IDENTIFIER.fullmatch(s["id"]) for s in scenarios))
        self.assertFalse({s["id"] for s in scenarios} & bank.HELD_OUT_TEST_IDS)
        self.assertFalse(manifest["blocks"]["geological"])

    def test_injection_follows_produced_water_and_baseline_wrat_shares(self):
        request, scenarios, _, derived = build()
        scenario = next(s for s in scenarios
                        if s["parameters"].get("phi") == 0.85
                        and s["parameters"].get("producer_multiplier") == 1.0
                        and s["family"] == "F1")
        month = derived["months"][0]
        injection = [a for a in scenario["controls"]
                     if a["month"] == month and a["role"] == "injector"]
        self.assertEqual(len(injection), len(INJECTORS))
        total = sum(a["value"] for a in injection)
        self.assertAlmostEqual(total, min(0.85 * derived["water_m3d"][month],
                                          bank.INJECTION_CAP_M3D))
        # Equal baseline WRAT means an equal split.
        self.assertAlmostEqual(injection[0]["value"], injection[1]["value"])
        producers = [a for a in scenario["controls"]
                     if a["month"] == month and a["role"] == "producer"]
        source = {(a["month"], a["well"]): a for a in request["controls"]}
        self.assertTrue(all(a["value"] == source[a["month"], a["well"]]["value"]
                            for a in producers))

    def test_producer_multiplier_clips_liquid_at_500(self):
        _, scenarios, _, _ = build()
        scenario = next(s for s in scenarios if s["id"] == "feasible-f1-p070-m110")
        liquid = [a["value"] for a in scenario["controls"] if a["target"] == "LRAT"]
        self.assertTrue(all(value <= bank.LRAT_CAP_M3D + 1e-9 for value in liquid))
        self.assertEqual(max(liquid), bank.LRAT_CAP_M3D)  # the clip must actually bind

    def test_shut_family_shuts_the_wettest_producers_one_way(self):
        _, scenarios, _, derived = build()
        scenario = next(s for s in scenarios if s["id"] == "feasible-f2-s10-m006")
        chosen = set(scenario["parameters"]["shut_wells"])
        self.assertEqual(len(chosen), 10)
        # Water cut grows with the well position, so the last producers are the wettest.
        self.assertIn(str(len(PRODUCERS) - 1), chosen)
        for action in scenario["controls"]:
            if action["well"] in chosen:
                index = derived["months"].index(action["month"])
                self.assertEqual(action["status"], "SHUT" if index >= 6 else "OPEN")

    def test_conversions_include_the_anchor_with_explicit_rate_and_ceiling(self):
        _, scenarios, _, _ = build()
        conversions = [s for s in scenarios if s["family"] == "F3"]
        anchored = [s for s in conversions
                    if any(c["well"] == "104" for c in s["parameters"]["conversions"])]
        self.assertEqual(len(anchored), 3)
        scenario = next(s for s in conversions
                        if [c["well"] for c in s["parameters"]["conversions"]] == ["104"])
        converted = [a for a in scenario["controls"] if a["well"] == "104"]
        self.assertTrue(all(a["role"] == "injector" and a["target"] == "WRAT"
                            and a["bhp_limit"] == bank.CONVERSION_BHP_BAR for a in converted))
        self.assertEqual(scenario["parameters"]["conversions"][0]["wrat_m3d"],
                         bank.CONVERSION_WRAT_M3D)

    def test_time_varying_and_block_families_vary(self):
        _, scenarios, _, derived = build()
        switching = next(s for s in scenarios if s["id"] == "feasible-f4-00")
        values = {}
        for action in switching["controls"]:
            if action["well"] == "1":
                values[derived["months"].index(action["month"])] = action["value"]
        self.assertLess(values[0], values[8])
        self.assertLess(values[8], values[16])
        block_scales = [s["parameters"]["block_scales"] for s in scenarios if s["family"] == "F5"]
        self.assertEqual(len(block_scales), 8)
        self.assertEqual(len(block_scales[0]), 6)
        self.assertTrue(all(bank.BLOCK_SCALE_RANGE[0] <= value <= bank.BLOCK_SCALE_RANGE[1]
                            for scales in block_scales for value in scales.values()))
        self.assertNotEqual(block_scales[0], block_scales[1])

    def test_generation_is_deterministic_for_one_seed(self):
        latin = lambda scenarios: [s["parameters"]["block_scales"]  # noqa: E731
                                   for s in scenarios if s["family"] == "F5"]
        first, second, other = build()[1], build()[1], build(seed=7)[1]
        self.assertEqual([(s["id"], s["parameters"]) for s in first],
                         [(s["id"], s["parameters"]) for s in second])
        self.assertNotEqual(latin(first), latin(other))


class Invariants(unittest.TestCase):
    def test_every_emitted_control_survives_the_hard_rules(self):
        request, scenarios, _, _ = build()
        baseline = request["controls"]
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                bank.check_invariants(scenario["controls"], baseline)
                roles: dict[str, str] = {}
                for action in sorted(scenario["controls"], key=lambda a: (a["month"], a["well"])):
                    self.assertGreaterEqual(action["value"], 0.0)
                    if action["status"] == "SHUT":
                        self.assertEqual(action["value"], 0.0)
                    if action["role"] == "injector":
                        self.assertEqual(action["target"], "WRAT")
                    if action["target"] == "LRAT":
                        self.assertLessEqual(action["value"], bank.LRAT_CAP_M3D + 1e-9)
                    self.assertFalse(roles.get(action["well"]) == "injector"
                                     and action["role"] == "producer")
                    roles[action["well"]] = action["role"]
                    if action["well"] == "3" and action["month"] < month_at(5):
                        self.assertEqual((action["status"], action["value"]), ("SHUT", 0.0))

    def test_production_before_the_first_source_month_is_rejected(self):
        request = baseline_request()
        broken = [dict(a) for a in request["controls"]]
        early = next(a for a in broken if a["well"] == "3" and a["month"] == month_at(0))
        early.update(status="OPEN", value=50.0)
        with self.assertRaises(bank.BankError):
            bank.check_invariants(broken, request["controls"])

    def test_relaxed_bhp_bounds_are_rejected(self):
        request = baseline_request()
        for well, limit in (("1", 50.0), ("201", 300.0)):
            broken = [dict(a) for a in request["controls"]]
            next(a for a in broken if a["well"] == well)["bhp_limit"] = limit
            with self.assertRaises(bank.BankError):
                bank.check_invariants(broken, request["controls"])

    def test_reverse_conversion_is_rejected(self):
        request = baseline_request()
        broken = [dict(a) for a in request["controls"]]
        for action in broken:
            if action["well"] == "201" and action["month"] == month_at(MONTHS - 1):
                action.update(role="producer", target="LRAT", value=10.0, bhp_limit=280.0)
        with self.assertRaises(bank.BankError):
            bank.check_invariants(broken, request["controls"])


class Split(unittest.TestCase):
    def test_frozen_test_scenarios_never_train(self):
        for name in ("physical-sweep-03", "physical-sweep-05", "physical-sweep-06",
                     "fresh-uncertainty-00"):
            self.assertEqual(bank.split_of(name), "test")

    def test_assignment_is_deterministic_and_roughly_balanced(self):
        names = [f"feasible-case-{index:04d}" for index in range(4000)]
        assignment = [bank.split_of(name) for name in names]
        self.assertEqual(assignment, [bank.split_of(name) for name in names])
        share = {split: assignment.count(split) / len(names) for split in set(assignment)}
        self.assertAlmostEqual(share["train"], 0.70, delta=0.03)
        self.assertAlmostEqual(share["validation"], 0.15, delta=0.03)
        self.assertAlmostEqual(share["test"], 0.15, delta=0.03)


class Output(unittest.TestCase):
    def test_requests_and_manifest_pass_the_cycle_contract(self):
        request, scenarios, manifest, derived = build()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "bank"
            written = bank.write_bank(output, request, scenarios, manifest, seed=20260909,
                                      water_m3d=derived["water_m3d"], months=derived["months"])
            self.assertEqual(written["counts"]["F1"], 9)
            self.assertEqual(sum(written["counts"][f"split:{s}"]
                                 for s in ("train", "validation", "test")), 34)
            # A hash split of 34 identifiers is lumpy; the manifest must say so.
            thin = [s for s in ("train", "validation", "test")
                    if written["counts"][f"split:{s}"] < 3]
            self.assertEqual(bool(thin), bool(written["split_warning"]))
            entries = {entry["id"]: entry for entry in written["scenarios"]}
            self.assertEqual(len(entries), 34)
            for entry in entries.values():
                payload = json.loads((output / entry["request"]).read_text())
                cycle = bank.CycleRequest.from_mapping(payload)
                self.assertEqual(cycle.controls_sha256, entry["controls_sha256"])
                self.assertEqual(cycle.request_sha256, entry["request_sha256"])
                self.assertEqual(cycle.horizon_months, MONTHS)
                self.assertEqual(len(entry["expected"]["water_m3d"]), MONTHS)
                self.assertEqual(entry["expected"]["injection_m3d"][0],
                                 round(sum(a["value"] for a in payload["controls"]
                                           if a["month"] == derived["months"][0]
                                           and a["role"] == "injector" and a["status"] == "OPEN"),
                                       6))
            conversion = entries["feasible-f3-00"]
            self.assertTrue(json.loads((output / conversion["request"]).read_text())
                            ["context"]["constraints"]["allow_conversion_to_injection"])
            self.assertFalse(json.loads((output / entries["feasible-f1-p085-m100"]["request"])
                                        .read_text())["context"]["constraints"]
                             ["allow_conversion_to_injection"])
            with self.assertRaises(FileExistsError):
                bank.write_bank(output, request, scenarios, manifest, seed=20260909,
                                water_m3d=derived["water_m3d"], months=derived["months"])

    def test_physical_entries_register_without_a_request(self):
        with tempfile.TemporaryDirectory() as raw:
            export = Path(raw) / "candidate-03"
            export.mkdir()
            for name in ("chdd.csv", "trajectory.csv", "manifest.json"):
                (export / name).write_text("{}")
            entries = bank.physical_entries([f"physical-sweep-03={export}"])
            self.assertEqual(entries[0]["family"], "F6")
            self.assertEqual(entries[0]["split"], "test")
            self.assertIsNone(entries[0]["request"])
            self.assertEqual(len(entries[0]["canonical_sha256"]), 3)
            with self.assertRaises(bank.BankError):
                bank.physical_entries([f"physical-sweep-03={raw}/absent"])


class Blocks(unittest.TestCase):
    def test_blocks_file_overrides_the_production_quantile_fallback(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "blocks.json"
            # Mirrors scripts/export_blocks.py: integer block ids, injectors included.
            groups = [{"id": index, "wells": list((PRODUCERS + INJECTORS)[index::3]),
                       "component": 1, "centroid_ij": [1.0, 2.0]} for index in range(3)]
            path.write_text(json.dumps({"schema": "timesoil.blocks/1", "blocks": groups,
                                        "well_to_block": {well: index for index in range(3)
                                                          for well in groups[index]["wells"]}}))
            _, scenarios, manifest, _ = build(blocks_path=path)
            self.assertTrue(manifest["blocks"]["geological"])
            scales = next(s for s in scenarios if s["family"] == "F5")["parameters"]["block_scales"]
            self.assertEqual(sorted(scales), ["0", "1", "2"])

    def test_incomplete_blocks_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "blocks.json"
            path.write_text(json.dumps({"blocks": [{"id": "a", "wells": ["1", "2"]}]}))
            with self.assertRaises(bank.BankError):
                build(blocks_path=path)


if __name__ == "__main__":
    unittest.main()


class CapBasisAndLiquidCap(unittest.TestCase):
    """The test case injects an external supply near the cap and caps field liquid."""

    def _baseline(self):
        months = ["2007-01-01", "2007-02-01"]
        controls = []
        for month in months:
            controls += [
                dict(month=month, well="P1", role="producer", status="OPEN", target="LRAT", value=400.0, bhp_limit=55.0),
                dict(month=month, well="P2", role="producer", status="OPEN", target="LRAT", value=400.0, bhp_limit=55.0),
                dict(month=month, well="I1", role="injector", status="OPEN", target="WRAT", value=300.0, bhp_limit=290.0),
                dict(month=month, well="I2", role="injector", status="OPEN", target="WRAT", value=100.0, bhp_limit=290.0),
            ]
        return controls, {m: 40.0 for m in months}

    def test_cap_basis_allocates_share_of_the_cap_not_produced_water(self):
        from build_feasible_bank import regime_controls
        baseline, water = self._baseline()
        out = regime_controls(baseline, producer_scale=lambda *_: 1.0, phi=lambda _: 0.85,
                              water_m3d=water, injection_cap_m3d=600.0, injection_basis="cap")
        injectors = [a for a in out if a["month"] == "2007-01-01" and a["role"] == "injector"]
        self.assertAlmostEqual(sum(a["value"] for a in injectors), 510.0)
        self.assertAlmostEqual(next(a["value"] for a in injectors if a["well"] == "I1"), 382.5)
        water_out = regime_controls(baseline, producer_scale=lambda *_: 1.0, phi=lambda _: 0.85,
                                    water_m3d=water, injection_cap_m3d=600.0, injection_basis="water")
        self.assertAlmostEqual(sum(a["value"] for a in water_out if a["month"] == "2007-01-01" and a["role"] == "injector"), 34.0)

    def test_field_liquid_cap_scales_open_lrat_producers_proportionally(self):
        from build_feasible_bank import regime_controls, BankError
        baseline, water = self._baseline()
        out = regime_controls(baseline, producer_scale=lambda *_: 1.0, phi=lambda _: 1.0,
                              water_m3d=water, injection_cap_m3d=600.0, injection_basis="cap",
                              liquid_cap_m3d=600.0)
        producers = [a for a in out if a["month"] == "2007-02-01" and a["role"] == "producer"]
        self.assertAlmostEqual(sum(a["value"] for a in producers), 600.0)
        self.assertTrue(all(abs(a["value"] - 300.0) < 1e-9 for a in producers))
        with self.assertRaises(BankError):
            regime_controls(baseline, producer_scale=lambda *_: 1.0, phi=lambda _: 1.0,
                            water_m3d=water, injection_cap_m3d=600.0, injection_basis="tank")
