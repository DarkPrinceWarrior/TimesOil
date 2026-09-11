from hashlib import sha256
import importlib.util
from pathlib import Path

import pytest
from timesoil.aios.opm import build_summary_overlay

spec = importlib.util.spec_from_file_location("compare_cycles", Path(__file__).parents[1] / "scripts/compare_track2_cycles.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_canonical_optional_summary_change_is_comparable(tmp_path):
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    rel = "input/Model_Z/_TIMESOIL_SUMMARY.INC"
    current = build_summary_overlay(("P", "I")).encode()
    legacy = current.replace(b"WVPT\n/\n", b"").replace(b"WVIT\n/\n", b"")
    def compare(raw, *, grid="same", extra=False):
        inputs = []
        for root, data in zip(roots, (legacy, raw)):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
            inputs.append({rel: sha256(data).hexdigest(), "input/grid.inc": "same"})
        inputs[1]["input/grid.inc"] = grid
        if extra:
            inputs[1]["input/other.inc"] = "extra"
        return module.compare_physical_inputs(roots, inputs, ("I", "P"))
    assert compare(legacy) == []
    changes = compare(current)
    assert changes[0]["path"] == rel and "WVPT/WVIT" in changes[0]["allowed_difference"]
    for raw in (current + b"PINCH\n2 /\n", current.replace(b"WOPR\n/\n", b""),
                current.replace(b"WVIT\n/\n", b""), build_summary_overlay(("P",)).encode()):
        with pytest.raises(AssertionError):
            compare(raw)
    with pytest.raises(AssertionError):
        compare(current, grid="changed")
    with pytest.raises(AssertionError):
        compare(current, extra=True)
