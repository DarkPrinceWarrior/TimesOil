from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Callable
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_track2_surrogate.py"
SPEC = importlib.util.spec_from_file_location("train_track2_surrogate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeModel:
    def __init__(self, mutate: Callable[[], object] | None = None) -> None:
        self._mutate = mutate

    def save(self, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True)
        manifest = {"artifact_hash": "a" * 64}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if self._mutate is not None:
            self._mutate()
        return manifest


class _FakeRun:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def report(self) -> dict[str, object]:
        return {"status": "ok"}


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=Path("dataset.csv"),
        manifest=Path("manifest.json"),
        raw_dir=Path("raw_data"),
        output=output,
        test_fraction=0.25,
        ensemble_size=2,
        n_estimators=4,
        horizon=2,
        seed=42,
        conformal_level=0.9,
    )


class TrainTrack2SurrogateSourceContractTest(unittest.TestCase):
    def test_read_regular_rejects_final_swap_to_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_bytes(b"trusted source\n")
            attacker = root / "attacker.py"
            attacker.write_bytes(b"attacker source\n")
            moved = root / "moved-source.py"
            real_open = MODULE.os.open
            swapped = False

            def swapping_open(
                path: str | bytes,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal swapped
                if path == source.name and dir_fd is not None and not swapped:
                    source.rename(moved)
                    source.symlink_to(attacker)
                    swapped = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(MODULE.os, "open", side_effect=swapping_open):
                with self.assertRaises(OSError):
                    MODULE._read_regular(source, "source")

            self.assertTrue(swapped)

    def test_torn_source_read_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"A" * (2 * 1024**2))
            frozen = trainer_source.stat()
            target = (frozen.st_dev, frozen.st_ino)
            real_fstat = MODULE.os.fstat
            real_read = MODULE.os.read
            mutated = False

            def frozen_fstat(descriptor: int) -> object:
                result = real_fstat(descriptor)
                return frozen if (result.st_dev, result.st_ino) == target else result

            def mutating_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                chunk = real_read(descriptor, size)
                result = real_fstat(descriptor)
                if not mutated and (result.st_dev, result.st_ino) == target and chunk:
                    trainer_source.write_bytes(b"B" * (2 * 1024**2))
                    mutated = True
                return chunk

            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE.os, "fstat", side_effect=frozen_fstat),
                patch.object(MODULE.os, "read", side_effect=mutating_read),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(_FakeModel()),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed while it was read"):
                    MODULE.main()

            self.assertTrue(mutated)
            self.assertFalse(output.exists())

    def test_metrics_bind_executed_sources_and_model_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "surrogate"
            with (
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(_FakeModel()),
                ),
            ):
                self.assertEqual(MODULE.main(), 0)

            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            expected_paths = (
                Path(MODULE.__file__).absolute(),
                Path(MODULE._track2_module.__file__).absolute(),
                Path(MODULE._surrogate_module.__file__).absolute(),
            )
            self.assertEqual(
                metrics["executed_sources"],
                [
                    {
                        "path": str(path),
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in expected_paths
                ],
            )
            manifest = output / "model" / "manifest.json"
            self.assertEqual(metrics["surrogate_artifact_hash"], "a" * 64)
            self.assertEqual(
                metrics["surrogate_manifest_sha256"],
                sha256(manifest.read_bytes()).hexdigest(),
            )

    def test_mutation_before_metrics_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"original source\n")
            model = _FakeModel(
                lambda: trainer_source.write_bytes(b"mutated source\n")
            )
            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(model),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "executed source changed"):
                    MODULE.main()

            self.assertFalse((output / "metrics.json").exists())

    def test_mutation_inside_fit_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"original source\n")

            def mutating_fit(*args: object, **kwargs: object) -> _FakeRun:
                trainer_source.write_bytes(b"mutated source\n")
                return _FakeRun(_FakeModel())

            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(MODULE, "fit_track2_surrogate", side_effect=mutating_fit),
            ):
                with self.assertRaisesRegex(RuntimeError, "executed source changed"):
                    MODULE.main()

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
