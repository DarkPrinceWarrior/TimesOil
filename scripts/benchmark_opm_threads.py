"""Replay authenticated prepared inputs with different OPM thread counts on A100."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from timesoil.aios.opm import OPM_IMAGE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing successful OPM run")
    parser.add_argument("output", type=Path, help="New benchmark directory")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if min([args.jobs, *args.threads]) < 1 or len(set(args.threads)) != len(args.threads):
        parser.error("positive unique thread counts and positive jobs required")
    root = args.run.resolve()
    raw = (root / "manifest.json").read_bytes()
    digest = sha256(raw).hexdigest()
    if (root / "manifest.sha256").read_text().split()[0] != digest:
        raise ValueError("source manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest["status"] != "success" or manifest["image_reference"] != OPM_IMAGE:
        raise ValueError("successful pinned OPM run required")
    deck = Path(manifest["deck"])
    if deck.is_absolute() or ".." in deck.parts:
        raise ValueError("unsafe deck path")
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if path.parts[0] != "input":
            continue
        if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
            raise ValueError("unsafe input artifact")
        if sha256((root / path).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"input artifact changed: {path}")
    proof = json.loads((root / "summary-extraction.json").read_text())
    vectors = [v for v in proof["commands"]["report"]
               if v.split(":")[0] in {"WOPT", "WWPT", "WWIT", "WBHP", "WBP9"}]
    if not vectors:
        raise ValueError("well volumes and pressures required for numerical comparison")
    reference = pd.read_csv(root / "summary-report.txt", sep=r"\s+", usecols=vectors)[vectors]
    args.output.mkdir(parents=True, exist_ok=False)

    def replay(threads: int) -> dict:
        output = (args.output / str(threads)).resolve()
        output.mkdir()
        name = "timesoil-thread-bench-" + sha256(str(output).encode()).hexdigest()[:16]
        command = ["docker", "run", "--rm", "--name", name, "--network=none",
                   "--user", "0:0", "--mount", f"type=bind,src={root / 'input'},dst=/case,readonly",
                   "--mount", f"type=bind,src={output},dst=/output", OPM_IMAGE, "flow",
                   "--output-dir=/output", f"--threads-per-process={threads}"]
        if "--parsing-strictness=low" in manifest["command"]:
            command.append("--parsing-strictness=low")
        command.append(f"/case/{deck.as_posix()}")
        started = time.monotonic()
        with (output / "stdout.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "stop", "--time", "1", name], check=False)
                raise
        elapsed = time.monotonic() - started
        if result.returncode:
            raise RuntimeError(f"OPM failed for {threads} threads; inspect {output}")
        summary = next(output.glob("*.SMSPEC"))
        report = output / "well-report.txt"
        with report.open("w") as stream:
            subprocess.run(["docker", "run", "--rm", "--network=none", "--user", "0:0",
                            "--mount", f"type=bind,src={output},dst=/output,readonly", OPM_IMAGE,
                            "summary", "-r", f"/output/{summary.name}", *vectors],
                           stdout=stream, check=True, timeout=300)
        actual = pd.read_csv(report, sep=r"\s+")[vectors]
        if actual.shape != reference.shape or not np.isfinite(actual.to_numpy()).all():
            raise ValueError("benchmark produced incomplete or non-finite well history")
        delta = np.abs(actual.to_numpy() - reference.to_numpy())
        record = {"threads": threads, "opm_seconds": elapsed, "command": command,
                  "source_manifest_sha256": digest, "rows": len(actual), "columns": len(vectors),
                  "max_absolute_difference": float(delta.max()),
                  "max_relative_difference": float((delta / np.maximum(np.abs(reference), 1)).max().max()),
                  "within_1ppm": bool(np.allclose(actual, reference, atol=1e-6, rtol=1e-6)),
                  "benchmark_only": True, "official_chdd_evaluated": False}
        (output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "command"}), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(replay, args.threads))
    (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
