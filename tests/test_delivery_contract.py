from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_compose() -> dict[str, Any] | None:
    docker = shutil.which("docker")
    if docker is None:
        return None
    version = subprocess.run(
        [docker, "compose", "version"], capture_output=True, text=True, timeout=10, env={}
    )
    if version.returncode != 0:
        return None
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            str(ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env={"QWEN_API_KEY_FILE": "/dev/null"},
    )
    assert result.returncode == 0, result.stderr
    effective = json.loads(result.stdout)
    assert isinstance(effective, dict)
    return effective


def test_release_delivery_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    base_images = [line.split()[1] for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(base_images) >= 2
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in base_images)

    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:${AIOS_PORT:-8000}:8000"' in compose
    assert "read_only: true" in compose
    assert re.search(r"(?m)^\s+cap_drop:\s*\n\s+- ALL$", compose)
    assert "no-new-privileges:true" in compose
    assert re.search(r"(?m)^\s+secrets:\s*\n\s+- qwen_api_key$", compose)
    assert re.search(
        r'(?m)^  qwen_api_key:\s*\n    file: "\$\{QWEN_API_KEY_FILE:\?.+\}"$', compose
    )
    assert "/run/secrets/qwen_api_key" in compose
    assert "test -s /run/secrets/qwen_api_key" in compose
    assert 'key="$$(cat /run/secrets/qwen_api_key)" || exit 1' in compose
    assert 'test -n "$$key"' in compose
    assert 'export LLM_API_KEY="$$key"' in compose
    assert "unset key" in compose
    assert 'exec "$$@"' in compose
    assert "docker.sock" not in compose.lower()

    effective_compose = _effective_compose()
    if effective_compose is not None:
        api = effective_compose["services"]["api"]
        assert len(api["ports"]) == 1
        port = api["ports"][0]
        assert port["host_ip"] == "127.0.0.1"
        assert port["target"] == 8000
        assert port["published"] == "8000"
        assert port["protocol"] == "tcp"
        assert api["read_only"] is True
        assert api["cap_drop"] == ["ALL"]
        assert api["security_opt"] == ["no-new-privileges:true"]
        assert api["secrets"] == [
            {"source": "qwen_api_key", "target": "/run/secrets/qwen_api_key"}
        ]
        assert api["command"] == [
            "uvicorn",
            "timesoil.aios.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        assert api["entrypoint"][-1] == "--"
        assert "test -s /run/secrets/qwen_api_key" in api["entrypoint"][-2]
        assert 'key="$$(cat /run/secrets/qwen_api_key)" || exit 1' in api["entrypoint"][-2]
        assert 'test -n "$$key"' in api["entrypoint"][-2]
        assert 'export LLM_API_KEY="$$key"' in api["entrypoint"][-2]
        assert "unset key" in api["entrypoint"][-2]
        assert 'exec "$$@"' in api["entrypoint"][-2]
        assert effective_compose["secrets"]["qwen_api_key"]["file"] == "/dev/null"
        assert all("docker.sock" not in str(volume).lower() for volume in api.get("volumes", []))

    compose_env = _env(ROOT / ".env.example")
    assert compose_env["QWEN_API_KEY_FILE"] == "/dev/shm/timesoil-qwen-api-key"
    assert "LLM_API_KEY" not in compose_env

    config_path = ROOT / "config" / "kt2.operator.example.env"
    bash = shutil.which("bash")
    assert bash is not None
    for command in (
        [bash, "-n", str(config_path)],
        [bash, "--noprofile", "--norc", "-eu", "-o", "pipefail", "-c", 'source "$1"', "_", str(config_path)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, env={})
        assert result.returncode == 0, result.stderr

    config = _env(config_path)
    expected = {
        "LOCAL_PROJECT_ROOT": "/CHANGE_ME/local/TimesOil",
        "A100_PROJECT_ROOT": "/CHANGE_ME/a100/TimesOil",
        "A100_INPUT_ROOT": "/CHANGE_ME/a100/input",
        "A100_WORK_ROOT": "/CHANGE_ME/a100/work",
        "OPM_IMAGE_TAG": "openporousmedia/opmreleases:2026.04_amd64",
        "OPM_IMAGE_DIGEST": "sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9",
        "OPM_IMAGE_REFERENCE": "openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9",
        "MODEL_Y_SOURCE": "/CHANGE_ME/a100/input/Model_Y (3).zip",
        "TRACK1_BASELINE_DIR": "/CHANGE_ME/a100/TimesOil/results/model-y-baseline-20260831-a100-v4",
        "TRACK1_REFERENCE_WORKBOOK": "/CHANGE_ME/a100/TimesOil/docs/hackathon/chdd/reference_baselines/Расчет ЧДД через OPM Flow Model_Y.xlsx",
        "TRACK1_SOURCE_ARCHIVE_RELATIVE_PATH": "Model_Y (3).zip",
        "TRACK1_OFFICIAL_SOURCE_SHA256": "261591b458084eaaf8c86a601e68d3bdc6e91fed9f0117fdcbe58cfca4eb882e",
        "TRACK1_DECK_RELATIVE_PATH": "MODEL_Y/MODEL_Y.DATA",
        "TRACK1_SCHEDULE_RELATIVE_PATH": "MODEL_Y/INCLUDE/DemoSpe_002_2_sch.inc",
        "TRACK1_PLANNING_HORIZON_MONTHS": "6",
        "TRACK1_OPM_TIMEOUT_SECONDS": "3600",
        "TRACK1_OPM_PARSING_STRICTNESS": "low",
        "TRACK1_CHDD_VALIDATION_START_YEAR": "2007",
        "TRACK1_CHDD_VALIDATION_PROFILE": "organizer_full_history",
        "TRACK1_CHDD_VALIDATION_CHARGE_INITIAL_PUMP": "true",
        "TRACK1_CHDD_OPERATIONAL_START_YEAR": "2014",
        "TRACK1_CHDD_OPERATIONAL_PROFILE": "operational_sunk_assets",
        "TRACK1_CHDD_OPERATIONAL_CHARGE_INITIAL_PUMP": "false",
        "TRACK2_SOURCE_ARCHIVE_RELATIVE_PATH": "Model_Z_final_OPM.zip",
        "TRACK2_OFFICIAL_SOURCE_SHA256": "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa",
        "TRACK2_SCENARIO_INDEX_SHA256": "71edcb70cf4e04871f81e6d6ed4842f8cc91d542731024269060a1c8f5cfaf54",
        "MODEL_Z_BASELINE_CHDD_SHA256": "446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884",
        "TRACK2_DECK_RELATIVE_PATH": "Model_Z/Model_Z.data",
        "TRACK2_SCHEDULE_RELATIVE_PATH": "Model_Z/Model_Z_sch.inc",
        "TRACK2_SCENARIO_SEED": "20260831",
        "TRACK2_TRAIN_SCENARIO_COUNT": "4",
        "TRACK2_TRAIN_PERTURBATION_FRACTION": "0.15",
        "TRACK2_SURROGATE_TEST_FRACTION": "0.25",
        "TRACK2_SURROGATE_ENSEMBLE_SIZE": "5",
        "TRACK2_SURROGATE_N_ESTIMATORS": "160",
        "TRACK2_SEARCH_CANDIDATE_COUNT": "32",
        "TRACK2_SEARCH_PERTURBATION_FRACTION": "0.05",
        "TRACK2_SEARCH_UNCERTAINTY_WEIGHT": "1.0",
        "TRACK2_SEARCH_INJECTION_COST_EQUIVALENT": "0.01",
        "TRACK2_SEARCH_HORIZON_MONTHS": "6",
        "TRACK2_SEARCH_SCENARIO_ID": "baseline",
        "TRACK2_OPM_TIMEOUT_SECONDS": "3600",
        "TRACK2_OPM_PARSING_STRICTNESS": "low",
        "TRACK2_CHDD_VALIDATION_START_YEAR": "1991",
        "TRACK2_CHDD_VALIDATION_PROFILE": "organizer_full_history",
        "TRACK2_CHDD_VALIDATION_CHARGE_INITIAL_PUMP": "true",
        "TRACK2_CHDD_OPERATIONAL_START_YEAR": "2007",
        "TRACK2_CHDD_OPERATIONAL_PROFILE": "operational_sunk_assets",
        "TRACK2_CHDD_OPERATIONAL_CHARGE_INITIAL_PUMP": "false",
        "CHDD_TIMEOUT_SECONDS": "120",
    }
    assert config.items() >= expected.items()
    runtime_paths = {
        "MODEL_Z_SOURCE",
        "MODEL_Z_DECK",
        "SCHEDULE_RELATIVE_PATH",
        "SCHEDULE_INCLUDE",
        "SCENARIO_BUNDLE",
        "OUTPUT_BUNDLE",
        "SCENARIO_RUNS",
        "MODEL_Z_DATASET_DIR",
        "MODEL_Z_MANIFEST_DIR",
        "MODEL_Z_DATASET",
        "MODEL_Z_EXPORT_MANIFEST",
        "TRACK2_TRAIN_OUTPUT",
        "TRACK2_MODEL",
        "TRACK2_METRICS",
        "TRACK2_SEARCH_SCENARIO_ID",
        "SEARCH_DIR",
        "REPLAY_DIR",
        "OPM_RUNS",
        "MODEL_Y_SOURCE",
        "TRACK1_BASELINE_DIR",
        "TRACK1_REFERENCE_WORKBOOK",
        "TRACK1_PROOF_OUTPUT_ROOT",
    }
    assert config.keys() >= runtime_paths
    assert all(config[name] for name in runtime_paths)
    assert all(
        "/CHANGE_ME/" in config[name]
        for name in runtime_paths
        - {"MODEL_Z_DECK", "SCHEDULE_RELATIVE_PATH", "TRACK2_SEARCH_SCENARIO_ID"}
    )
    assert config["TRACK2_SEARCH_SCENARIO_ID"] == "baseline"
    assert config["MODEL_Z_DATASET"].endswith("/dataset/baseline.csv")
    assert config["MODEL_Z_EXPORT_MANIFEST"].endswith("/manifests/baseline.json")
    assert config["TRACK2_MODEL"] == config["TRACK2_TRAIN_OUTPUT"] + "/model"
    assert config["OUTPUT_BUNDLE"] != config["SCENARIO_BUNDLE"]
    assert config["TRACK2_METRICS"] == config["TRACK2_TRAIN_OUTPUT"] + "/metrics.json"
    assert not any(
        marker in key.upper()
        for key in config
        for marker in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")
    )

    checkpoint = (ROOT / "docs" / "hackathon" / "checkpoint_2_2026-08-31.md").read_text(
        encoding="utf-8"
    )
    assert checkpoint.index('cd "$A100_PROJECT_ROOT"') < checkpoint.index(
        "uv sync --locked"
    )
    bash_blocks = re.findall(r"(?ms)^```bash\n(.*?)^```$", checkpoint)
    available = set(config)
    missing: set[str] = set()
    for line in "\n".join(bash_blocks).splitlines():
        defaulted = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*):[-=?+]", line))
        references = {
            braced or plain
            for braced, plain in re.findall(
                r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}|([A-Za-z_][A-Za-z0-9_]*))",
                line,
            )
        }
        missing.update(references - available - defaulted)
        assignment = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        if assignment:
            available.add(assignment.group(1))
    assert not missing, f"checkpoint references undefined shell variables: {sorted(missing)}"
    track1_command = next(
        block for block in bash_blocks if "run_model_y_track1_proof.py" in block
    )
    for fragment in (
        '--source "$MODEL_Y_SOURCE"',
        '--baseline-dir "$TRACK1_BASELINE_DIR"',
        '--reference-workbook "$TRACK1_REFERENCE_WORKBOOK"',
        '--deck "$TRACK1_DECK_RELATIVE_PATH"',
        '--schedule-relative-path "$TRACK1_SCHEDULE_RELATIVE_PATH"',
        '--output-root "$TRACK1_PROOF_OUTPUT_ROOT"',
    ):
        assert fragment in track1_command

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert dockerignore == [
        "**",
        "!pyproject.toml",
        "!uv.lock",
        "!src/",
        "!src/**",
        "!docs/",
        "!docs/hackathon/",
        "!docs/hackathon/chdd/",
        "!docs/hackathon/chdd/CHDD_PYTHON/",
        "!docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py",
        "!docs/hackathon/chdd/CHDD_PYTHON/chdd_model.py",
        "!docs/hackathon/chdd/CHDD_PYTHON/excel_io.py",
        "!docs/hackathon/chdd/CHDD_PYTHON/input/",
        "!docs/hackathon/chdd/CHDD_PYTHON/input/Нормативы_ЧДД.xlsx",
        "**/__pycache__/",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
    ]

    deliverable = ROOT / "docs" / "hackathon" / "deliverables" / "track1_proof"
    submission = json.loads((deliverable / "submission.json").read_text(encoding="utf-8"))
    for schedule in submission["schedules"].values():
        assert _sha256(deliverable / schedule["path"]) == schedule["sha256"]
    evidence_path = (deliverable / submission["evidence_receipt"]["path"]).resolve()
    assert _sha256(evidence_path) == submission["evidence_receipt"]["sha256"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert (
        submission["schedules"]["objective_winner"]["sha256"]
        == evidence["result_bundle"]["wells_schedule_sha256"]
    )
