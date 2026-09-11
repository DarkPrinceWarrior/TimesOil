#!/usr/bin/env bash
# End-to-end Track 2 run for the organizers' test case on A100.
#
# One new results directory, one protocol, exit files per stage, and a wall-clock
# budget: the window is 17:00–21:00 (4 hours). Every stage refuses to overwrite an
# existing directory; a failed stage stops the chain with a non-zero exit so the
# receipt shows exactly where it stopped. Nothing here weakens a gate: the case
# profile is the only source of limits, and the sealed selection still gets exactly
# one final OPM run.
#
# Usage (from the pinned code worktree on A100):
#   scripts/run_case_z.sh ARCHIVE.zip [--skip-bank] [--skip-finetune] [--epochs N]
#
# Required environment (see docs/CASE_INTAKE_20260911.md):
#   LLM_API_KEY via the runtime key file (test -s, never cat), LLM_BASE_URL, LLM_MODEL
#   OPM_MPI_PROCESSES=16 OPM_THREADS_PER_PROCESS=1 OPM_CPU_AFFINITY=30-45
set -euo pipefail

ARCHIVE=${1:?archive path required}; shift || true
SKIP_BANK=0; SKIP_FINETUNE=0; EPOCHS=20; BANK_RUNS=16
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-bank) SKIP_BANK=1;;
    --skip-finetune) SKIP_FINETUNE=1;;
    --epochs) EPOCHS=$2; shift;;
    --bank-runs) BANK_RUNS=$2; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
  shift
done

CODE=$(cd "$(dirname "$0")/.." && pwd)
R=${R:-/root/projects/TimesOil/results/audit-20260909}
PY_PROJECT=${PY_PROJECT:-/root/projects/TimesOil/.venv/bin/python}
PY_TORCH=${PY_TORCH:-/tmp/timesoil-kt3-20260908/venv/bin/python}
PROFILE=${PROFILE:-$CODE/config/case_z_test.json}
WEIGHTS=${WEIGHTS:-$R/timesfm-economic-regimes-precise-z-20260910/training/full-model.pt}
WEIGHTS_REPORT=${WEIGHTS_REPORT:-$R/timesfm-economic-regimes-precise-z-20260910/training/report.json}
CONNECTIVITY=${CONNECTIVITY:-$R/static-head-geology-20260909/model-z/connectivity.json}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$R/case-z-$STAMP
test ! -e "$OUT"; mkdir "$OUT"

# LLM route: Cerebras (main) unless LLM_BASE_URL is already set; the key never leaves the file.
if [ -z "${LLM_BASE_URL:-}" ]; then
  if test -s /root/.config/timesoil/cerebras-key; then
    export LLM_BASE_URL=https://api.cerebras.ai/v1 LLM_MODEL=qwen-3.8-27b
    export LLM_REASONING_EFFORT=${LLM_REASONING_EFFORT:-high} LLM_SEED=${LLM_SEED:-20260909}
    export LLM_API_KEY="$(cat /root/.config/timesoil/cerebras-key)"
  elif test -s /dev/shm/timesoil-tatneft-20260909-key; then
    export LLM_BASE_URL=https://litellm.tatneft.guru/v1 LLM_MODEL=qwen3.8-27b
    export LLM_API_KEY="$(cat /dev/shm/timesoil-tatneft-20260909-key)"
  else
    echo "no LLM key file found (cerebras or tatneft)" >&2; exit 3
  fi
fi
export LLM_TIMEOUT_SECONDS=${LLM_TIMEOUT_SECONDS:-600} LLM_MAX_OUTPUT_TOKENS=${LLM_MAX_OUTPUT_TOKENS:-8192}
export LLM_CALL_LOG=${LLM_CALL_LOG:-$OUT/llm_calls.jsonl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$CODE/src:$CODE/scripts
export OPM_MPI_PROCESSES=${OPM_MPI_PROCESSES:-16} OPM_THREADS_PER_PROCESS=${OPM_THREADS_PER_PROCESS:-1}
export OPM_CPU_AFFINITY=${OPM_CPU_AFFINITY:-30-45}
CPUS=${CPUS:-14-29}
T0=$(date +%s)

stage() {  # stage NAME COMMAND... — logs, exit file, elapsed; stops the chain on failure
  local name=$1; shift
  local started=$(date +%s)
  echo "[$(date -u +%H:%M:%S)] >>> $name" | tee -a "$OUT/run.log"
  set +e; "$@" >"$OUT/$name.log" 2>&1; local code=$?; set -e
  printf '%s\n' "$code" >"$OUT/$name.exit"
  echo "[$(date -u +%H:%M:%S)] <<< $name exit=$code $(( $(date +%s) - started ))s (total $(( $(date +%s) - T0 ))s)" | tee -a "$OUT/run.log"
  [ "$code" -eq 0 ] || { echo "stage $name failed; see $OUT/$name.log" | tee -a "$OUT/run.log"; exit "$code"; }
}

cd "$CODE"
sha=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
$PY_PROJECT - "$OUT" "$ARCHIVE" "$sha" "$PROFILE" "$WEIGHTS" "$CONNECTIVITY" "$EPOCHS" "$BANK_RUNS" "$SKIP_BANK" "$SKIP_FINETUNE" <<'PY'
import json, subprocess, sys
out, archive, sha, profile, weights, conn, epochs, runs, skip_bank, skip_ft = sys.argv[1:]
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"schema": "timesoil.case-z-run/v1", "archive": archive, "archive_sha256": sha,
           "profile": profile, "initial_weights": weights, "connectivity": conn, "code_commit": commit,
           "epochs": int(epochs), "bank_runs": int(runs), "skip_bank": skip_bank == "1",
           "skip_finetune": skip_ft == "1", "final_opm_calls_allowed": 1},
          open(out + "/protocol.json", "w"), indent=2)
PY

# 1. Intake: inspect the archive, build the incumbent request under the case profile.
stage intake $PY_PROJECT scripts/intake_case_z.py build-request "$ARCHIVE" \
  --cut 2006-12-31 --start 2007-01-01 --end 2025-09-01 --profile "$PROFILE" --output "$OUT/intake"

# 2. Baseline physics of the incumbent (one OPM run; this is the paired base for the audit).
stage baseline $PY_PROJECT -m timesoil.aios.cli full-cycle "$OUT/intake/request.json" \
  --runs-dir "$OUT/baseline" --run-id incumbent --timeout 7200

# 3. Scenario bank on the new case (optional; ~4 min per run at 16 MPI, two workers).
if [ "$SKIP_BANK" -eq 0 ]; then
  stage bank $PY_PROJECT scripts/build_feasible_bank.py "$OUT/intake/request.json" \
    "$OUT/baseline/incumbent/canonical" --profile "$PROFILE" --runs "$BANK_RUNS" --output "$OUT/bank"
  stage bank-run bash scripts/launch_feasible_bank.sh "$OUT/bank" "$OUT/bank-runs"
fi

# 4. Short fine-tune from the 60-epoch weights on the new bank (optional).
HEAD=$WEIGHTS; HEAD_REPORT=$WEIGHTS_REPORT
if [ "$SKIP_BANK" -eq 0 ] && [ "$SKIP_FINETUNE" -eq 0 ]; then
  stage finetune taskset -c "$CPUS" $PY_TORCH scripts/finetune_timesfm_head.py \
    --batch "$OUT/bank-runs" --batch-sha256 "$(sha256sum "$OUT/bank-runs/manifest.json" | cut -d' ' -f1)" \
    --connectivity "$CONNECTIVITY" --initial-head "$WEIGHTS" \
    --initial-head-sha256 "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" \
    --output "$OUT/training" --epochs "$EPOCHS" --learning-rate 1e-5 \
    --unfreeze-backbone --condition-last-layer --condition-first-layer \
    --cold-start-normalization --retain-initial-scale --economic-targets --precise-variate-softmax
  HEAD=$OUT/training/full-model.pt; HEAD_REPORT=$OUT/training/report.json
fi

# 5. Search under the profile: forecast → official calculator → gates → seal (zero OPM calls).
stage search taskset -c "$CPUS" $PY_TORCH scripts/propose_track2_policies.py \
  "$OUT/baseline/incumbent" "$OUT/intake/request.json" "$OUT/search" \
  --rounds 3 --economic-selection --case-profile "$PROFILE" \
  --head "$HEAD" --head-sha256 "$(sha256sum "$HEAD" | cut -d' ' -f1)" --head-report "$HEAD_REPORT" \
  --connectivity "$CONNECTIVITY" --gpu-memory-fraction 0.5

# 6. Exactly one final OPM of the sealed graph, paired audit against the incumbent.
SEAL=$(sha256sum "$OUT/search/selection-before-opm.json" | cut -d' ' -f1)
stage final $PY_PROJECT scripts/track2_final_selection.py "$OUT/search" "$OUT/baseline/incumbent" \
  "$OUT/final" --seal-sha256 "$SEAL"

# 7. Interpretability report over the run (never touches the sealed artifacts).
stage explain $PY_PROJECT scripts/explain_run.py "$OUT/search" "$OUT/final" --output "$OUT/interpretability"

echo "DONE $OUT" | tee -a "$OUT/run.log"
