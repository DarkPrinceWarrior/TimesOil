#!/usr/bin/env bash
# Template only: run the feasible-regime bank through timesoil-aios full-cycle on A100.
# It is NOT executed from WSL and it never runs OPM here. Deliver it through git,
# then launch it inside tmux on the server. Every launch needs a NEW results directory.
#
#   BANK=/root/projects/TimesOil/results/audit-20260909/feasible-bank-YYYYMMDD
#   scripts/launch_feasible_bank.sh "$BANK" "$BANK/runs"
#
# Two workers pinned to disjoint cpusets, 16 MPI x 1 thread each, as in
# scripts/run_full_period_sweep.py. Check nvidia-smi/htop and the other project's
# tmux sessions before launching: cores 14-29 and 32-47 are shared hardware.

set -euo pipefail

BANK_DIR=${1:?usage: launch_feasible_bank.sh BANK_DIR RESULTS_DIR [PYTHON]}
RESULTS_DIR=${2:?usage: launch_feasible_bank.sh BANK_DIR RESULTS_DIR [PYTHON]}
PYTHON=${3:-/root/projects/TimesOil/.venv/bin/python}
AFFINITIES=("${WORKER_0_CPUS:-14-29}" "${WORKER_1_CPUS:-32-47}")
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-7200}

export OPM_MPI_PROCESSES=16
export OPM_THREADS_PER_PROCESS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

[[ -f "$BANK_DIR/manifest.json" ]] || { echo "no bank manifest in $BANK_DIR" >&2; exit 2; }
mkdir "$RESULTS_DIR"                      # deliberately fails if the directory exists
mkdir "$RESULTS_DIR/exit" "$RESULTS_DIR/logs"

# Requests to run: every family except F6, which points at already-computed exports.
mapfile -t REQUESTS < <("$PYTHON" - "$BANK_DIR/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
for entry in manifest["scenarios"]:
    if entry.get("request"):
        print(f"{entry['id']}\t{entry['request']}")
PY
)
[[ ${#REQUESTS[@]} -gt 0 ]] || { echo "bank has no runnable requests" >&2; exit 2; }

"$PYTHON" - "$BANK_DIR" "$RESULTS_DIR" "${#REQUESTS[@]}" <<'PY' > "$RESULTS_DIR/protocol.json"
import hashlib, json, os, subprocess, sys, time
bank, results, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
digest = hashlib.sha256(open(f"{bank}/manifest.json", "rb").read()).hexdigest()
json.dump({
    "schema": "timesoil.feasible-bank-run/v1",
    "bank_dir": os.path.abspath(bank),
    "bank_manifest_sha256": digest,
    "results_dir": os.path.abspath(results),
    "scenario_count": count,
    "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
    "opm": {"mpi_processes": 16, "threads_per_process": 1,
            "cpu_affinity": [os.environ.get("WORKER_0_CPUS", "14-29"),
                             os.environ.get("WORKER_1_CPUS", "32-47")]},
    "selection_performed": False,
    "forecast_used": False,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, sys.stdout, indent=2)
sys.stdout.write("\n")
PY

run_one() {
  local index=$1 entry=$2 worker=$3
  local id=${entry%%$'\t'*} request=${entry##*$'\t'}
  local cpus=${AFFINITIES[$worker]}
  OPM_CPU_AFFINITY="$cpus" taskset -c "$cpus" \
    "$PYTHON" -m timesoil.aios.cli full-cycle "$BANK_DIR/$request" \
      --runs-dir "$RESULTS_DIR/cycles" --run-id "$id" --timeout "$TIMEOUT_SECONDS" \
      > "$RESULTS_DIR/logs/$id.log" 2>&1 && code=0 || code=$?
  printf '%s\n' "$code" > "$RESULTS_DIR/exit/$id"
  printf '%s worker=%s cpus=%s exit=%s\n' "$id" "$worker" "$cpus" "$code"
}

for ((i = 0; i < ${#REQUESTS[@]}; i += 2)); do
  run_one "$i" "${REQUESTS[$i]}" 0 &
  first=$!
  second=
  if (( i + 1 < ${#REQUESTS[@]} )); then
    run_one "$((i + 1))" "${REQUESTS[$((i + 1))]}" 1 &
    second=$!
  fi
  wait "$first" || true
  [[ -n $second ]] && { wait "$second" || true; }
done

# A non-zero exit file is the finding, not a reason to rerun: report it verbatim.
grep -rLx 0 "$RESULTS_DIR/exit" > "$RESULTS_DIR/failed.txt" || true
printf 'bank complete: %s\n' "$RESULTS_DIR"
