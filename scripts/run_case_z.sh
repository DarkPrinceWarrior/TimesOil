#!/usr/bin/env bash
# End-to-end Track 2 run for the organizers' test case on A100.
#
# One new results directory, one protocol, an exit file and an elapsed time per stage,
# and a wall-clock budget (default 4 h, the 17:00-21:00 window). Every stage refuses to
# overwrite an existing directory; a failed stage stops the chain with a non-zero exit so
# the receipt shows exactly where it stopped. Nothing here weakens a gate: the case profile
# is the only source of limits, the search makes zero OPM calls, and the sealed selection
# still gets exactly one final OPM run that is never re-selected afterwards.
#
# Usage (from the pinned code worktree on A100):
#   scripts/run_case_z.sh ARCHIVE.zip [options]
#
#   --plan-a              fast path: no bank, no fine-tune (~50 min); the search runs on the
#                         frozen 60-epoch head
#   --skip-bank           no scenario bank (implies --skip-finetune)
#   --skip-finetune       build the bank but keep the frozen head
#   --bank-runs N         OPM runs taken from the bank, round-robin over families (default 16)
#   --epochs N            fine-tune epochs (default 20)
#   --evaluate            run evaluate_timesfm_scenarios.py (OFF by default: it needs a
#                         5/3 calibration/test split that the case bank does not have)
#   --extend-schedule     pass through to intake when the case schedule stops at the cut
#   --search-seconds N    CMA-ES budget inside the search (default 900)
#   --budget-seconds N    wall-clock budget used for the remaining-time print (default 14400)
#   --dry-run             print every command, run nothing, create nothing
#
# Required environment (see docs/CASE_INTAKE_20260911.md):
#   LLM route is picked up from the runtime key files (test -s, never cat): Tatneft is the
#   primary route, Cerebras the fallback the client retries once on after a primary failure.
#   Overridable: R, PY_PROJECT, PY_TORCH, PROFILE, WEIGHTS, CONNECTIVITY, CPUS,
#   TATNEFT_KEY, CEREBRAS_KEY,
#   OPM_MPI_PROCESSES, OPM_THREADS_PER_PROCESS, OPM_CPU_AFFINITY.
set -euo pipefail

ARCHIVE=${1:?archive path required}; shift || true
SKIP_BANK=0; SKIP_FINETUNE=0; EVALUATE=0; EPOCHS=20; BANK_RUNS=16
EXTEND_SCHEDULE=0; SEARCH_SECONDS=900; BUDGET_SECONDS=14400; DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --plan-a) SKIP_BANK=1; SKIP_FINETUNE=1;;
    --skip-bank) SKIP_BANK=1; SKIP_FINETUNE=1;;
    --skip-finetune) SKIP_FINETUNE=1;;
    --evaluate) EVALUATE=1;;
    --extend-schedule) EXTEND_SCHEDULE=1;;
    --epochs) EPOCHS=$2; shift;;
    --bank-runs) BANK_RUNS=$2; shift;;
    --search-seconds) SEARCH_SECONDS=$2; shift;;
    --budget-seconds) BUDGET_SECONDS=$2; shift;;
    --dry-run) DRY_RUN=1;;
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
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-$R/case-z-$STAMP}

# The assembler turns the bank's OPM cycles into a training batch; without it the bank and
# the fine-tune cannot be used, and finding that out after 45 minutes of OPM is too late.
if [ "$SKIP_FINETUNE" -eq 0 ] && [ ! -f "$CODE/scripts/assemble_scenario_batch.py" ]; then
  echo "scripts/assemble_scenario_batch.py is missing: run with --plan-a or --skip-finetune" >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 0 ]; then
  test ! -e "$OUT"; mkdir "$OUT"
fi

# LLM route, unless LLM_BASE_URL is already set: Tatneft first (zero failures on 11.09),
# Cerebras as the fallback route the client retries once after a primary LLMError.
# Keys stay in their files (test -s, never printed); the fallback key is passed by path.
TATNEFT_KEY=${TATNEFT_KEY:-/dev/shm/timesoil-tatneft-20260909-key}
CEREBRAS_KEY=${CEREBRAS_KEY:-/root/.config/timesoil/cerebras-key}
if [ -z "${LLM_BASE_URL:-}" ]; then
  if test -s "$TATNEFT_KEY"; then
    export LLM_BASE_URL=https://litellm.tatneft.guru/v1 LLM_MODEL=qwen3.8-27b
    export LLM_API_KEY="$(cat "$TATNEFT_KEY")"
    # A proxy left over in the shell (this script's own Cerebras branch exports one) would
    # route every Tatneft call through the Cerebras egress and fail it.
    unset LLM_PROXY_URL
    if test -s "$CEREBRAS_KEY"; then
      export LLM_FALLBACK_BASE_URL=https://api.cerebras.ai/v1 LLM_FALLBACK_MODEL=qwen-3.8-27b
      export LLM_FALLBACK_API_KEY_FILE="$CEREBRAS_KEY"
      # api.cerebras.ai is geo-blocked from A100; the primary route must not use the proxy.
      export LLM_FALLBACK_PROXY_URL="${LLM_FALLBACK_PROXY_URL:-http://127.0.0.1:10809}"
    fi
  elif test -s "$CEREBRAS_KEY"; then
    export LLM_BASE_URL=https://api.cerebras.ai/v1 LLM_MODEL=qwen-3.8-27b
    export LLM_API_KEY="$(cat "$CEREBRAS_KEY")"
    export LLM_PROXY_URL="${LLM_PROXY_URL:-http://127.0.0.1:10809}"
  else
    echo "no LLM key file found (tatneft or cerebras)" >&2; exit 3
  fi
fi
# Determinism fields apply to whichever route answers; Cerebras reads both, Tatneft ignores them.
export LLM_REASONING_EFFORT=${LLM_REASONING_EFFORT:-high} LLM_SEED=${LLM_SEED:-20260909}
export LLM_TIMEOUT_SECONDS=${LLM_TIMEOUT_SECONDS:-600} LLM_MAX_OUTPUT_TOKENS=${LLM_MAX_OUTPUT_TOKENS:-8192}
export LLM_CALL_LOG=${LLM_CALL_LOG:-$OUT/llm_calls.jsonl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$CODE/src:$CODE/scripts
export OPM_MPI_PROCESSES=${OPM_MPI_PROCESSES:-16} OPM_THREADS_PER_PROCESS=${OPM_THREADS_PER_PROCESS:-1}
export OPM_CPU_AFFINITY=${OPM_CPU_AFFINITY:-30-45}
CPUS=${CPUS:-14-29}
T0=$(date +%s)

log() {
  if [ "$DRY_RUN" -eq 1 ]; then echo "# $*"; else echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; fi
}

sha_of() {  # sha_of FILE — a placeholder under --dry-run, the real digest otherwise
  if [ "$DRY_RUN" -eq 1 ]; then printf '<sha256:%s>' "$1"; else sha256sum "$1" | cut -d' ' -f1; fi
}

budget() {  # budget LABEL — remaining wall clock before an optional stage
  local left=$(( BUDGET_SECONDS - ($(date +%s) - T0) ))
  log "budget before $1: $(( left / 60 )) min $(( left % 60 ))s of $(( BUDGET_SECONDS / 60 )) min remaining"
  [ "$left" -gt 0 ] || log "WARNING: budget exhausted; consider stopping after the current stage"
}

# Key outputs of the next stage, hashed into protocol.json. Reset by every stage() call.
OUTPUTS=()

record() {  # record NAME EXIT SECONDS FILE... — append a stage entry to protocol.json
  "$PY_PROJECT" - "$OUT/protocol.json" "$@" <<'PY'
import hashlib, json, sys, time
path, name, code, seconds, *files = sys.argv[1:]
document = json.load(open(path))
outputs = {}
for item in files:
    try:
        outputs[item] = hashlib.sha256(open(item, "rb").read()).hexdigest()
    except OSError as error:
        outputs[item] = f"unavailable: {error}"
document.setdefault("stages", []).append({
    "stage": name, "exit": int(code), "seconds": int(seconds),
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "outputs": outputs})
json.dump(document, open(path, "w"), indent=2)
PY
}

stage() {  # stage NAME COMMAND... — log, exit file, elapsed, protocol; stops the chain on failure
  local name=$1; shift
  local outputs=(); [ ${#OUTPUTS[@]} -gt 0 ] && outputs=("${OUTPUTS[@]}"); OUTPUTS=()
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%s: ' "$name"; printf '%q ' "$@"; printf '\n'
    return 0
  fi
  local started=$(date +%s)
  log ">>> $name"
  set +e; "$@" >"$OUT/$name.log" 2>&1; local code=$?; set -e
  printf '%s\n' "$code" >"$OUT/$name.exit"
  local elapsed=$(( $(date +%s) - started ))
  record "$name" "$code" "$elapsed" "${outputs[@]}"
  log "<<< $name exit=$code ${elapsed}s (total $(( $(date +%s) - T0 ))s)"
  [ "$code" -eq 0 ] || { log "stage $name failed; see $OUT/$name.log"; exit "$code"; }
}

cd "$CODE"
ARCHIVE_SHA=$(sha_of "$ARCHIVE")
# Every gate pins to this archive. Exported before any python runs; re-exported after intake,
# because --extend-schedule makes the request point at a rewritten archive with its own hash.
export TIMESOIL_CASE_SOURCE_SHA256=$ARCHIVE_SHA

# The search flags C1 adds are probed once instead of assumed: a frozen driver must not die
# on an unknown option, and the log has to say which search actually ran.
SEARCH_EXTRA=()
SEARCH_HELP=$([ "$DRY_RUN" -eq 1 ] && echo "--search --blocks --llm-round0" \
              || "$PY_TORCH" scripts/propose_track2_policies.py --help 2>/dev/null || true)
case "$SEARCH_HELP" in *--search*) SEARCH_EXTRA+=(--search cma --search-seconds "$SEARCH_SECONDS");; esac
case "$SEARCH_HELP" in *--llm-round0*) SEARCH_EXTRA+=(--llm-round0);; esac
case "$SEARCH_HELP" in *--blocks*) SEARCH_EXTRA+=(--blocks "$OUT/blocks.json");; esac

if [ "$DRY_RUN" -eq 0 ]; then
  "$PY_PROJECT" - "$OUT" "$ARCHIVE" "$ARCHIVE_SHA" "$PROFILE" "$WEIGHTS" "$EPOCHS" "$BANK_RUNS" \
      "$SKIP_BANK" "$SKIP_FINETUNE" "$EVALUATE" "${SEARCH_EXTRA[*]:-none}" <<'PY'
import hashlib, json, subprocess, sys
out, archive, sha, profile, weights, epochs, runs, skip_bank, skip_ft, evaluate, search = sys.argv[1:]
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
json.dump({"schema": "timesoil.case-z-run/v1", "archive": archive, "archive_sha256": sha,
           "profile": profile, "profile_sha256": hashlib.sha256(open(profile, "rb").read()).hexdigest(),
           "initial_weights": weights, "code_commit": commit, "code_dirty": bool(dirty),
           "epochs": int(epochs), "bank_runs": int(runs), "skip_bank": skip_bank == "1",
           "skip_finetune": skip_ft == "1", "evaluation": evaluate == "1", "search_flags": search,
           "search_opm_calls": 0, "final_opm_calls_allowed": 1,
           "reselection_after_opm_allowed": False, "stages": []},
          open(out + "/protocol.json", "w"), indent=2)
PY
  log "archive $ARCHIVE sha256=$ARCHIVE_SHA -> $OUT"
  log "search flags: ${SEARCH_EXTRA[*]:-baseline grid + agent rounds only}"
fi

# 1. Intake: inspect the archive, build the incumbent request under the case profile.
INTAKE_FLAGS=()
[ "$EXTEND_SCHEDULE" -eq 1 ] && INTAKE_FLAGS=(--extend-schedule)
OUTPUTS=("$OUT/intake/request.json" "$OUT/intake/manifest.json")
stage intake "$PY_PROJECT" scripts/intake_case_z.py build-request "$ARCHIVE" \
  --cut 2006-12-31 --start 2007-01-01 --end 2025-09-01 --profile "$PROFILE" \
  --scenario-id baseline --output "$OUT/intake" "${INTAKE_FLAGS[@]}"
# The incumbent is a reference, not a submission: its physics is run without the profile
# rules (a case's source schedule need not satisfy the case limits), while every candidate
# and the final full-cycle keep the rules embedded in intake/request.json.
OUTPUTS=("$OUT/intake-incumbent/request.json")
# With --extend-schedule the first intake writes case-extended.zip; the second intake reads
# that very archive so both requests pin the same source hash.
INCUMBENT_ARCHIVE=$ARCHIVE; INCUMBENT_FLAGS=("${INTAKE_FLAGS[@]}")
if [ "$EXTEND_SCHEDULE" -eq 1 ] && [ -f "$OUT/intake/case-extended.zip" ]; then
  INCUMBENT_ARCHIVE=$OUT/intake/case-extended.zip; INCUMBENT_FLAGS=()
fi
stage intake-incumbent "$PY_PROJECT" scripts/intake_case_z.py build-request "$INCUMBENT_ARCHIVE" \
  --cut 2006-12-31 --start 2007-01-01 --end 2025-09-01 --profile "$PROFILE" --rules none \
  --scenario-id baseline --output "$OUT/intake-incumbent" "${INCUMBENT_FLAGS[@]}"
INCUMBENT_REQUEST=$OUT/intake-incumbent/request.json

# The request is what every later gate is pinned to, so the hash comes from the request.
if [ "$DRY_RUN" -eq 0 ]; then
  REQUEST_SOURCE_SHA=$("$PY_PROJECT" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["request"]["source_sha256"])' \
    "$OUT/intake/manifest.json")
  if [ "$REQUEST_SOURCE_SHA" != "$ARCHIVE_SHA" ]; then
    log "intake rewrote the archive (--extend-schedule): pinning gates to $REQUEST_SOURCE_SHA"
  fi
  export TIMESOIL_CASE_SOURCE_SHA256=$REQUEST_SOURCE_SHA
fi

# 2. Baseline physics of the incumbent (one OPM run; this is the paired base for the audit).
OUTPUTS=("$OUT/baseline/incumbent/manifest.json" "$OUT/baseline/incumbent/canonical/chdd.csv")
stage baseline "$PY_PROJECT" -m timesoil.aios.cli full-cycle "$INCUMBENT_REQUEST" \
  --runs-dir "$OUT/baseline" --run-id incumbent --timeout 7200

# 3. Connectivity of the case grid, from the authenticated INIT/EGRID of that very run.
OUTPUTS=("$OUT/connectivity/connectivity.json")
stage connectivity "$PY_PROJECT" scripts/export_opm_connectivity.py \
  "$OUT/baseline/incumbent" "$OUT/connectivity"
CONNECTIVITY=${CONNECTIVITY:-$OUT/connectivity/connectivity.json}

# 4. Geometry blocks for the block agents, the bank's F5 family and the interpretability report.
OUTPUTS=("$OUT/blocks.json")
stage blocks "$PY_PROJECT" scripts/export_blocks.py "$ARCHIVE" "$CONNECTIVITY" \
  "$OUT/blocks.json" --blocks 6

# 5. Scenario bank on the new case, then its OPM runs and the training batch (optional).
BATCH=
if [ "$SKIP_BANK" -eq 0 ]; then
  budget "bank (~$(( BANK_RUNS * 4 / 2 + 8 )) min of OPM on two workers)"
  OUTPUTS=("$OUT/bank/manifest.json")
  # The bank wants the calculator rows of the control months only: the canonical export is
  # labelled by report date and spans the history, the calculator input is shifted back a
  # month and spans the history too, so the management-period rows are cut out here.
  OUTPUTS=("$OUT/bank-rows.csv")
  stage bank-rows "$PY_PROJECT" - "$OUT/baseline/incumbent/economics-2007/input.csv" \
      "$INCUMBENT_REQUEST" "$OUT/bank-rows.csv" <<'ROWS'
import csv, json, sys
source, request, target = sys.argv[1:]
months = {action["month"][:10] for action in json.load(open(request))["controls"]}
with open(source, newline="") as inp, open(target, "w", newline="") as out:
    reader = csv.DictReader(inp)
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    kept = 0
    for row in reader:
        if row["DATA"][:10] in months:
            writer.writerow(row); kept += 1
print(f"bank rows: {kept} of the management period, {len(months)} months")
ROWS
  OUTPUTS=("$OUT/bank/manifest.json")
  stage bank "$PY_PROJECT" scripts/build_feasible_bank.py \
    --request "$INCUMBENT_REQUEST" \
    --canonical "$OUT/bank-rows.csv" \
    --export-manifest "$OUT/baseline/incumbent/canonical/manifest.json" \
    --output "$OUT/bank" --blocks "$OUT/blocks.json" \
    --injection-cap-m3d 600 --liquid-cap-m3d 600 --injection-basis cap

  # The full bank is ~34 OPM runs; the window holds about half. Pick round-robin over the
  # families so every regime family survives the trim, and keep the full bank next to it.
  OUTPUTS=("$OUT/bank-subset/manifest.json")
  stage bank-subset "$PY_PROJECT" - "$OUT/bank" "$OUT/bank-subset" "$BANK_RUNS" <<'PY'
import json, shutil, sys
from pathlib import Path
source, target, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
manifest = json.loads((source / "manifest.json").read_text())
runnable = [entry for entry in manifest["scenarios"] if entry.get("request")]
families: dict[str, list] = {}
for entry in runnable:
    families.setdefault(entry["family"], []).append(entry)
picked = []
while len(picked) < limit and any(families.values()):
    for family in list(families):
        if families[family] and len(picked) < limit:
            picked.append(families[family].pop(0))
kept = {entry["id"] for entry in picked}
manifest["scenarios"] = [entry for entry in manifest["scenarios"]
                         if entry["id"] in kept or not entry.get("request")]
manifest["subset_of"] = str(source.resolve())
manifest["subset_limit"] = limit
target.mkdir()                                    # deliberately fails if it already exists
for entry in picked:
    destination = target / entry["request"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / entry["request"], destination)
(target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"kept": sorted(kept), "count": len(kept)}), flush=True)
PY

  stage bank-run bash scripts/launch_feasible_bank.sh "$OUT/bank-subset" "$OUT/bank-runs" "$PY_PROJECT"

  if [ "$SKIP_FINETUNE" -eq 0 ]; then
    # assemble_scenario_batch.py never relocates a run (the summary extraction records the
    # absolute bind mount of <run>/output), so the batch is the directory the bank cycles were
    # executed into, and the incumbent has to be executed there too. It cannot be the run from
    # stage 2: that directory has to exist before the bank is built. The repeat costs one OPM
    # run of the same deck and the same controls; the audit pair stays stage 2's run.
    BATCH=$OUT/bank-runs/cycles
    OUTPUTS=("$BATCH/baseline/full-cycle-receipt.json")
    stage batch-baseline "$PY_PROJECT" -m timesoil.aios.cli full-cycle \
      "$INCUMBENT_REQUEST" --runs-dir "$BATCH" --run-id baseline --timeout 7200
    OUTPUTS=("$BATCH/manifest.json")
    stage assemble "$PY_PROJECT" scripts/assemble_scenario_batch.py "$BATCH"
  fi
fi

# 6. Short fine-tune from the frozen 60-epoch head on the case batch (optional).
HEAD=$WEIGHTS; HEAD_REPORT=$WEIGHTS_REPORT
# Reused weights are conditioned on the geology file they were trained with, and the search
# pins the head report to that file by hash. The case export must carry the same static
# features, weights and well ids (only provenance may differ); otherwise the head does not
# know this case and Plan A is refused — fine-tune on the case bank instead.
SEARCH_CONNECTIVITY=$CONNECTIVITY
if [ -z "$BATCH" ] && [ "$DRY_RUN" -eq 0 ]; then
  HEAD_CONNECTIVITY=${HEAD_CONNECTIVITY:-$R/static-head-geology-20260909/model-z/connectivity.json}
  "$PY_PROJECT" - "$HEAD_REPORT" "$HEAD_CONNECTIVITY" "$CONNECTIVITY" <<'GEO'
import hashlib, json, sys
report, head_file, case_file = sys.argv[1:]
expected = json.load(open(report))["connectivity_sha256"]
actual = hashlib.sha256(open(head_file, "rb").read()).hexdigest()
if actual != expected:
    sys.exit(f"head connectivity {head_file} sha256 {actual} != report {expected}")
head, case = json.load(open(head_file)), json.load(open(case_file))
for key in ("well_ids", "static", "weights"):
    if head.get(key) != case.get(key):
        sys.exit(f"case geology differs from the head's geology in '{key}': reuse refused, fine-tune on the case bank")
print("case geology matches the head's geology (well_ids, static, weights)")
GEO
  SEARCH_CONNECTIVITY=$HEAD_CONNECTIVITY
  # The voucher lets the search accept the head's file although its provenance names the
  # training deck; the content equality above is what justifies it (recorded in the receipt).
  export TIMESOIL_HEAD_GEOLOGY_VERIFIED_SHA256=$(sha_of "$HEAD_CONNECTIVITY")
  log "search uses the head's geology file $HEAD_CONNECTIVITY (content verified against the case export)"
fi
if [ -n "$BATCH" ]; then
  budget "fine-tune ($EPOCHS epochs on GPU ${CUDA_VISIBLE_DEVICES})"
  OUTPUTS=("$OUT/training/full-model.pt" "$OUT/training/report.json")
  stage finetune taskset -c "$CPUS" "$PY_TORCH" scripts/finetune_timesfm_head.py \
    --batch "$BATCH" --batch-sha256 "$(sha_of "$BATCH/manifest.json")" \
    --connectivity "$CONNECTIVITY" --initial-head "$WEIGHTS" \
    --initial-head-sha256 "$(sha_of "$WEIGHTS")" \
    --output "$OUT/training" --epochs "$EPOCHS" --learning-rate 1e-5 \
    --unfreeze-backbone --condition-last-layer --condition-first-layer \
    --cold-start-normalization --retain-initial-scale --economic-targets --precise-variate-softmax
  HEAD=$OUT/training/full-model.pt; HEAD_REPORT=$OUT/training/report.json

  # 6b. Forecast accuracy on the frozen holdout. OFF by default: the case bank is one batch
  # without the 5/3 calibration/test split this driver requires, so it would refuse.
  if [ "$EVALUATE" -eq 1 ]; then
    budget "evaluation"
    OUTPUTS=("$OUT/evaluation/report.json")
    stage evaluate taskset -c "$CPUS" "$PY_TORCH" scripts/evaluate_timesfm_scenarios.py \
      --batch "$BATCH" --batch-sha256 "$(sha_of "$BATCH/manifest.json")" \
      --head "$HEAD" --head-report "$HEAD_REPORT" --connectivity "$CONNECTIVITY" \
      --output "$OUT/evaluation" --test-only --gpu-memory-fraction 0.5
  fi
fi

# 7. Search under the profile: forecast -> official calculator -> gates -> seal. Zero OPM calls.
budget "search"
OUTPUTS=("$OUT/search/candidates.json" "$OUT/search/proposal-receipt.json" \
         "$OUT/search/selection-before-opm.json")
stage search taskset -c "$CPUS" "$PY_TORCH" scripts/propose_track2_policies.py \
  "$OUT/baseline/incumbent" "$OUT/intake/request.json" "$OUT/search" \
  --rounds 3 --economic-selection --case-profile "$PROFILE" \
  --head "$HEAD" --head-sha256 "$(sha_of "$HEAD")" --head-report "$HEAD_REPORT" \
  --connectivity "$SEARCH_CONNECTIVITY" --gpu-memory-fraction 0.5 "${SEARCH_EXTRA[@]}"

# 8. Exactly one final OPM of the sealed graph, paired audit against the incumbent.
SEAL=$(sha_of "$OUT/search/selection-before-opm.json")
log "selection seal sha256=$SEAL"
OUTPUTS=("$OUT/final/audit.json" "$OUT/search/final-verification-attempt.json")
stage final "$PY_PROJECT" scripts/track2_final_selection.py "$OUT/search" \
  "$OUT/baseline/incumbent" "$OUT/final" --seal-sha256 "$SEAL"

# 9. Interpretability report over the finished run (never touches the sealed artifacts).
OUTPUTS=("$OUT/explain/interpretability/report.json")
stage explain "$PY_PROJECT" scripts/explain_run.py "$OUT/final" "$OUT/explain" \
  --profile "$PROFILE" --blocks "$OUT/blocks.json"

[ "$DRY_RUN" -eq 1 ] || log "DONE $OUT ($(( ($(date +%s) - T0) / 60 )) min)"
