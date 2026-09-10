# TimesOil · AIOS, team Scorp

Guidance for Claude Code when working in this repository. Machine-level wiring —
MCP servers, plugins, hooks, skills, tool versions and the shared working style —
is not committed here because this repository is public; it lives in
`~/.claude/CLAUDE.md` and in the untracked `CLAUDE.local.md` next to this file.

## Project purpose

TimesOil is team Scorp's entry for the AIOS hackathon (Blue Sky Research ×
Tatneft). The goal is a **≥15% gain in official NPV (ЧДД), proven separately on
Model Y (Track 1) and Model Z (Track 2)** over the complete management period
and the complete well stock, computed by the organizers' Python calculator on a
paired baseline/candidate simulation.

- **Track 1 — Model Y**: a monthly physical MPC. Agents propose controls for the
  whole field, a deterministic validator checks them, OPM Flow simulates the full
  remaining period for the incumbent and the proposal, the official calculator
  ranks them, and only the current month is committed. No surrogate takes part in
  the working controller.
- **Track 2 — Model Z (this branch)**: Google TimesFM 3.0 forecasts the response
  to candidate controls, the official calculator scores each forecast, the best
  eligible schedule is sealed, and then **exactly one** final OPM verification
  runs. Choosing another schedule after seeing that OPM result is forbidden.
- **Agents**: four AIOS roles — coordinator, reservoir engineer, planner, critic —
  on Tatneft `qwen3.8-27b` (OpenAI-compatible endpoint). LLM text never decides
  feasibility or gain; the validator, the simulator and the calculator do.

Numbers move fast in this project. Read
`docs/HANDOFF_CLAUDE_CODE_20260910.md` and
`docs/BOTH_TRACKS_ACCEPTANCE_20260909.md` before quoting any result, and treat
this file as stale wherever it disagrees with them.

## Branches

The GitHub remote `git@github.com:DarkPrinceWarrior/TimesOil.git` carries three
branches on purpose: `main` (formal, the branch point of Track 2),
`track-1-model-y` and `track-2-model-z`. Retired lines survive as `archive/*`
tags. This checkout, `/home/ruslan_safaev/TimesOil`, is `track-2-model-z` — the
active track. Track 1 lives in
`/home/ruslan_safaev/TimesOil-audit-track1-water-20260910`.

Do not merge the two tracks, do not rewrite published history, and update
branches fast-forward only.

## Repository layout

```text
src/timesoil/aios/   workflow.py (MPC cycle), opm.py + opm_chdd.py (OPM Flow and
                     canonical export), economics.py (official calculator
                     adapter), operating_constraints.py, schedule*.py,
                     surrogate.py, agents.py + llm.py (Qwen), api.py, cli.py, ui.py
scripts/             Track 2 chain: propose_track2_policies.py (search),
                     track2_final_selection.py (seal + single final OPM),
                     timesfm_economics.py (nine economic outputs),
                     finetune_timesfm_head.py, evaluate_timesfm_scenarios.py,
                     timesfm_geology.py, export_opm_connectivity.py,
                     generate_track2_scenarios.py, run_track2_scenarios.py
tests/               contract and numeric checks; need PYTHONPATH=src:scripts
docs/                algorithm, acceptance matrix, control/cost gaps, runbook,
                     handoff; docs/hackathon/chdd/CHDD_PYTHON is the organizers'
                     NPV calculator and must never be edited
deliverables/        sealed evidence bundles cited by the acceptance matrix
```

`deliverables/` is evidence, not scratch space: do not delete, rewrite or
"tidy" a bundle. Never committed and kept locally only: `raw_data/`,
`results/`, `docs/hackathon/models`, `docs/hackathon/sources`, `secrets/`,
`.codegraph/`.

## Where computation happens

Edit in WSL. Run **every** scientific computation, real-data test, OPM run and
GPU job on A100 — `ssh -o BatchMode=yes -o ConnectTimeout=10 a100-remote`
(`a100` is the office-LAN alias for the same host).

```text
repo on A100        /root/projects/TimesOil
results root        R=/root/projects/TimesOil/results/audit-20260909
project venv        /root/projects/TimesOil/.venv/bin/python
Torch/TimesFM venv  /tmp/timesoil-kt3-20260908/venv/bin/python  (tmpfs: lost on reboot)
GPU                 physical GPU 5 → CUDA_VISIBLE_DEVICES=5 (cuda:0 in-process)
pinned decks        /tmp/timesoil-kt2/model_y, /tmp/timesoil-kt2/model_z (hashes in the handoff)
```

The card and the host are shared with other projects: check `nvidia-smi` before
launching and never stop another project's process or tmux session. Deliver code
through git only — commit in WSL, push, then `git pull --ff-only` or
`git worktree add --detach <SHA>` on A100. No scp of source, no editing on the
server. Long jobs run in `tmux` under a unique name with the PID, directory and
time recorded.

A local smoke check never substitutes for a server run. If only the local check
ran, say exactly that.

## Hard rules for this project

- Forecast NPV is never official NPV. Only the organizers' calculator applied to
  an OPM result is official, and a gain claim needs a comparable paired baseline.
- Track 2 sealing contract: zero OPM calls inside the search, exactly one final
  OPM, no re-selection afterwards. A sealed run is immutable — never rerun it,
  never repoint it, never reuse its physics to pick a different schedule.
- No future observations in any forecast input. Keep the frozen
  train/validation/test split; never train or select on test.
- Do not weaken a numeric guard to make something pass — FP64 variate softmax,
  `own_control_constraints`, schedule and BHP bounds, hash checks. A guard that
  refuses is a result to report, not an obstacle to route around.
- Every experiment gets a new output directory, a written protocol and hashes.
  Drivers use `mkdir(exist_ok=False)` and `open('x')` on purpose; do not "fix"
  that by overwriting an existing run.
- Never change physics, data splits or frozen weights silently. State what was
  found, what will change and why before editing.
- Secrets: the Tatneft API key lives in a runtime file on A100 — probe it with
  `test -s`, never `cat`. Never print, commit or bundle it.
- Report outcomes verbatim. A non-zero exit, a rejected audit or an NPV below
  target is the finding; do not present a partial or historical result as a
  completed one.

## Environment and task gate

`uv` with Python 3.13; never bare `pip`.

```bash
uv sync --locked
uv run python -m compileall -q src scripts
PYTHONPATH=src:scripts uv run pytest tests/<targeted files> -q
git diff --check
```

After a numeric or model change, additionally run the profile tests covering it
on A100 with the project venv, and record the metrics and receipt hashes in the
matching `docs/` or `deliverables/` entry. Large artifacts stay on A100.

## Conventions

- `from __future__ import annotations`, type hints, `X | Y` unions.
- Reports and documents in Russian with LaTeX formulas; code, comments and
  commit messages in English.
- Targeted edits, unrelated work preserved; no renames or public-interface
  changes unless the task asks for them.
- The web surface (`src/timesoil/aios/ui.py`, `api.py`) is part of the required
  submission: if you change it, verify in the browser with Playwright and keep
  screenshot evidence for the affected states. Numeric work needs no browser
  pass.
