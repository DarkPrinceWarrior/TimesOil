# Suggested Commands

## Agent startup

```bash
cd /home/ruslan_safaev/TimesOil
codex
codegraph status
```

Codex/MCP/permissions глобальные; project-level `.mcp.json` не создавать. fff, Serena и CodeGraph получают проект из cwd.

## Local environment and smoke

```bash
uv venv --python 3.13 .venv
uv sync --locked
uv sync --extra tirex
uv run python -m compileall -q src scripts
uv run python -c "import timesoil"
```

## Representative entry points

```bash
uv run python scripts/run_baselines.py
uv run python scripts/run_tirex.py
uv run python scripts/run_chronos.py
uv run python scripts/run_crm.py
uv run python scripts/run_ensemble.py
uv run python scripts/run_stacking.py
uv run python scripts/forecast_forward.py
uv run python scripts/optimize_injection.py
```

Не запускать весь модельный набор автоматически: выбрать сценарий, соответствующий изменению, и сверить его параметры/выходы.

## a100

```bash
ssh a100-remote
cd /root/projects/TimesOil
git pull --ff-only
nvidia-smi
tmux new -d -s <name> '<command>'
```

Для файлов вне git: `scp -p`. `rsync` на сервере отсутствует.