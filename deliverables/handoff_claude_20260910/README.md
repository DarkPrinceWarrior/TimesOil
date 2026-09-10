# Передача Claude Opus / Claude Code CLI

Главный документ: [HANDOFF_CLAUDE_CODE_20260910.md](../../docs/HANDOFF_CLAUDE_CODE_20260910.md).
Начальный запрос: [START_CLAUDE.txt](START_CLAUDE.txt).

Срез A100: **2026-09-10 19:06:39 UTC / 22:06:39 Москва**. Новых OPM/обучений
при подготовке передачи не запускали. Повторён только read-only аудит Y.

- `training/`: новое обучение 60 эпох завершено, checkpoint SHA в report.
  Веса остаются на A100. Development WAPE месячной нефти 11,928–12,198%; это не независимая оценка.
- `evaluation/`: закреплённая проверка девяти целей остановилась по GPU OOM.
  Протоколы и полный traceback сохранены; итоговых метрик/прогнозов нет.
- `track1/`: новый Y завершил 23 месяца, ЧДД 1597,087572 млн руб., новых переводов нет.
  Контроллер exit 0; постаудит exit 1 из-за SUMMARY `WVPT/WVIT` относительно базы.
  `audit-observation.json` описывает точный отказ; полный аудит не завершён.
- `state-snapshot.json`: текущие GPU/tmux/PID и SHA-256 21 скопированного исходного файла;
  все локальные копии сверены с хешами A100.
- `remote-sources.json`: соответствие приложений исходным абсолютным путям A100.
- `manifest.json`: хеши файлов этого комплекта, кроме самого манифеста.

`track1/wells_schedule.inc` — фрагмент управлений. `track1/full_input_schedule.inc` —
полный include, который реально прочитал последний OPM Y. Это разные артефакты.
Старый snapshot `track1_schedule_latest_13e8abec.inc` в Windows Downloads относится
к промежуточному состоянию через декабрь 2014; не заменяет полный конечный include.

Архив передачи сохраняет структуру `docs/` и `deliverables/`, включает связанные
комплекты доказательств. Секретов, checkpoint и сырых ГДМ в нём нет.
Он не заменяет доступ к Git и A100. Исторические progress-файлы остаются
снимками прежних этапов; текущий статус задаёт эта передача.

Проверка после распаковки из корня архива:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('deliverables/handoff_claude_20260910')
for item in json.loads((root/'manifest.json').read_text())['files']:
    content = (root/item['path']).read_bytes()
    assert len(content) == item['bytes'], item['path']
    assert hashlib.sha256(content).hexdigest() == item['sha256'], item['path']
print('Handoff files verified')
PY
```
