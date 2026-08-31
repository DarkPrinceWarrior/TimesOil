# Комплект поставки КТ2 и финальных заявок

Этот каталог — индекс комплекта по п. 3.4 ТЗ. Команды запуска, подтверждённые
результаты и честные границы готовности приведены в
[сценарии КТ2](../checkpoint_2_2026-08-31.md), реестр коротких квитанций — в
[индексе доказательств](../evidence/README.md), воспроизводимые параметры — в
[операторском шаблоне](../../../config/kt2.operator.example.env), общая
структура проекта — в [корневом README](../../../README.md).

## Текущее состояние

| Артефакт | Состояние | Назначение |
|---|---|---|
| [`track1_source_bound_v2_1/`](./track1_source_bound_v2_1/README.md) | Проверен | Актуальное source-bound одномесячное доказательство Model Y: выбран no-op, ЧДД `110,782383361` млн руб. Это **не финальная конкурсная заявка** и не доказательство шестимесячного улучшения. |
| [`track1_proof/`](./track1_proof/README.md) | Исторический | Старый post-hoc proof сохранён для аудита; candidate-значения и execution provenance superseded v2.1. |
| Финальный комплект Track 1 | Не сформирован | Должен пройти полный заявленный горизонт и соответствовать контракту ниже. |
| [`track2_model_z_v3/`](./track2_model_z_v3/README.md) | Проверен self-replay | Baseline выбран среди 32 кандидатов и прошёл OPM/официальный ЧДД: `11918,789227262983` млн руб. Улучшение не заявляется; `organizer_certified=false`. |
| [`qwen36_agent_tool_registry_a100_v3.json`](../evidence/qwen36_agent_tool_registry_a100_v3.json) | Проверен отдельный host-side live-workflow | SHA-256 `6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5`; 8/8 provider responses, четыре роли и четыре реальные read-tool calls. Approvals `[true,true,true,false]`, `critic_approved=false`; проверен только 412-action срез, не все 618, и не Compose-path. |
| [`docker_a100_v5.json`](../evidence/docker_a100_v5.json) | Проверен terminal runtime | SHA-256 `c7a48d202cd337f3d0915aac47688834d8a5ff6647b1aa6911535f633f00cf12`; UID 10001, `CapEff=0`, NNP/read-only, HTTP/doctor и negative secret gates PASS. Qwen workflow не вызван. |

Docker-образ поставляет только FastAPI/control plane и официальный Python ЧДД.
OPM, Track 2 training/search scripts, сохранённый суррогат и большие numerical
artifacts в образ не входят; их воспроизводимость обеспечивается отдельным
host-side операторским контуром и SHA-256 receipts. Поэтому успешный API health
не равен готовности конкурсного `wells_schedule.inc`.

Qwen terminal v3 — рекомендательный evidence: модель не запускала OPM/ЧДД,
uncertainty не conformally calibrated, а `organizer_certified=false`. Полное
расписание из 618 действий отдельно прошло OPM replay с ЧДД
`11918,789227262983` млн руб. Предыдущий таймаут сохранён неизменяемой
квитанцией
[`qwen36_agent_terminal_failure_a100_v2.json`](../evidence/qwen36_agent_terminal_failure_a100_v2.json)
(SHA-256
`e3a8fe69cc2764972e3679edb953e6f599c87875ef39718923f5a8bdb6547ecb`).

Track 2 replay receipts:
[`model_z_track2_final_replay_a100_v3.json`](../evidence/model_z_track2_final_replay_a100_v3.json)
(`359bd379d77adedb8d4bfb39f267335af40f64fcd67314e4cb8111d45fed483c`)
и
[`model_z_track2_final_replay_audit_a100_v3.json`](../evidence/model_z_track2_final_replay_audit_a100_v3.json)
(`3c6d50e5cf355b08f0c67d5feaadee9e319050499d91162c6061e87a650dda32`).
Машиночитаемый паспорт
[`submission.json`](./track2_model_z_v3/submission.json) имеет SHA-256
`695fcb09f4448c95c502d72b764468188024e74fef98b4f1142ee0fd55add849`;
README комплекта —
`a19e19062b4f15017e6fb2f8e626eb72045975416685e54dd94c20054c074ff5`.
Первая fail-closed попытка `a4593ea36d39529a44784313a875ad286e78b268f0a83fbe5716237cb7c02f09`
сохранена в evidence: OPM/export завершились, но ЧДД не запускался до scoped
исправления общего 64 MiB cap.

Имя `track2_model_z_v3` намеренно не заменено на `track2_final`: комплект
закрывает внутренний self-replay, но `organizer_certified=false` и улучшение
baseline не заявляется.

## Обязательный контракт финального каталога

Каждый финальный каталог Track 1 и Track 2 имеет одинаковый минимальный
контракт:

```text
trackN_final/
├── wells_schedule.inc
└── submission.json
```

- `wells_schedule.inc` — точный подаваемый организаторам **controls-only**
  include-файл, без альтернативных «почти финальных» вариантов. Его SHA-256
  обязан совпадать со значением в `submission.json`. Для replay этот файл
  детерминированно накладывается на исходный полный schedule модели; поэтому
  controls-only include и полный replay schedule не обязаны быть побайтово
  равны. Квитанция обязана фиксировать криптографическую цепочку
  `source_schedule_sha256 + controls_sha256 → replay_overlay_sha256` и
  подтверждать, что путь и SHA-256 точного input schedule в
  аутентифицированном OPM run manifest совпадают с replay overlay.
- `submission.json` — единственный машиночитаемый паспорт заявки. Он обязан
  содержать: номер трека и модель; `competition_submission=true`; SHA-256
  исходного архива и `wells_schedule.inc`; закреплённый simulator image с
  digest; точный горизонт (`start_inclusive`, `end_exclusive`, число месяцев);
  заявленный ЧДД в млн руб.; имя профиля ЧДД, `start_year` и
  `charge_initial_pump`; run-id финального Flow и расчёта ЧДД; пути и SHA-256
  всех квитанций и манифестов, которыми подтверждаются simulation, SUMMARY
  extraction, экспорт, ЧДД и итоговое решение.
- Для Track 2 `submission.json` дополнительно обязан ссылаться на scenario,
  dataset, surrogate и search receipts; фиксировать SHA-256 аутентифицированного
  scenario index, identity-baseline CHDD и всех четырёх run manifests;
  подтверждать побайтовый baseline и сохранение вторичных WCON/BHP-ограничений;
  содержать train/test-метрики, seed, model/metrics SHA-256 и факт финального OPM
  replay выбранного расписания. Controls-only заявка и полный overlay хранятся
  как два отдельных артефакта с собственными путями и SHA-256.
- Любая JSON-ссылка обязательна и проверяема: локальный относительный путь к
  короткой квитанции либо манифесту, SHA-256 самого JSON и, если большой
  артефакт хранится удалённо, точный абсолютный путь на A100 вместе с SHA-256
  артефакта. Запись только пути без хеша или только хеша без JSON provenance
  считается неполной.

Большие каталоги Flow, raw SUMMARY, отчёты, датасеты, модели, таблицы ЧДД и
Docker tar не копируются в git: они остаются на A100. Локальная поставка хранит
только расписание, `submission.json` и короткие JSON-квитанции/манифесты,
которые однозначно связывают локальные файлы с этими A100-артефактами.

Каталог считается финальным только когда все перечисленные ссылки существуют,
все хеши повторно проверены, заявленный ЧДД рассчитан из результата именно
указанного финального OPM run, а соответствующая терминальная квитанция внесена
в [индекс доказательств](../evidence/README.md). До этого имя `trackN_final/`
использовать нельзя.
