# Track 1 — шестимесячное доказательство Model Y v3

Поставка фиксирует воспроизводимый шестимесячный контур
`controls → OPM Flow → ЧДД → выбор MPC` за `2014-01-01..2014-07-01`.
Для каждого из шести месяцев проверены три варианта управления нагнетательной
скважиной 2 (`140/150/160` м³/сут), всего 18 полных replay. Выбранный график
содержит 49 управляющих действий в месяц: 33 добывающие и 16 нагнетательных
скважин.

Базовый ЧДД равен `361,7478381787023` млн руб., выбранный —
`361,95624753135576` млн руб.; разница `+0,2084093526534616` млн руб.
На каждом шаге выбран вариант `160` м³/сут для скважины 2 вместо базовых
`150` м³/сут. Остальные управления сохранены.

Это самостоятельная техническая сертификация: `self_certified=true`,
`organizer_certified=false`. Значение `certified=true` в `proof.json` означает
успешную внутреннюю проверку воспроизводимости, физических ограничений,
OPM/ЧДД и цепочки lineage. Глобальная оптимальность и сертификация организатора
не заявляются: выбор сделан только в объявленной трёхточечной сетке кандидатов.

Состав:

- `wells_schedule.inc` и `baseline_wells_schedule.inc` — выбранный и базовый
  шестимесячные controls-only include;
- `result.json`, `proof.json`, `manifest.json`, `manifest.sha256` — побайтовые
  копии терминальных артефактов A100;
- `terminal_lineage/` — терминальная квитанция выбранной рекурсивной цепочки;
- `submission.json` — паспорт поставки, hashes и границы утверждений.

Крупные результаты симулятора, исходный архив и `selected_chdd.csv` в git не
добавлены. Их пути и SHA-256 сохранены в терминальной квитанции
`../../evidence/model_y_track1_six_month_a100_v3.json`, SHA-256
`175a73d9fc7608841b52ade7d64e34ebc313bed08c19fb5f2780286891118e39`.

Воспроизведение на A100 из commit
`f142d2343395fe3865a3fd1178ccea9394846449`:

```bash
cd /root/projects/TimesOil
uv run python scripts/run_model_y_track1_proof.py \
  --resume-baseline \
  --output-root /tmp/timesoil-kt2/track1-six-month-v1/output \
  --source '/tmp/timesoil-kt2/model_y/Model_Y (3).zip' \
  --baseline-dir /tmp/timesoil-kt2/track1-v1/results/model-y-baseline-20260831-a100-v4 \
  --reference-workbook '/root/projects/TimesOil/docs/hackathon/chdd/reference_baselines/Расчет ЧДД через OPM Flow Model_Y.xlsx' \
  --deck MODEL_Y/MODEL_Y.DATA \
  --schedule-relative-path MODEL_Y/INCLUDE/DemoSpe_002_2_sch.inc
```

Ожидаемая терминальная квитанция:
`output/results/model-y-track1-mpc-20260831-a100-v3/track1-658094d8f598d7ae6112ad91/proof.json`,
SHA-256 `01a3ec7bdf3fd3bdcc8209a7d8ecee6881f923e2cbd0b24636dff0de2010b3b0`.
