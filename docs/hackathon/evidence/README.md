# Доказательства КТ2

Каталог содержит короткие локальные квитанции и SHA-256 для проверки заявлений
КТ2. Основной сценарий показа: [контрольная точка 2](../checkpoint_2_2026-08-31.md),
актуальный source-bound комплект Track 1:
[описание поставки](../deliverables/track1_source_bound_v2_1/README.md); комплект
Track 2: [track2_model_z_v3](../deliverables/track2_model_z_v3/README.md).

| Квитанция | SHA-256 текущего файла | Что подтверждает | Статус и граница доказательства |
|---|---|---|---|
| [`qwen36_agent_tool_registry_a100.json`](./qwen36_agent_tool_registry_a100.json) | `42b05683992588fafe3a9eaac2545b9df6e86df5bcfe0975da339c3ac16bc031` | Исторический host-side A100 receipt четырёх ролей Qwen 3.6 | `critic_approved=false`; superseded terminal v3 ниже |
| [`qwen36_agent_terminal_failure_a100_v2.json`](./qwen36_agent_terminal_failure_a100_v2.json) | `e3a8fe69cc2764972e3679edb953e6f599c87875ef39718923f5a8bdb6547ecb` | Live-попытка v2 остановилась по таймауту до заголовков ответа: 0 provider responses, 0 ролей и 0 tool calls | Неизменяемая fail-closed история; retry в этой попытке не было |
| [`qwen36_agent_tool_registry_a100_v3.json`](./qwen36_agent_tool_registry_a100_v3.json) | `6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5` | Один terminal host-side live-workflow через внешний Tatneft `qwen3.6-35b-a3b`: 8/8 provider responses, четыре роли, четыре реальные allow-listed read-tool calls, без local model/fallback | PASS/complete/exit 0, approvals `[true,true,true,false]`, `critic_approved=false`; agent-tool context — только 412 действий января–апреля, не все 618; не provider attestation, не Compose→Qwen, не OPM/ЧДД и не organizer certification |
| [`model_y_track1_a100_v1.json`](./model_y_track1_a100_v1.json) | `1b59a7c60289fac36c05851c23af69955749b0987bf496cab63a65fe2d9b02a9` | Исторический Model Y baseline/candidate proof | PASS одного месяца, но численные candidate-claims и source provenance superseded v2.1 |
| [`model_y_track1_source_provenance_posthoc_v1.json`](./model_y_track1_source_provenance_posthoc_v1.json) | `fe3aae8e462bd331647ebc03d3316027ca77133a950b5997ca45a2a2dca1f84e` | Восстановленные staged sources старого proof | История: `cryptographic_execution_binding=false` |
| [`model_y_track1_source_bound_failure_a100_v2.json`](./model_y_track1_source_bound_failure_a100_v2.json) | `f2d838ea5e9e4dc3fd5febe32e6918c79030c8d62abcfc67d9ca483013c7da01` | Первая source-bound попытка остановилась на некорректной форме SUMMARY-vector contract до candidate Flow | Неизменяемая fail-closed история: `complete=false`, OPM-кандидаты не запускались |
| [`model_y_track1_source_bound_a100_v2_1.json`](./model_y_track1_source_bound_a100_v2_1.json) | `b3aebefce30a0a948a84702d3b6fd7598aa183c735f296d9cb7ae5a0003cf050` | Source-bound replay двух кандидатов: no-op `110,782383361`, alternative `110,776225880` млн руб.; выбран no-op | PASS одного месяца; `cryptographic_execution_binding=true`, `binary_restart=false`, `organizer_certified=false`, full horizon/global optimum не заявлены |
| [`model_z_baseline_a100_v1.json`](./model_z_baseline_a100_v1.json) | `78993a8e0f5b3c1cd53c4bd41c50478280d94bbafef6730df193f76393fb0927` | Неизменяемый Model Z baseline: исходный ZIP → OPM Flow 2026.04 → SUMMARY → экспорт → ЧДД | Проверено в режиме только чтения; подтверждает baseline, но не обучение, поиск и финальный replay Track 2 |
| [`model_z_economics_terminal_a100_v2.json`](./model_z_economics_terminal_a100_v2.json) | `6a814805cce34aaf91f815b567248dde88966e522cd5a11526b4d77611858d8c` | Повторный расчёт baseline ЧДД на закреплённых `opm/opm_chdd/economics`: профиль организаторов 1991 = 5 181,184136 млн руб.; рабочий профиль 2007 = 11 918,789227 млн руб. | PASS экономики baseline; `flow_rerun=false`, суррогат и оптимизация не заявлены; supersedes pre-terminal receipt |
| A100: `/tmp/timesoil-kt2/track2-v2/scenario-bundle-verification.json` | `f8284da948312446318e14acaf768bd2f06bacb61dec7f1fc667b7400b4273d9` | Identity-baseline, lossless вторичные WCON/BHP-поля и byte-identical regeneration; индекс `71edcb70cf4e04871f81e6d6ed4842f8cc91d542731024269060a1c8f5cfaf54`, эталонный CHDD `446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884` | Только preflight: подтверждает входные hash-gates, но не сценарные OPM-прогоны и не terminal Track 2 |
| [`track2_scenario_bundle_repro_a100_v1.json`](./track2_scenario_bundle_repro_a100_v1.json) | `e0354df254d22b74dad3157c7453819a1f6f59af0529776119be4039bad9ef2e` | Официальный canonical 10-column CSV `b49d…` детерминированно проецируется в controls `1a92…`; все 13 файлов совпали с активным bundle, index остался `71ed…` | PASS воспроизводимости генератора; не заменяет OPM-прогоны |
| [`track2_scenario_runner_live_source_observation_v1.json`](./track2_scenario_runner_live_source_observation_v1.json) | `59c37301bd5cc0cf323f4e5d7664aa24b0bb18fbfd74ddb06a9a2360fa6025ba` | Живой PID/tmux, argv и staged bytes раннера во время `perturbation-003`; snapshot map `d7ad…` | Наблюдение процесса, не pre-execution binding: `cryptographic_execution_binding=false` |
| [`track2_scenario_batch_completion_a100_v1.json`](./track2_scenario_batch_completion_a100_v1.json) | `d50386ca122c5e3608b14157661ff0974c9329f89072e7b46cbfc3b3d0f797b6` | Терминальный batch manifest `fd4a8d11ca85886f56e143ef0a48686da2677b80d770f95b5fd7e938ac33e946`: baseline + три возмущения, четыре точных dataset/manifests, по 38 213 строк, расхождений 0 | PASS сценариев; dataset `1253a351c8ea58dbd7e618cd50b9817f6d6b646962d2052744d4efb21a778cc0`; source-binding limitation из live observation сохранена |
| [`model_z_track2_training_a100_v1.json`](./model_z_track2_training_a100_v1.json) | `dc8dcab4de429ec9b7c96c1a368cf69dc204c332cc1741deb8c06de4438593b6` | Первая успешная тренировка и модельный artifact | История: OOD-domain contract оказался слишком узким; superseded v3 |
| [`model_z_track2_search_failure_a100_v1.json`](./model_z_track2_search_failure_a100_v1.json) | `0387cc735f15c97448b541d3bb8bd8b7394ca03daa79fed8b853ad614df0a033` | Search v1 fail-closed: baseline признан вне surrogate domain | Неизменяемая история: schedule/OPM/ЧДД не создавались |
| [`model_z_track2_training_terminal_a100_v3.json`](./model_z_track2_training_terminal_a100_v3.json) | `511e1950d43244ab9cd7ca034a26cd811e7532df155fe259555728a53c10036a` | Model `82dfc80d535345fddcf3ec3540c8ea66df89bce7ff50f1f262256fdf07cce4d3`, manifest `6f9532414dc8ea7291fca11159ab751c5370f9e59f234041ab017eda76698772`, metrics `95e3773ac6dc5036304c6f5fd697b9e603c9220ab7747d36469426a57ab8c4c4`; train WAPE oil/liquid `7,1949 %`/`4,5042 %`, RMSE `8,728 bar`, OOD `0,7286 %`; test `7,7940 %`/`5,1943 %`, `8,808 bar`, `0,5464 %` | Training PASS; nominal 90%-coverage train/test `43,71 %`/`40,50 %`, uncertainty не conformal; `organizer_certified=false`, exact argv cryptographic binding не заявляется |
| [`model_z_track2_search_terminal_a100_v3.json`](./model_z_track2_search_terminal_a100_v3.json) | `ccc9705475209baaa306a2e8a4bbff034cfa6996564f725fd4f58fcd9333c006` | 32/32 приняты, 0 OOD rejected; выбран `baseline`, proxy `127494,1351`; controls `74580379bf3b1551eac0b85fd9684dd6873a69149924491871dad92b7b31e659`, schedule `2cf99d0e70901d3881c8ce14b9901b82fa21e0ee2945ff6f1ee82a35429af372`, overlay `4fa3d5efb189bb365b426c6c6acca98cd41894ad88a53749b4d7742a83d0af35`, manifest `b4a6721adf38da833d41e21dd64a2496ee833e4b84e10dc2d583b5511052e5e8`, lineage `1cfddbd255d55a048ffe2da93af9b1db878111cd70a50b08efa0e959bc49de24` | Search PASS для selection; proxy не ЧДД; последующий replay закрыт receipts ниже |
| [`model_z_track2_final_replay_failure_a100_v3.json`](./model_z_track2_final_replay_failure_a100_v3.json) | `a4593ea36d39529a44784313a875ad286e78b268f0a83fbe5716237cb7c02f09` | Первая replay-попытка: OPM/export PASS, но SUMMARY 88 598 498 байт превысил общий 64 MiB evidence-cap до запуска ЧДД | Неизменяемая fail-closed история; retry в этой попытке запрещён |
| [`model_z_track2_final_replay_a100_v3.json`](./model_z_track2_final_replay_a100_v3.json) | `359bd379d77adedb8d4bfb39f267335af40f64fcd67314e4cb8111d45fed483c` | Baseline replay: OPM manifest `e4eba3f347d5e46d9319a8c52592c39635c3f2192974166e866fdc42b5bcb617`, SUMMARY extraction `2587105e3b4d14f26855d6a575408f6db921321436611f0ee0b915f727611b15`, canonical CHDD `446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884`; `operational_sunk_assets` = `11918.789227262983` млн руб. | PASS self-replay; выбран baseline, улучшение не заявляется, `organizer_certified=false` |
| [`model_z_track2_final_replay_audit_a100_v3.json`](./model_z_track2_final_replay_audit_a100_v3.json) | `3c6d50e5cf355b08f0c67d5feaadee9e319050499d91162c6061e87a650dda32` | Scoped cap-fix run: 20 linked artifacts, 9 execution sources и 22 OPM artifacts аутентифицированы без mismatch; failure receipt сохранён | PASS terminal audit; размерный cap снят только для authenticated linked replay artifacts |
| [`docker_a100_v1.json`](./docker_a100_v1.json) | `f82e2ee311bfa6ce587e57923bad37da1ef420bd83b1c3b7e4fbd553d324dc53` | Историческая сборка API-образа | Provisional; superseded v5 |
| [`docker_a100_v5.json`](./docker_a100_v5.json) | `c7a48d202cd337f3d0915aac47688834d8a5ff6647b1aa6911535f633f00cf12` | Fresh-context image `sha256:41116e94f97801d7f5b234a1e597134149e22ddccf60d9b58445dc6459b56802`; 11/11 current-source hashes, UID 10001, `CapEff=0`, NNP, read-only rootfs, HTTP/doctor, empty/unreadable secret gates и cleanup | PASS terminal Docker runtime/security; Qwen workflow/connectivity не запускались |

Track 2 final replay закрыт. Локальный комплект
[`track2_model_z_v3`](../deliverables/track2_model_z_v3/README.md) связывает
controls, overlay, lineage, search manifest, OPM, canonical export и ЧДД.
Его `submission.json` имеет SHA-256
`695fcb09f4448c95c502d72b764468188024e74fef98b4f1142ee0fd55add849`,
README — `a19e19062b4f15017e6fb2f8e626eb72045975416685e54dd94c20054c074ff5`.
Self-replay authenticated; это не сертификация организаторов.

Qwen terminal v3 закрывает отдельный host-side live-gate: одна попытка без
retry, 8/8 ответов, четыре роли и четыре реальные allow-listed read-tool calls.
Секрет, reasoning и raw content не копировались, не печатались и не
сохранялись. Критик явно отказал (`critic_approved=false`), потому что его
инструментальный context ограничен 412 действиями января–апреля. Все 618
действий отдельно валидированы full OPM replay с ЧДД
`11918,789227262983` млн руб.; Qwen не запускал OPM/ЧДД, uncertainty не
conformally calibrated, а `organizer_certified=false`.

Docker gate закрыт terminal v5 receipt. Образ не содержит OPM, training/search
scripts, model weights или большие численные артефакты. Protected key
материализовался только в RAM file secret, затем был удалён; Qwen workflow не
входил в Docker-проверку.

Большие исходники, отчёты SUMMARY, OPM run-каталоги, модели и таблицы ЧДД
хранятся на A100 под `/tmp/timesoil-kt2/`; точные пути записаны внутри
квитанций. Локальные JSON — только hash-ledger происхождения и проверок. Они не
заменяют исходные артефакты и не позволяют восстановить их содержимое.
