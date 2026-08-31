from __future__ import annotations

UI_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; img-src 'none'; object-src 'none'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; "
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

OPERATOR_PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TimesOil AIOS · оператор</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; line-height: 1.5; }
    body { margin: 0; color: #18211d; background: #f4f1e8; }
    main { width: min(900px, calc(100% - 2rem)); margin: 2rem auto; }
    header, section { margin-bottom: 1rem; padding: 1.25rem; background: #fff; border: 1px solid #d5d0c4; border-radius: 10px; }
    h1, h2, h3 { margin-top: 0; }
    h1 { font-size: clamp(1.6rem, 5vw, 2.35rem); }
    h2 { font-size: 1.2rem; }
    .note, .muted { color: #4c5852; }
    .status { min-height: 1.5em; font-weight: 650; }
    ul { padding-left: 1.25rem; }
    li.ok::marker { color: #167044; }
    li.warn::marker { color: #a04420; }
    label { display: block; margin-bottom: .4rem; font-weight: 700; }
    textarea, pre { box-sizing: border-box; width: 100%; padding: .75rem; border: 1px solid #8a918d; border-radius: 6px; background: #fbfbf8; color: inherit; }
    textarea { min-height: 13rem; resize: vertical; font: .9rem ui-monospace, monospace; }
    pre { min-height: 8rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
    .actions { display: flex; flex-wrap: wrap; gap: .75rem; margin: .75rem 0; }
    button { padding: .65rem 1rem; border: 0; border-radius: 6px; color: #fff; background: #185d45; font: inherit; font-weight: 700; cursor: pointer; }
    button.secondary { background: #46534d; }
    button:disabled { cursor: wait; opacity: .6; }
    button:focus-visible, textarea:focus-visible, pre:focus-visible { outline: 3px solid #b36a25; outline-offset: 2px; }
  </style>
</head>
<body>
<main>
  <header>
    <p class="muted">TimesOil · контрольная точка 2</p>
    <h1>Панель оператора AIOS</h1>
    <p class="note">Статусы ниже показывают фактическую готовность компонентов. Доступный компонент не считается сертифицированным.</p>
  </header>

  <section aria-labelledby="system-title">
    <h2 id="system-title">Состояние системы</h2>
    <p id="api-status" class="status" role="status" aria-live="polite">Проверка API…</p>
    <ul id="capabilities" aria-label="Возможности AIOS"></ul>
    <button id="refresh" class="secondary" type="button">Обновить статус</button>
  </section>

  <section aria-labelledby="experiment-title">
    <h2 id="experiment-title">Эксперимент с агентами</h2>
    <form id="experiment-form">
      <label for="context">Контекст запроса в JSON</label>
      <textarea id="context" name="context" spellcheck="false" required>{
  "request": "Проверить готовность предложенного плана",
  "track": 2,
  "field_state": {},
  "constraints": {},
  "evidence": {
    "model_z_ready": false,
    "chdd_complete": false
  }
}</textarea>
      <div class="actions">
        <button id="run" type="submit">Запустить четыре роли</button>
      </div>
      <p id="experiment-status" class="status" role="status" aria-live="polite"></p>
    </form>
    <h3 id="result-title">Структурированный результат</h3>
    <pre id="result" tabindex="0" aria-labelledby="result-title" aria-live="polite">Результат ещё не получен.</pre>
  </section>
</main>
<script>
  "use strict";

  const element = (id) => document.getElementById(id);

  function showResult(value) {
    element("result").textContent = JSON.stringify(value, null, 2);
  }

  async function requestJson(path, options = {}) {
    const response = await fetch(path, options);
    const raw = await response.text();
    let payload;
    try {
      payload = raw ? JSON.parse(raw) : null;
    } catch (_) {
      payload = {detail: "Сервер вернул некорректный JSON"};
    }
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`);
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function addCapability(list, label, value, positive) {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    const state = document.createElement("span");
    name.textContent = `${label}: `;
    state.textContent = value;
    item.className = positive ? "ok" : "warn";
    item.append(name, state);
    list.append(item);
  }

  function renderCapabilities(data) {
    const list = element("capabilities");
    list.replaceChildren();
    const qwen = data.qwen || {};
    const track2 = data.track2 || {};
    const chdd = data.chdd || {};
addCapability(
  list,
  "Qwen3.6",
  qwen.configured
    ? (qwen.connectivity_verified ? "настроен, связь подтверждена" : "API настроен, связь не проверена")
    : "не настроен",
  Boolean(qwen.configured && qwen.connectivity_verified),
);
    addCapability(
      list,
      "Трек 2",
      track2.certified
        ? "сертифицирован"
        : (track2.component_available
          ? `доступен, не сертифицирован; Model Z ${track2.model_z_trained ? "обучен" : "не обучен"}`
          : "недоступен"),
      Boolean(track2.certified)
    );
    const chddReady = Boolean(chdd.component_available && chdd.ready);
    addCapability(list, "ЧДД", chddReady ? "калькулятор готов" : "калькулятор не настроен", chddReady);
  }

  async function refreshStatus() {
    const button = element("refresh");
    button.disabled = true;
    element("api-status").textContent = "Проверка API…";
    try {
      const [health, capabilities] = await Promise.all([
        requestJson("/health"),
        requestJson("/v1/capabilities")
      ]);
      element("api-status").textContent = health.status === "ok" ? "API отвечает" : "Неизвестный статус API";
      renderCapabilities(capabilities);
    } catch (error) {
      element("api-status").textContent = "Не удалось получить состояние API";
      showResult({error: error.message, response: error.payload || null});
    } finally {
      button.disabled = false;
    }
  }

  element("refresh").addEventListener("click", refreshStatus);
  element("experiment-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = element("run");
    const status = element("experiment-status");
    let context;
    try {
      context = JSON.parse(element("context").value);
      if (!context || Array.isArray(context) || typeof context !== "object") {
        throw new Error("Контекст должен быть JSON-объектом");
      }
    } catch (error) {
      status.textContent = `Ошибка ввода: ${error.message}`;
      showResult({error: error.message});
      return;
    }

    button.disabled = true;
    status.textContent = "Агенты выполняют проверку…";
    showResult({status: "running"});
    try {
      const payload = await requestJson("/v1/experiments/agents", {
        method: "POST",
        headers: {"Accept": "application/json", "Content-Type": "application/json"},
        body: JSON.stringify({context})
      });
      showResult(payload);
      status.textContent = payload.critic_approved
        ? "Критик одобрил рекомендацию; это не сертификат результата."
        : "Критик не одобрил рекомендацию; результат не сертифицирован.";
    } catch (error) {
      status.textContent = "Эксперимент завершился ошибкой";
      showResult({error: error.message, response: error.payload || null});
    } finally {
      button.disabled = false;
    }
  });

  refreshStatus();
</script>
</body>
</html>
"""
