'use strict';
// TimesOil Track 2 viewer: vanilla JS, no build step, no external resources.
// ?mock=1 loads mock/results.json instead of the backend API.

const MOCK = new URLSearchParams(location.search).get('mock') === '1';
const PALETTE = ['#0b6bcb', '#c2410c', '#15803d', '#7c3aed', '#b45309', '#0e7490', '#be123c', '#4d7c0f'];

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const num = (v, d = 1) => (v === null || v === undefined || !isFinite(v)) ? '—'
  : Number(v).toLocaleString('ru-RU', { minimumFractionDigits: d, maximumFractionDigits: d });

let state = { data: null, well: null };

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
  return r.json();
}

// ---------- data loading ----------

async function loadRuns() {
  if (MOCK) {
    const d = await getJSON('mock/results.json');
    return [{ run_id: d.run_id, path: 'mock', final: true, updated_utc: '' }];
  }
  const d = await getJSON('/v1/results');
  return d.runs || [];
}

async function loadRun(runId) {
  return MOCK ? getJSON('mock/results.json') : getJSON('/v1/results/' + encodeURIComponent(runId));
}

async function init() {
  const sel = $('run-select');
  try {
    const runs = await loadRuns();
    sel.innerHTML = '';
    if (!runs.length) {
      sel.appendChild(el('option', null, 'расчётов нет'));
      $('status').textContent = 'нет доступных расчётов';
      return;
    }
    for (const r of runs) {
      const o = el('option', null, r.run_id + (r.final ? ' · финал' : ''));
      o.value = r.run_id;
      sel.appendChild(o);
    }
    sel.onchange = () => select(sel.value);
    await select(runs[0].run_id);
  } catch (e) {
    sel.innerHTML = '';
    sel.appendChild(el('option', null, 'ошибка'));
    $('status').textContent = 'ошибка загрузки: ' + e.message;
  }
}

async function select(runId) {
  $('status').textContent = 'загрузка ' + runId + '…';
  try {
    state.data = await loadRun(runId);
    state.well = null;
    render();
    $('status').textContent = MOCK ? 'демо-данные (?mock=1)' : '';
  } catch (e) {
    $('status').textContent = 'ошибка загрузки: ' + e.message;
  }
}

// ---------- rendering ----------

function render() {
  const d = state.data || {};
  const npv = d.npv || {};
  $('npv-final').textContent = num(npv.final_m);
  $('npv-inc').textContent = num(npv.incumbent_m);
  $('npv-gain').textContent = num(npv.gain_pct, 2);
  $('oil-total').textContent = num(npv.oil_total_t, 0);

  const badge = $('badge');
  badge.textContent = d.official ? 'официально' : 'прогноз';
  badge.className = 'badge ' + (d.official ? 'badge-official' : 'badge-forecast');

  renderMap();
  renderFieldCharts();
  renderWellCharts();
  renderConstraints();
  renderDiff();
  renderInterp();
}

function wellSize(w) {
  const t = (w.totals || {});
  const v = w.role === 'injector' ? (t.injection_m3 || 0) : (t.oil_t || 0);
  return 3 + 7 * Math.sqrt(Math.max(v, 0) / (maxTotal() || 1));
}

let _maxTotal = null;
function maxTotal() {
  if (_maxTotal !== null) return _maxTotal;
  const ws = (state.data && state.data.wells) || [];
  _maxTotal = ws.reduce((m, w) => {
    const t = w.totals || {};
    return Math.max(m, w.role === 'injector' ? (t.injection_m3 || 0) : (t.oil_t || 0));
  }, 0);
  return _maxTotal;
}

function blockColor(b) {
  return b === null || b === undefined ? '#8a96a3' : PALETTE[Math.abs(b) % PALETTE.length];
}

function renderMap() {
  _maxTotal = null;
  const host = $('map');
  host.innerHTML = '';
  const wells = ((state.data || {}).wells || []).filter((w) => w.x !== null && w.y !== null && w.x !== undefined && w.y !== undefined);
  if (!wells.length) {
    host.appendChild(el('div', 'empty', 'Координаты скважин недоступны.'));
    return;
  }
  const xs = wells.map((w) => w.x), ys = wells.map((w) => w.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = 900, H = 520, pad = 40;
  const sx = (v) => pad + (x1 === x0 ? (W - 2 * pad) / 2 : (v - x0) / (x1 - x0) * (W - 2 * pad));
  const sy = (v) => H - pad - (y1 === y0 ? (H - 2 * pad) / 2 : (v - y0) / (y1 - y0) * (H - 2 * pad));

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` });
  for (const w of wells) {
    const cx = sx(w.x), cy = sy(w.y), r = wellSize(w);
    const shut = w.status_last === 'SHUT';
    const attrs = {
      class: 'well' + (state.well === w.well ? ' selected' : ''),
      fill: blockColor(w.block),
      'fill-opacity': shut ? 0.25 : 0.85
    };
    let node;
    if (w.role === 'injector') {
      node = svgEl('polygon', Object.assign({ points: `${cx},${cy - r} ${cx + r},${cy + r} ${cx - r},${cy + r}` }, attrs));
    } else {
      node = svgEl('circle', Object.assign({ cx, cy, r }, attrs));
    }
    const t = w.totals || {};
    node.addEventListener('mouseenter', () => {
      $('map-tip').textContent = `${w.well} · ${w.role === 'injector' ? 'нагнетательная' : 'добывающая'} · блок ${w.block ?? '—'} · `
        + `нефть ${num(t.oil_t, 0)} т · жидкость ${num(t.liquid_m3, 0)} м³ · вода ${num(t.water_m3, 0)} м³ · закачка ${num(t.injection_m3, 0)} м³ · ${shut ? 'остановлена' : 'в работе'}`;
    });
    node.addEventListener('click', () => { state.well = state.well === w.well ? null : w.well; render(); });
    svg.appendChild(node);
    const lab = svgEl('text', { class: 'well-label', x: cx + r + 2, y: cy + 3 });
    lab.textContent = w.well;
    svg.appendChild(lab);
  }
  host.appendChild(svg);
}

function svgEl(tag, attrs) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

// generic line chart: series = [{name, values, color, dash}], hlines = [{value, label, color}]
function chart(title, months, series, hlines) {
  const box = el('div', 'chart');
  box.appendChild(el('h3', null, title));
  const usable = series.filter((s) => (s.values || []).some((v) => v !== null && v !== undefined && isFinite(v)));
  if (!usable.length) {
    box.appendChild(el('div', 'empty', 'Нет данных.'));
    return box;
  }
  const W = 520, H = 200, L = 52, R = 10, T = 12, B = 30;
  let lo = Infinity, hi = -Infinity;
  for (const s of usable) for (const v of s.values) if (v !== null && v !== undefined && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const h of (hlines || [])) if (h.value !== null && h.value !== undefined && isFinite(h.value)) { lo = Math.min(lo, h.value); hi = Math.max(hi, h.value); }
  if (lo > 0) lo = 0;
  if (lo === hi) hi = lo + 1;
  const n = months.length || 1;
  const px = (i) => L + (n === 1 ? (W - L - R) / 2 : i / (n - 1) * (W - L - R));
  const py = (v) => H - B - (v - lo) / (hi - lo) * (H - T - B);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` });
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = py(v);
    svg.appendChild(svgEl('line', { class: 'grid', x1: L, y1: y, x2: W - R, y2: y }));
    const t = svgEl('text', { class: 'tick', x: L - 4, y: y + 3, 'text-anchor': 'end' });
    t.textContent = num(v, Math.abs(hi) < 10 ? 2 : 0);
    svg.appendChild(t);
  }
  svg.appendChild(svgEl('line', { class: 'axis', x1: L, y1: H - B, x2: W - R, y2: H - B }));
  const step = Math.max(1, Math.ceil(n / 6));
  for (let i = 0; i < n; i += step) {
    const t = svgEl('text', { class: 'tick', x: px(i), y: H - B + 12, 'text-anchor': 'middle' });
    t.textContent = (months[i] || '').slice(0, 7);
    svg.appendChild(t);
  }
  for (const h of (hlines || [])) {
    if (h.value === null || h.value === undefined || !isFinite(h.value)) continue;
    svg.appendChild(svgEl('line', { class: 'cap', stroke: h.color || '#b3261e', x1: L, y1: py(h.value), x2: W - R, y2: py(h.value) }));
  }
  for (const s of usable) {
    let dstr = '', pen = false;
    s.values.forEach((v, i) => {
      if (v === null || v === undefined || !isFinite(v)) { pen = false; return; }
      dstr += (pen ? 'L' : 'M') + px(i).toFixed(1) + ' ' + py(v).toFixed(1) + ' ';
      pen = true;
    });
    svg.appendChild(svgEl('path', { class: 'ser', d: dstr, stroke: s.color }));
  }
  box.appendChild(svg);
  const leg = el('div', 'chart-legend');
  for (const s of usable) {
    const sp = el('span');
    const i = el('i'); i.style.background = s.color;
    sp.appendChild(i); sp.appendChild(document.createTextNode(s.name));
    leg.appendChild(sp);
  }
  for (const h of (hlines || [])) {
    if (h.value === null || h.value === undefined || !isFinite(h.value)) continue;
    leg.appendChild(el('span', 'muted', h.label + ' ' + num(h.value, 2)));
  }
  box.appendChild(leg);
  return box;
}

function renderFieldCharts() {
  const host = $('field-charts');
  host.innerHTML = '';
  const d = state.data || {};
  const months = d.months || [];
  const f = ((d.series || {}).field) || {};
  const prof = d.profile || {};
  const c = d.constraints || {};
  if (!months.length) {
    host.appendChild(el('div', 'empty', 'Помесячные ряды недоступны.'));
    return;
  }
  host.appendChild(chart('Жидкость и закачка, м³/сут', months, [
    { name: 'жидкость', values: f.liquid_m3d || [], color: PALETTE[0] },
    { name: 'закачка', values: f.injection_m3d || [], color: PALETTE[2] }
  ], [
    { value: prof.liquid_cap_m3d ?? c.max_liquid_m3d, label: 'лимит жидкости', color: '#b3261e' },
    { value: prof.injection_cap_m3d ?? c.max_injection_m3d, label: 'лимит закачки', color: '#0e7490' }
  ]));
  host.appendChild(chart('Нефть, т/мес', months, [
    { name: 'нефть', values: f.oil_t || [], color: PALETTE[1] }
  ], []));
  const vrr = prof.vrr || {};
  host.appendChild(chart('VRR', months, [
    { name: 'VRR', values: f.vrr || [], color: PALETTE[3] }
  ], [
    { value: vrr.min ?? c.vrr_min, label: 'VRR min', color: '#b3261e' },
    { value: vrr.max ?? c.vrr_max, label: 'VRR max', color: '#b3261e' }
  ]));
}

function renderWellCharts() {
  const host = $('well-charts');
  host.innerHTML = '';
  $('well-name').textContent = state.well || 'не выбрана';
  if (!state.well) {
    host.appendChild(el('div', 'empty', 'Выберите скважину на карте.'));
    return;
  }
  const d = state.data || {};
  const months = d.months || [];
  const s = (((d.series || {}).wells) || {})[state.well];
  if (!s) {
    host.appendChild(el('div', 'empty', 'Ряды по скважине недоступны.'));
    return;
  }
  host.appendChild(chart('Помесячные объёмы', months, [
    { name: 'нефть, т', values: s.oil_t || [], color: PALETTE[1] },
    { name: 'жидкость, м³', values: s.liquid_m3 || [], color: PALETTE[0] },
    { name: 'вода, м³', values: s.water_m3 || [], color: PALETTE[5] },
    { name: 'закачка, м³', values: s.injection_m3 || [], color: PALETTE[2] }
  ], []));
  const bh = d.profile && d.profile.bhp_bounds ? d.profile.bhp_bounds : [null, null];
  host.appendChild(chart('Забойное давление, атм', months, [
    { name: 'BHP', values: s.bhp_bar || [], color: PALETTE[4] }
  ], [
    { value: bh[0], label: 'BHP min', color: '#b3261e' },
    { value: bh[1], label: 'BHP max', color: '#b3261e' }
  ]));
}

function renderConstraints() {
  const host = $('constraints');
  host.innerHTML = '';
  const d = state.data || {};
  const c = d.constraints || {};
  const p = d.profile || {};
  const vrr = p.vrr || {};
  const rows = [
    ['Максимум жидкости, м³/сут', c.max_liquid_m3d ?? p.liquid_cap_m3d],
    ['Максимум закачки, м³/сут', c.max_injection_m3d ?? p.injection_cap_m3d],
    ['VRR минимум', c.vrr_min ?? vrr.min],
    ['VRR максимум', c.vrr_max ?? vrr.max],
    ['Окно VRR, мес', vrr.window_months],
    ['Границы BHP, атм', p.bhp_bounds ? p.bhp_bounds.map((v) => num(v, 0)).join(' … ') : null]
  ];
  const tbl = el('table');
  for (const [k, v] of rows) {
    const tr = el('tr');
    tr.appendChild(el('th', null, k));
    tr.appendChild(el('td', 'num', typeof v === 'string' ? v : num(v, 2)));
    tbl.appendChild(tr);
  }
  host.appendChild(tbl);

  const viol = c.violations || [];
  if (!viol.length) {
    host.appendChild(el('div', 'ok', 'нарушений нет'));
    return;
  }
  for (const v of viol) {
    const parts = [v.rule, v.month, v.well, v.detail].filter(Boolean);
    host.appendChild(el('div', 'viol', parts.join(' · ')));
  }
}

function renderDiff() {
  const host = $('diff');
  host.innerHTML = '';
  const all = (state.data || {}).controls_diff || [];
  const rows = state.well ? all.filter((r) => r.well === state.well) : all.slice(0, 50);
  $('diff-scope').textContent = state.well ? '· ' + state.well : '· топ-50 по всем скважинам';
  if (!rows.length) {
    host.appendChild(el('div', 'empty', 'Изменений нет.'));
    return;
  }
  const tbl = el('table');
  const head = el('tr');
  for (const h of ['Месяц', 'Скважина', 'Поле', 'Инкамбент', 'Финал']) head.appendChild(el('th', null, h));
  tbl.appendChild(head);
  for (const r of rows) {
    const tr = el('tr');
    tr.appendChild(el('td', null, r.month || ''));
    tr.appendChild(el('td', null, r.well || ''));
    tr.appendChild(el('td', null, r.field === 'status' ? 'статус' : 'значение'));
    tr.appendChild(el('td', 'num', fmtCell(r.incumbent)));
    tr.appendChild(el('td', 'num', fmtCell(r.final)));
    tbl.appendChild(tr);
  }
  const wrap = el('div', 'scroll');
  wrap.appendChild(tbl);
  host.appendChild(wrap);
}

function fmtCell(v) {
  if (v === null || v === undefined) return '—';
  return typeof v === 'number' ? num(v, 2) : String(v);
}

function renderInterp() {
  const host = $('interp');
  host.innerHTML = '';
  const it = (state.data || {}).interpretability || {};
  if (it.summary_md) {
    host.appendChild(el('pre', 'md', it.summary_md));
  } else {
    host.appendChild(el('div', 'empty', 'Пояснение недоступно.'));
  }
  const rep = it.report;
  if (rep && typeof rep === 'object') {
    for (const k of Object.keys(rep)) {
      const dt = el('details');
      dt.appendChild(el('summary', null, k));
      dt.appendChild(el('pre', null, JSON.stringify(rep[k], null, 2)));
      host.appendChild(dt);
    }
  }
}

init();
