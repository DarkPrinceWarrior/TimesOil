# Flow grid — T1 / T2 swimlanes

**Date:** 2026-08-17
**Files:** `docs/hackathon/schemes/track1-flow.svg`, `docs/hackathon/schemes/track2-flow.svg`
**This file:** spec only. Do not paste a full drawing here.
**Tokens:** `design.md`. **Not** the 2×4 poster grid in `t1-layout.md` / `t2-layout.md`.

`_build_flows.py` is a draft. It is wrong where it disagrees: label plate **168**, heads **10×10**, T1 loop **4 segments**, T2 gate **H–V–H**. This file wins.

---

## Canvas

| | px |
|---|---:|
| Size | **2400 × 1350** |
| `viewBox` | `0 0 2400 1350` |
| Safe | **48 … 2352 × 48 … 1302** (2304 × 1254) |
| Paper | `--paper` `#F3EEE4` |
| Frame | `(48, 48, 2304, 1254)`, stroke `--gold` 1.25, no second frame |

Hairlines (stroke `--gold` 1.25, x = 48 … 2352): **y = 200**, **y = 1100**.
Lane joints: 1 px `--line` at y = **350, 500, 650, 800, 950**.

No `feTurbulence`, no fault gradient, no diagonal, no 80 px gold grid.

---

## Bands

| Band | y | h | y_end |
|---|---:|---:|---:|
| HEADER | 48 | 152 | 200 |
| 6 lanes | 200 | 900 | 1100 |
| FOOTER | 1100 | 202 | 1302 |

Header: kicker `(64, 76)`, `.h1` `(64, 108)`, lede `(64, 132)`.
Role chips: **250 × 52**, y = 72, x = **1288, 1550, 1812, 2074** (last right = 2324).
Colophon baseline **y = 1284**: left x = 64, right x = 2336 `text-anchor="end"`.

---

## Lanes (same y on both files)

Label plate: **x = 48, w = 160** → 48 … 208. Fill `#EBE4D4`. Type `.lane`, fill `--gold-ink` `#6B5012`, x = 64, y = lane_y + 79.
`--gold` `#C4A35A` is never text.

Content well: **208 … 2352**. Gutter after plate **24** → first node x = **232**.

| # | y | h | y_end | T1 | T2 | wash |
|---:|---:|---:|---:|---|---|---|
| 1 | **200** | 150 | 350 | ДАННЫЕ | ДАННЫЕ | `#F7F2E6` |
| 2 | **350** | 150 | 500 | МАС | СУРРОГАТ | T1 `--paper` / T2 `--gold-wash` |
| 3 | **500** | 150 | 650 | ШЛЮЗ | МАС | `#F7F2E6` / `--paper` |
| 4 | **650** | 150 | 800 | МИР | ШЛЮЗ | `--gold-wash` / `#F7F2E6` |
| 5 | **800** | 150 | 950 | ДЕНЬГИ | ФИЗИЧЕСКАЯ ВЕРИФИКАЦИЯ | `#F7F2E6` / `--gold-wash` |
| 6 | **950** | 150 | 1100 | СДАЧА | СДАЧА | `--rust-wash` |

Lane wash: `(48, y, 2304, 150)`, then the 160 px plate, then nodes.

---

## Node slots

Formula (row r = 0…5, col c = 0…5):

```
SLOT(r, c) = (232 + 360·c,  228 + 150·r,  280,  72)
```

| | px |
|---|---:|
| Node | **280 × 72**, `rx = 2` |
| Col step | **360** (280 + gutter 80) |
| Row step | **150** (same as lane h) |
| Top inset in lane | 28 (bottom inset 50) |

Column x: **232, 592, 952, 1312, 1672, 2032**. Right of col 5 = **2312** (40 px to 2352).
Row y: **228, 378, 528, 678, 828, 978**. Bottom of row 5 = **1050** (50 px to 1100).

Ports:

| side | x | y |
|---|---|---|
| L | x | y + 36 |
| R | x + 280 | y + 36 |
| T | x + 140 | y |
| B | x + 140 | y + 72 |

Node chrome: fill `--card` / `--gold-wash` / `--rust-wash`; stroke `--line` 1; left bar 4 px `--gold` or `--rust`. Title + one sub only. Empty slots: do not draw ghosts.

---

## Arrows

Only **H**, **V**, or **one elbow** (two segments). No cubic, no diagonal, no 2+ elbows on one path.
No `marker-start` / `marker-mid` / `marker-end`.

Head = **8 × 10** polygon, no stroke. Fill `--gold-ink` default, `--rust` on risk / return / certify. Tip at the target port. Shaft stops **8** px before the tip.

```
RIGHT  (tx,ty) (tx-8, ty-5) (tx-8, ty+5)
LEFT   (tx,ty) (tx+8, ty-5) (tx+8, ty+5)
DOWN   (tx,ty) (tx-5, ty-8) (tx+5, ty-8)
UP     (tx,ty) (tx-5, ty+8) (tx+5, ty+8)
```

Label `.el` 11 / 600 / `--gold-ink` (never `--gold`).
H: `text-anchor="middle"`, x = mid(shaft), y = y_line − 8.
V: x = x_line + 10, y = mid(y1, y2).
Elbow: same rule on the longer segment.
Every vertex and the label box stay in **48 … 2352 × 48 … 1302**. If a label would exit, shorten the copy.

---

## T1 nodes

| id | r | c | x | y | w | h | title |
|---|---:|---:|---:|---:|---:|---:|---|
| zip | 0 | 0 | 232 | 228 | 280 | 72 | Model Y |
| case | 0 | 1 | 592 | 228 | 280 | 72 | Кейс и лимиты |
| state | 0 | 2 | 952 | 228 | 280 | 72 | Состояние фонда |
| llm | 1 | 2 | 952 | 378 | 280 | 72 | Ядро МАС |
| intent | 1 | 3 | 1312 | 378 | 280 | 72 | Намерения |
| lim | 2 | 3 | 1312 | 528 | 280 | 72 | Жёсткие лимиты |
| crm | 2 | 4 | 1672 | 528 | 280 | 72 | CRM / Джентил |
| cmp | 2 | 5 | 2032 | 528 | 280 | 72 | Компилятор |
| gdm | 3 | 5 | 2032 | 678 | 280 | 72 | Полная ГДМ |
| tab | 4 | 4 | 1672 | 828 | 280 | 72 | 14 полей |
| chdd | 4 | 5 | 2032 | 828 | 280 | 72 | CHDD_PYTHON |
| pack | 5 | 4 | 1672 | 978 | 280 | 72 | Пакет сдачи |
| org | 5 | 5 | 2032 | 978 | 280 | 72 | Организаторы |

Wash: llm / intent / cmp / gdm / chdd → `--gold-wash`. pack / org → `--rust-wash`.

### T1 edges

H:

| from → to | line | tip | label |
|---|---|---|---|
| zip → case | (512, 264) → (584, 264) | (592, 264) R | архив / deck |
| case → state | (872, 264) → (944, 264) | (952, 264) R | лимиты кейса |
| llm → intent | (1232, 414) → (1304, 414) | (1312, 414) R | оценка состояния |
| lim → crm | (1592, 564) → (1664, 564) | (1672, 564) R | допустимые действия |
| crm → cmp | (1952, 564) → (2024, 564) | (2032, 564) R | принятый месяц |
| tab → chdd | (1952, 864) → (2024, 864) | (2032, 864) R | таблица ГДМ |
| pack → org | (1952, 1014) → (2024, 1014) | (2032, 1014) R | заявка · rust |

V:

| from → to | line | tip | label |
|---|---|---|---|
| state → llm | (1092, 300) → (1092, 370) | (1092, 378) D | канон. состояние |
| intent → lim | (1452, 450) → (1452, 520) | (1452, 528) D | ControlIntent |
| cmp → gdm | (2172, 600) → (2172, 670) | (2172, 678) D | include месяца |
| chdd → org | (2172, 900) → (2172, 970) | (2172, 978) D | заявленный ЧДД · rust |

Elbow (one corner):

| from → to | points | tip | label |
|---|---|---|---|
| gdm → tab | (2032, 714) → (1952, 714) → (1952, 856) | (1952, 864) D | физика месяца @ (1992, 706) |

---

## T1 return (month loop)

West rail through zip/case is illegal: an H at y = 264 from x < 952 crosses those nodes.
A 4-segment U (down–left–up–right) is illegal (three elbows).

**Two arrows. East gutter of `state`, x = 1272** (mid of 1232 … 1312). Empty on every row T1 uses.

### R1 — one elbow, rust, dashed `6 4`

GDM left-mid → left along МИР (only `gdm` lives there) → up the gutter to the right of `state`.

```
line    (2032, 714) → (1272, 714) → (1272, 272)
tip     (1272, 264) UP
head    (1272, 264) (1267, 272) (1277, 272)
label   (1652, 706)  месячный возврат · restart
```

Vertices: x ∈ [1267, 2032], y ∈ [264, 714]. Inside safe.

### R2 — H, rust

```
line    (1272, 264) → (1240, 264)
tip     (1232, 264) LEFT   = state R
head    (1232, 264) (1240, 259) (1240, 269)
label   (1252, 256)  t+1
```

Vertices: x ∈ [1232, 1272], y ∈ [259, 269]. Inside safe.

Meaning: restart from full GDM updates state. Not Excel 2014–2015. CRM / LLM do not close this loop.

---

## T2 nodes

| id | r | c | x | y | w | h | title |
|---|---:|---:|---:|---:|---:|---:|---|
| zip | 0 | 0 | 232 | 228 | 280 | 72 | Model Z |
| opm | 0 | 1 | 592 | 228 | 280 | 72 | Свои прогоны OPM |
| train | 1 | 1 | 592 | 378 | 280 | 72 | Обучение |
| surr | 1 | 2 | 952 | 378 | 280 | 72 | Суррогат мира |
| llm | 2 | 2 | 952 | 528 | 280 | 72 | Ядро МАС |
| many | 2 | 3 | 1312 | 528 | 280 | 72 | Много графиков |
| gate | 3 | 3 | 1312 | 678 | 280 | 72 | Один кандидат |
| rep | 4 | 4 | 1672 | 828 | 280 | 72 | Один replay OPM |
| chdd | 4 | 5 | 2032 | 828 | 280 | 72 | CHDD_PYTHON |
| inc | 5 | 4 | 1672 | 978 | 280 | 72 | wells_schedule.inc |
| org | 5 | 5 | 2032 | 978 | 280 | 72 | Организаторы |

Wash: train / surr / many / rep / chdd → `--gold-wash`. gate / inc / org → `--rust-wash`.
No T1 month return on T2.

### T2 edges

H:

| from → to | line | tip | label |
|---|---|---|---|
| zip → opm | (512, 264) → (584, 264) | (592, 264) R | неизменяемый deck |
| train → surr | (872, 414) → (944, 414) | (952, 414) R | признаки / траектории |
| llm → many | (1232, 564) → (1304, 564) | (1312, 564) R | намерения |
| rep → chdd | (1952, 864) → (2024, 864) | (2032, 864) R | 14 полей overlay |
| inc → org | (1952, 1014) → (2024, 1014) | (2032, 1014) R | график + число · rust |

V:

| from → to | line | tip | label |
|---|---|---|---|
| opm → train | (732, 300) → (732, 370) | (732, 378) D | свои сценарии |
| surr → llm | (1092, 450) → (1092, 520) | (1092, 528) D | дешёвый мир |
| many → gate | (1452, 600) → (1452, 670) | (1452, 678) D | лучший график · rust |
| rep → inc | (1812, 900) → (1812, 970) | (1812, 978) D | принятый план |
| chdd → org | (2172, 900) → (2172, 970) | (2172, 978) D | заявленный ЧДД · rust |

Elbow (one corner, rust) — **not** H–V–H:

| from → to | points | tip | label |
|---|---|---|---|
| gate → rep | (1592, 714) → (1592, 864) → (1664, 864) | (1672, 864) R | на контрольный replay @ (1602, 789) |

V sits in the 80 px gutter 1592 … 1672. No node on that x.

---

## Header chips / colophon

T1 chips: `LLM / агенты` · `CRM` · `ГДМ` · `CHDD_PYTHON` (last bar rust).
T2 chips: `Суррогат` · `LLM / агенты` · `OPM replay` (bar rust) · `CHDD_PYTHON` (bar rust).

```
T1 colo L: TIMESOIL · ТРЕК 1 · ДОРОЖКИ 2400×1350 · СТРЕЛКИ ВНУТРИ ПОЛЯ
T1 colo R: LLM = ДЕЙСТВИЯ · CRM = ОТСЕВ · ГДМ = ФИЗИКА · CHDD = ДЕНЬГИ
T2 colo L: TIMESOIL · ТРЕК 2 · ДОРОЖКИ 2400×1350 · TIMESOIL ≠ СУРРОГАТ
T2 colo R: СУРРОГАТ = ПОИСК · OPM = ИСТИНА · CHDD = ДЕНЬГИ · 2007 ≠ 1991 ≠ 4418
```

---

## Forbidden

- `marker-*`, cubic, quadratic, diagonal, fault, grain
- any vertex or label outside 48 … 2352 × 48 … 1302
- `--gold` as type
- webfont `@import` inside the SVG
- drawing empty slots
- T1 loop as one polyline with 2+ elbows
- T2 gate → replay as three segments
- writing SVG in this file; do not retarget `track1.svg` / `track2.svg` (posters)
