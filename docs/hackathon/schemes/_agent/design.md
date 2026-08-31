# Design system — AIOS process diagrams

**Date:** 2026-08-17
**Scope:** two standalone landscape SVGs, Track 1 and Track 2.
**This file:** spec only. Do not paste a full drawing here.

---

## Decision

**One system. Light boardroom. Both tracks.**

Canvas is cream paper, ink type, one gold, one rust. Dark “oil field” is rejected for these two files.

Dark *can* hit WCAG AA (`#E8E0D0` on `#1A1C19` = 13.1:1). It still fails the brief:

| Failure | Evidence in current files |
|---|---|
| Not 16:9 | `track1.svg` / `track2.svg` are `1600×2200` — laptop must scroll |
| Sequence painted, not numbered | SVG `marker` arrows + lines that leave the viewBox (`y="-40"` / `y="2240"`) |
| Texture eats type | `feTurbulence` grain + 80 px gold grid + diagonal fault |
| Stacked rows | eight full-width 188 px bands + a 168 px “пакет сдачи” footer |
| Webfonts | Instrument Serif / IBM Plex — missing in a file opened as SVG |

McKinsey-grade here means: one screen, one reading order, no atmosphere.

---

## Canvas

| Token | Value | Rule |
|---|---|---|
| `viewBox` | `0 0 1920 1080` | identical on both files |
| `width` / `height` | `1920` / `1080` | plus `width="100%"` `height="auto"` |
| Aspect | 16:9 | `preserveAspectRatio="xMidYMid meet"` |
| Safe inset | **32 px** | no stroke, text, badge, or chevron outside |
| Hard clip | **24 px** from every viewBox edge | geometry that would sit closer is deleted, not clipped |
| Embed | `max-height: 100vh` | laptop, no page scroll |

Bands (y, both files):

| Band | y | h |
|---|---|---|
| Header | 32 | 88 |
| Numbered ribbon | 128 | 44 |
| Main grid | 184 | 820 |
| Footer | 1016 | 32 |

No second gold frame. One 1.25 px hairline at y = 120 and y = 1008, inset 32 px.

---

## Kill list (current posters)

Do not carry forward:

- `1600×2200`
- `#1A1C19` field, mast glow, grain filter, 80 px grid
- diagonal fault gradient
- double gold rectangle frame
- full-width stacked “пласт” rows
- `marker-end` / `marker-mid` arrows
- lines that start or end outside the viewBox
- `@import` / Google Fonts inside the SVG
- Instrument Serif or IBM Plex as **required** faces
- more than three body lines per card

---

## Type

Standalone SVG. No webfont request. Stacks only.

```css
--font-serif: Georgia, "Times New Roman", Times, serif;
--font-sans:  "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
```

Cyrillic lives in these faces on Windows / macOS / typical Linux office boxes. If a face is missing, the next one still renders.

Optional Google Fonts `@import` is **off** in the SVG. The HTML index may load display faces; the diagram file must survive `file://` and e-mail attachment.

| Role | Class | Face | Size | Weight | Line | Tracking | Color |
|---|---|---|---|---|---|---|---|
| Kicker | `.kicker` | sans | 11 | 600 | 1.0 | 0.16 em | `--gold-ink` |
| Title | `.h1` | serif | 32 | 400 | 1.05 | 0 | `--ink` |
| Lede | `.lede` | sans | 13 | 400 | 1.25 | 0 | `--muted` |
| Stat number | `.statn` | serif | 22 | 400 | 1.0 | 0 | `--ink` |
| Stat label | `.statl` | sans | 11 | 400 | 1.2 | 0 | `--muted` |
| Card title | `.st` | serif | 16 | 400 | 1.15 | 0 | `--ink` |
| Body | `.body` | sans | 12 | 400 | 1.35 | 0 | `--ink` |
| Rule / path | `.rule` | sans | 11 | 400 | 1.3 | 0 | `--muted` |
| Chip | `.chip` | sans | 10.5 | 600 | 1.0 | 0.02 em | `--ink` |
| Badge | `.idx` | sans | 11 | 600 | 1.0 | 0.04 em | `--paper` |
| Footer | `.colo` | sans | 10 | 400 | 1.0 | 0.06 em | `--muted` |

Title is one line, max ~42 characters. Italic gold word (model name) uses `--gold-ink`, not `--gold`.

---

## Color

| Token | Hex | Use | Not for |
|---|---|---|---|
| `--paper` | `#F3EEE4` | canvas | type |
| `--card` | `#FFFCF6` | card fill | |
| `--ink` | `#1C1914` | titles, body, badge fill | hairlines |
| `--muted` | `#5A544A` | lede, footer, `.rule` | titles |
| `--gold` | `#C4A35A` | 1–1.25 px rules, chip stroke, chevron fill | **any text** |
| `--gold-ink` | `#6B5012` | kicker, italic model name, ribbon label | large fills |
| `--gold-wash` | `#F4EBD4` | highlight card (ГДМ / суррогат) | type |
| `--rust` | `#8B3228` | risk stroke, risk badge, risk chip | decoration |
| `--rust-wash` | `#F3E4DF` | risk card (проверка / дисквалификация) | type |
| `--line` | `#D4CBB8` | card stroke, ribbon cell stroke | text |

`--gold` on `--paper` = **2.08:1 — fail**. Keep it as metal, not ink.

Measured contrast (WCAG 2.2, relative luminance):

| Pair | Ratio | Bar |
|---|---|---|
| `--ink` on `--paper` | 15.2:1 | AA / AAA body |
| `--muted` on `--paper` | 6.5:1 | AA body |
| `--gold-ink` on `--paper` | 6.5:1 | AA body |
| `--rust` on `--paper` | 7.0:1 | AA body |
| `--paper` on `--ink` | 15.2:1 | AA badge |
| `--paper` on `--rust` | 7.0:1 | AA risk badge |
| `--ink` on `--gold-wash` | 14.8:1 | AA |
| `--ink` on `--rust-wash` | 14.2:1 | AA |
| `--gold` on `--paper` | 2.08:1 | **text forbidden** |

---

## Stroke, radius, chip, badge

| Part | Value |
|---|---|
| Canvas hairline | 1.25 px `--gold` |
| Card stroke | 1 px `--line` |
| Highlight card stroke | 1.15 px `--gold` |
| Risk card stroke | 1.15 px `--rust` |
| Chip stroke | 0.75 px `--gold` or `--rust` |
| Ribbon cell stroke | 1 px `--line` |
| Card radius | **2 px** |
| Chip radius | **2 px** |
| Badge radius | **2 px** (square, not circle) |
| Chip height | 20 px |
| Chip pad x | 8 px |
| Chip fill | `--card` (default) or `--gold-wash` / `--rust-wash` |
| Badge | 22×22, fill `--ink`, type `--paper` |
| Risk badge | 22×22, fill `--rust`, type `--paper` |
| Shadow | none |
| Dash | risk callout only, `5 4`, `--rust`, never as a flow arrow |

Max **three** chips per card, one row. Overflow chips are dropped, not wrapped.

---

## Connectors (no overflow)

Sequence is **numbers first**, chevrons second. Never a path that needs a marker.

### 1. Numbered ribbon

Horizontal strip, y = 128, x = 32 … 1888, height 44.

Eight equal cells (both tracks). Label = `01` … `08` plus a 10–14 character name.

Between cells: a **chevron in the gutter**, not a line into the next cell.

```
cell gap = 16
chevron  = 8 × 10, fill --gold
polygon  = (0,0) (8,5) (0,10)
```

First cell starts at x = 32. Last cell ends at x = 1888. Chevrons sit in the 16 px gap, vertically centered on the ribbon. Distance from viewBox edge to any chevron vertex ≥ 40 px.

Ribbon is the only wrap cue. Card 4 does **not** grow a snake arrow down to card 5.

### 2. Adjacent-card chevrons

In the **gutter between two cards in the same row** (and only there):

- gutter 16 px
- same 8×10 gold polygon, centered in the gutter, vertically mid-card
- risk-to-risk uses `--rust` fill
- no stroke on the polygon
- no `<marker>`
- no polyline, no cubic, no elbow

Vertical sequence (row 1 → row 2) is the ribbon number, not a down-arrow.

### 3. Forbidden

- `marker-start` / `marker-mid` / `marker-end`
- any coordinate `< 24` or `> 1896` (x) / `> 1056` (y)
- arrows that point at the frame
- a “flow spine” down the left margin (current gold fade line)

---

## Layout

### Shared header (88 px)

Left: kicker + one-line `.h1` + one-line `.lede`.
Right: 420×72 invariant card — four stats in 2×2, no quote line.

Track 1 kicker: `AIOS · ТАТНЕФТЬ · ТРЕК 1 · MODEL Y · 49×47×141`
Track 2 kicker: `AIOS · ТАТНЕФТЬ · ТРЕК 2 · MODEL Z · 91×102×59`

### Track 1 — one loop, 2×4

Main: four columns × two rows. Gutter 16. Card ≈ 452×402.

| # | Title | Tint |
|---|---|---|
| 01 | Приём Model Y | default |
| 02 | Кейс и лимиты | default |
| 03 | Состояние фонда | default |
| 04 | МАС: намерения | default |
| 05 | График скважин | `--gold-wash` |
| 06 | Полная ГДМ | `--gold-wash` |
| 07 | Официальный ЧДД | `--gold-wash` |
| 08 | Проверка орг. | `--rust-wash` |

Card copy ceiling: **title + 3 body lines + 3 chips + 1 rule**. The old 8-line dump does not fit 16:9; cut, do not shrink type below the scale.

### Track 2 — two lanes, same grid

Same 2×4 and same ribbon. Lane is a 2 px left bar on the card, not a new chrome:

- 01–04 search / surrogate: bar `--gold`
- 05–08 control replay / verify: bar `--rust`

Do not draw a vertical fault between lanes.

### Footer (32 px)

Left: `TIMESOIL · ТРЕК n · 1920×1080 · 2026-08-17`
Right: one formula, Track 1 `LLM = НАМЕРЕНИЯ · CRM = ОТСЕВ · ГДМ = ИСТИНА · CHDD = ДЕНЬГИ`
Track 2 swaps the middle to `СУРРОГАТ = ПОИСК · OPM = ФИЗИЧЕСКАЯ ВЕРИФИКАЦИЯ`.

No third “пакет сдачи” row. Those three artefacts become chips on cards 05 / 07 / 08.

---

## Accessibility

Every root `<svg>`:

```xml
<svg xmlns="http://www.w3.org/2000/svg" lang="ru" xml:lang="ru"
     viewBox="0 0 1920 1080" width="1920" height="1080"
     role="img" aria-labelledby="title desc"
     preserveAspectRatio="xMidYMid meet">
  <title id="title">…</title>
  <desc id="desc">…</desc>
```

| File | `title` | `desc` must name |
|---|---|---|
| Track 1 | Трек 1 · от Model Y до проверки организаторов | eight steps, monthly GDM, CHDD from 2014, package = schedule + NPV + UI |
| Track 2 | Трек 2 · от Model Z до пакета организаторов | surrogate search vs one OPM replay, CHDD from 2007, TimesOil is not the surrogate |

Further:

- Do not encode meaning in gold vs rust alone — badge number + word “риск” on card 08
- Decorative chevrons: `aria-hidden="true"`
- Minimum type 10 px (footer). Body stays 12
- Focus is not required (static image). If the SVG is inlined in HTML, the `<title>` still stands

---

## Shared CSS tokens

Copy this block into **both** files. Do not restyle per track except lane bar and card wash classes.

```css
:root {
  --paper: #F3EEE4;
  --card: #FFFCF6;
  --ink: #1C1914;
  --muted: #5A544A;
  --gold: #C4A35A;
  --gold-ink: #6B5012;
  --gold-wash: #F4EBD4;
  --rust: #8B3228;
  --rust-wash: #F3E4DF;
  --line: #D4CBB8;
  --font-serif: Georgia, "Times New Roman", Times, serif;
  --font-sans: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --stroke-hair: 1.25;
  --stroke-card: 1;
  --stroke-chip: 0.75;
  --radius: 2;
  --pad: 32;
  --safe: 24;
  --gutter: 16;
  --badge: 22;
  --chip-h: 20;
  --ribbon-h: 44;
}
.serif { font-family: var(--font-serif); }
.sans  { font-family: var(--font-sans); }
.kicker { font-size: 11px; font-weight: 600; letter-spacing: 0.16em; fill: var(--gold-ink); }
.h1    { font-size: 32px; fill: var(--ink); }
.lede  { font-size: 13px; fill: var(--muted); }
.statn { font-size: 22px; fill: var(--ink); }
.statl { font-size: 11px; fill: var(--muted); }
.st    { font-size: 16px; fill: var(--ink); }
.body  { font-size: 12px; fill: var(--ink); }
.rule  { font-size: 11px; fill: var(--muted); }
.chip  { font-size: 10.5px; font-weight: 600; fill: var(--ink); }
.idx   { font-size: 11px; font-weight: 600; fill: var(--paper); }
.colo  { font-size: 10px; letter-spacing: 0.06em; fill: var(--muted); }
```

SVG presentation: fill the root rect with `var(--paper)`. Cards use `var(--card)` / wash tokens. Do not reference `theme.css` — that file is the old dark page theme.

---

## Implementation notes

- One `<style>` in `<defs>`, CDATA, tokens above. No second stylesheet.
- Groups: `#header`, `#ribbon`, `#grid`, `#footer`. Cards `#s01` … `#s08`.
- Text is real `<text>`, not outlines.
- `theme.css` / `index.html` may stay dark until a later pass. These two SVGs do not follow them.
- Content source of truth remains the current posters and `docs/hackathon/architecture_2026-08-17.md`. This spec only constrains how that content is drawn.
