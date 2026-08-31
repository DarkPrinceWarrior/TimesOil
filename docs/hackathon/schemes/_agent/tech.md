# Схемы: один стек, без зависимостей проекта

Дата пробы: 2026-08-17. Хост: WSL2, Ubuntu 24.04, `DISPLAY=:0`.

## Вердикт

**Один стек:** рукописный landscape SVG с фиксированной сеткой + PNG через headless Google Chrome (`--force-device-scale-factor=2`).

Источник правды — `docs/hackathon/schemes/track1.svg` и `track2.svg`.
Открыть / отправить — сам SVG. Слайды и чат — PNG рядом в `render/`.

HTML+CSS (`track1.html`, `track2.html`, `index.html`) — только смотровая поверхность. Не источник PNG: это длинная страница, не один кадр.

В `pyproject.toml` / `uv` ничего не добавлять.

## Почему так (лестница)

1. Сетка плаката фиксированная — движок раскладки не нужен.
2. На машине уже есть Chrome 149 и Node; rsvg / Inkscape / d2 / Graphviz нет.
3. Mermaid (`mmdc` 11.15.0) стоит глобально, но автолейаут даёт высокие flowchart (уже в `src/*.mmd` → `render/*-flow.svg`: 1198×4543 и 663×2375). Для landscape-плаката это лишняя борьба.
4. Pillow в `.venv` есть, растеризатора SVG нет (`cairosvg` / `svglib` отсутствуют). Не ставить.
5. Playwright в PATH и в venv нет. Кэш `~/.cache/ms-playwright/` не использовать.

## Проба инструментов

| Инструмент | Статус | Путь / версия |
|---|---|---|
| Google Chrome | есть | `/usr/bin/google-chrome` → `/opt/google/chrome/google-chrome`, **149.0.7827.200** |
| Node / npm / npx | есть | nvm `v22.20.0` / `11.17.0` |
| `mmdc` (mermaid-cli) | есть, глобально | `~/.nvm/versions/node/v22.20.0/bin/mmdc` **11.15.0** |
| Puppeteer Chrome (для mmdc) | есть | `~/.cache/puppeteer/chrome/linux-147…` и соседние |
| `chromium` / `chromium-browser` | нет | — |
| `playwright` CLI | нет | кэш браузеров есть, в PATH нет |
| python `playwright` / `cairosvg` | нет | в `.venv` только Pillow |
| `rsvg-convert` | нет | — |
| Inkscape | нет | — |
| d2 | нет | — |
| Graphviz (`dot`) | нет | — |
| ImageMagick | нет | — |

Проверено: `--headless=new` пишет PNG. `--no-sandbox` не обязателен. Ошибки DBus/UPower в stderr — шум, игнорировать.

Системные шрифты: DejaVu, Liberation, Free, Noto Color Emoji. IBM Plex / Instrument Serif в SVG заявлены, локально не установлены — Chrome уйдёт в Georgia / Arial, пока нет `@font-face`.

## Холст

Для двух landscape-схем (слайды 16:9, чат):

```text
viewBox="0 0 1920 1080"  width="1920" height="1080"
```

2× PNG = **3840×2160**.

Сейчас в репозитории `track1.svg` / `track2.svg` — портрет `1600×2200` (плакат). Пока viewBox такой, 2× даёт **3200×4400**. Для слайдов сменить viewBox на 1920×1080 и переложить сетку; не крутить портрет в Chrome.

Не ставить `style="max-width:100%;height:auto"` на корневом `<svg>` для файла, который снимает Chrome: окно тогда не совпадает с viewBox.

## Команды PNG 2×

Рабочая директория — корень репозитория. Каталог `render/` уже есть.

### Landscape 1920×1080 → 3840×2160 (целевой)

Проба 2026-08-17: тот же вызов на тестовом SVG дал ровно `3840×2160`.

```bash
ROOT="$PWD/docs/hackathon/schemes"
CHROME=/usr/bin/google-chrome

for name in track1 track2; do
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 \
    --window-size=1920,1080 \
    --default-background-color=FF1A1C19 \
    --virtual-time-budget=2000 \
    --user-data-dir=/tmp/timesoil-chrome-schemes \
    --screenshot="$ROOT/render/${name}.png" \
    "file://$ROOT/${name}.svg"
done
```

`--force-device-scale-factor=2` + окно = CSS-размер SVG. Не раздувать `--window-size` до 3840×2160: получается тот же пиксельный размер, но текст мыльный (проба `probe-win2x.png` легче и хуже).

### Текущие портретные исходники 1600×2200 → 3200×4400

Проба 2026-08-17: `track1.svg` → `3200×4400` (≈14 МБ из-за `feTurbulence` grain), `track2.svg` → `3200×4400` (≈7 МБ).

```bash
ROOT="$PWD/docs/hackathon/schemes"
CHROME=/usr/bin/google-chrome

for name in track1 track2; do
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 \
    --window-size=1600,2200 \
    --default-background-color=FF1A1C19 \
    --virtual-time-budget=2000 \
    --user-data-dir=/tmp/timesoil-chrome-schemes \
    --screenshot="$ROOT/render/${name}.png" \
    "file://$ROOT/${name}.svg"
done
```

`--window-size` всегда равен `width,height` корневого SVG, не 2×.

## Что не делать

- Не добавлять mermaid / d2 / graphviz / cairosvg / playwright в `pyproject.toml`.
- Не снимать `track1.html` / `track2.html` как PNG: это скролл, не кадр.
- Не гонять `mmdc` как основной контур. Глобальный `mmdc` годится только для черновиков в `src/*.mmd`.
- Не ставить Inkscape / rsvg / d2 ради этого. Chrome уже закрывает PNG.
- Не писать экспорт-скрипт в Python-проект.

## Заметки

- `--user-data-dir=/tmp/timesoil-chrome-schemes` — отдельный профиль: на хосте уже крутится чужой headless Chrome (Playwright MCP).
- Grain (`feTurbulence`) на 2× раздувает PNG. Для чата можно снять filter с `<rect>` на время экспорта.
- Веб-шрифты: либо `@font-face` + локальный woff2 внутри SVG, либо смириться с DejaVu/Liberation. Не тянуть Google Fonts в SVG, который открывают офлайн.
- Цвет фона Chrome: `--default-background-color=AARRGGBB` (`FF1A1C19` = непрозрачный `#1a1c19`).
