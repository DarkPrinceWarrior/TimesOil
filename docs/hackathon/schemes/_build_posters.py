#!/usr/bin/env python3
"""16:9 AIOS posters. Stdlib only. Geometry stays inside 48..2352 × 48..1302."""

from __future__ import annotations

from pathlib import Path

W, H = 2400, 1350
OUT = Path(__file__).resolve().parent
XS = (48, 630, 1212, 1794)
CW, CH = 558, 328
Y1, Y2 = 248, 620
INK = "#1C1914"
MUTED = "#5A544A"
GOLD = "#C4A35A"
GOLD_INK = "#6B5012"
GOLD_WASH = "#F4EBD4"
RUST = "#8B3228"
RUST_WASH = "#F3E4DF"
CARD = "#FFFCF6"
PAPER = "#F3EEE4"
LINE = "#D4CBB8"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tspan(x: int, lines: list[str], dy: int = 18) -> str:
    parts = [f'<tspan x="{x}" dy="0">{esc(lines[0])}</tspan>']
    parts.extend(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>' for line in lines[1:])
    return "".join(parts)


def chip(x: int, y: int, w: int, label: str, stroke: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="2" fill="{CARD}" '
        f'stroke="{stroke}" stroke-width="0.75"/>'
        f'<text class="sans chip" x="{x + 8}" y="{y + 14}">{esc(label)}</text>'
    )


def card(
    x: int,
    y: int,
    num: str,
    title: str,
    bullets: list[str],
    chips: list[tuple[int, str]],
    *,
    bar: str,
    fill: str = CARD,
    stroke: str = LINE,
    badge: str = INK,
) -> str:
    body = "".join(
        f'<text class="sans body" x="{x + 20}" y="{y + 92 + i * 36}">{esc("·  " + b)}</text>'
        for i, b in enumerate(bullets)
    )
    cx, cy = x + 20, y + CH - 36
    chips_svg = []
    for cw, label in chips:
        chips_svg.append(chip(cx, cy, cw, label, bar))
        cx += cw + 8
    return f"""
  <g id="s{num}">
    <rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>
    <rect x="{x}" y="{y}" width="3" height="{CH}" fill="{bar}"/>
    <rect x="{x + 16}" y="{y + 16}" width="22" height="22" rx="2" fill="{badge}"/>
    <text class="sans idx" x="{x + 27}" y="{y + 32}" text-anchor="middle">{esc(num)}</text>
    <text class="serif st" x="{x + 46}" y="{y + 34}">{esc(title)}</text>
    {body}
    {"".join(chips_svg)}
  </g>"""


def kpi(x: int, value: str, label: str) -> str:
    return (
        f'<text class="serif statn" x="{x}" y="86">{esc(value)}</text>'
        f'<text class="sans statl" x="{x}" y="106">{esc(label)}</text>'
    )


def packet(x: int, y: int, w: int, tag: str, title: str, lines: list[str], accent: str) -> str:
    return f"""
  <g>
    <rect x="{x}" y="{y}" width="{w}" height="240" rx="2" fill="{INK}"/>
    <rect x="{x}" y="{y}" width="3" height="240" fill="{accent}"/>
    <text class="sans tag" x="{x + 18}" y="{y + 28}" fill="{accent}">{esc(tag)}</text>
    <text class="serif pkt" x="{x + 18}" y="{y + 58}" fill="{PAPER}">{esc(title)}</text>
    <text class="sans body2" x="{x + 18}" y="{y + 88}" fill="#D8D0BE">{tspan(x + 18, lines)}</text>
  </g>"""


def ribbon(steps: list[str]) -> str:
    # 8 cells, chevrons in 16px gutters, all inside x=48..2352
    cell_w = 270
    gap = 16
    x0 = 48
    parts = []
    for i, label in enumerate(steps):
        x = x0 + i * (cell_w + gap)
        parts.append(
            f'<rect x="{x}" y="168" width="{cell_w}" height="44" rx="2" fill="{CARD}" stroke="{LINE}"/>'
            f'<rect x="{x + 10}" y="179" width="22" height="22" rx="2" fill="{INK}"/>'
            f'<text class="sans idx" x="{x + 21}" y="195" text-anchor="middle">{i + 1:02d}</text>'
            f'<text class="sans rib" x="{x + 40}" y="196">{esc(label)}</text>'
        )
        if i < len(steps) - 1:
            cx = x + cell_w + 4
            parts.append(
                f'<polygon aria-hidden="true" points="{cx},185 {cx + 8},190 {cx},195" fill="{GOLD}"/>'
            )
    return "\n".join(parts)


HEAD = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" lang="ru" xml:lang="ru"
     viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-labelledby="title desc"
     preserveAspectRatio="xMidYMid meet">
  <defs>
    <style type="text/css"><![CDATA[
      .serif {{ font-family: Georgia, "Times New Roman", Times, serif; }}
      .sans  {{ font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
      .kicker {{ font-size: 11px; letter-spacing: 0.16em; font-weight: 600; fill: {GOLD_INK}; }}
      .h1 {{ font-size: 32px; fill: {INK}; }}
      .lede {{ font-size: 13px; fill: {MUTED}; }}
      .statn {{ font-size: 22px; fill: {INK}; }}
      .statl {{ font-size: 11px; fill: {MUTED}; }}
      .idx {{ font-size: 11px; font-weight: 600; fill: {PAPER}; }}
      .st {{ font-size: 16px; fill: {INK}; }}
      .body {{ font-size: 13px; fill: {INK}; }}
      .body2 {{ font-size: 13px; }}
      .chip {{ font-size: 11px; font-weight: 600; fill: {INK}; }}
      .rib {{ font-size: 13px; font-weight: 600; fill: {INK}; }}
      .tag {{ font-size: 10px; letter-spacing: 0.16em; font-weight: 700; }}
      .pkt {{ font-size: 22px; }}
      .note {{ font-size: 13px; fill: {INK}; }}
      .colo {{ font-size: 11px; fill: {MUTED}; letter-spacing: 0.06em; }}
    ]]></style>
  </defs>
  <rect width="{W}" height="{H}" fill="{PAPER}"/>
  <rect x="48" y="48" width="2304" height="1254" fill="none" stroke="{GOLD}" stroke-width="1.25"/>
  <rect x="54" y="54" width="2292" height="1242" fill="none" stroke="{LINE}" stroke-width="0.75"/>
"""


def track1() -> str:
    cards = [
        ("01", "Приём Model Y",
         ["Официальный архив Model_Y (3).zip. Deck MODEL_Y.DATA, START 01 MAY 2007.",
          "Сетка 49×47×141. Фонд — 49 скважин. Генератор — tNavigator.",
          "TimesOil Excel — история для отсева, не deck сдачи."],
         [(148, "Model_Y (3).zip"), (128, "MODEL_Y.DATA"), (168, "DemoSpe_002_2_sch.inc")],
         GOLD, CARD, LINE, INK),
        ("02", "Кейс и лимиты",
         ["Тестовый кейс — то же поле Model Y. Меняются ограничения и, возможно, история.",
          "Сейчас песочница без жёстких лимитов. На тесте ждут лимит воды.",
          "История до 01.01.2014 — стартовое состояние, не цель оптимизации."],
         [(148, "тестовый кейс орг."), (78, "ТЗ §5.2"), (96, "ТЗ §3.2.3")],
         GOLD, CARD, LINE, INK),
        ("03", "Состояние фонда",
         ["33 добывающие + 16 нагнетательных. Блоки A / B / B2 / C / D / E.",
          "Каноническое состояние: режимы, давления, WEFF, обводнённость.",
          "THP в выгрузке — пластовое. DobG — жидкость. 2015-12 — мусор."],
         [(96, "WELSPECS"), (96, "WCONPROD"), (90, "WCONINJE")],
         GOLD, CARD, LINE, INK),
        ("04", "Намерения МАС",
         ["Оценка, решение, воздействие, повторная оценка. LLM в ядре обязательна.",
          "Агенты выбирают стоп, старт, режим, перевод. Не дебиты и не ЧДД.",
          "TimesOil CRM / CRM2P / CRMP / Джентил — только предварительный отсев."],
         [(118, "ControlIntent"), (118, "ConstraintSet"), (110, "decision card")],
         GOLD, CARD, LINE, INK),
        ("05", "График скважин",
         ["Компилятор пишет include. Сдача — wells_schedule.inc, не CSV.",
          "В цикле — фрагмент месяца; к горизонту — полный файл без ручной правки.",
          "Parser round-trip обязателен. В файл идёт только первый месяц плана."],
         [(158, "wells_schedule.inc"), (68, "DATES"), (96, "WCONPROD")],
         GOLD, GOLD_WASH, GOLD, INK),
        ("06", "Полная ГДМ",
         ["Каждый принятый месяц — полный прогон. ГДМ — физика, не деньги.",
          "Новое состояние берётся из restart, не из CRM и не из LLM.",
          "TimesOil — отсев кандидатов, не замена этого шага."],
         [(128, "MODEL_Y.DATA"), (78, "restart"), (96, "script_1.py")],
         GOLD, GOLD_WASH, GOLD, INK),
        ("07", "Официальный ЧДД",
         ["Только CHDD_PYTHON, явный --start-year 2014. Других калькуляторов нет.",
          "14 полей. Нефть и жидкость — тонны, закачка — м³. Учёт с 01.01.2014.",
          "Выход: одно число — заявленный ЧДД, млн руб."],
         [(118, "CHDD_PYTHON"), (128, "РАСЧЕТ_ЧДД.py"), (148, "--start-year 2014")],
         GOLD, GOLD_WASH, GOLD, INK),
        ("08", "Проверка организаторов",
         ["Этап 1: автовалидация. Рейтинг по заявленному ЧДД, топ 20 %, не более 10.",
          "Этап 2: один прогон сданного wells_schedule.inc и тот же CHDD_PYTHON.",
          "Расхождение с заявкой — дисквалификация. Повторных прогонов нет."],
         [(158, "wells_schedule.inc"), (128, "заявленный ЧДД"), (118, "CHDD_PYTHON")],
         RUST, RUST_WASH, RUST, RUST),
    ]
    steps = ["Приём", "Кейс", "Фонд", "Намерения", "График", "ГДМ", "ЧДД", "Проверка"]
    body = f"""  <title id="title">Трек 1 · от Model Y до проверки организаторов</title>
  <desc id="desc">Восемь этапов слева направо. Полная ГДМ каждый месяц. CHDD с 2014. Пакет: график, заявленный ЧДД, код и экран.</desc>
  <text class="sans kicker" x="64" y="78">AIOS · ТАТНЕФТЬ · ТРЕК 1 · MODEL Y · 49×47×141</text>
  <text class="serif h1" x="64" y="112">От Model Y до проверки организаторов</text>
  <text class="sans lede" x="64" y="136">Агенты пишут действия. ГДМ — физика. CHDD_PYTHON — деньги. TimesOil CRM — только отсев.</text>
  {kpi(1648, "49", "скважин, блоки A–E")}
  {kpi(1848, "01.01.2014", "старт учёта ЧДД")}
  {kpi(2048, "1 мес", "шаг полной ГДМ")}
  {kpi(2200, "1 прогон", "проверка организаторов")}
  <line x1="64" y1="152" x2="2336" y2="152" stroke="{LINE}"/>
  {ribbon(steps)}
"""
    for i, spec in enumerate(cards):
        x = XS[i % 4]
        y = Y1 if i < 4 else Y2
        body += card(x, y, spec[0], spec[1], spec[2], spec[3], bar=spec[4], fill=spec[5], stroke=spec[6], badge=spec[7])
    body += f"""
  <rect x="48" y="576" width="2304" height="44" rx="2" fill="{RUST_WASH}" stroke="{RUST}" stroke-dasharray="5 4"/>
  <text class="sans note" x="64" y="604">Месячный цикл  ·  06 ГДМ обновляет состояние и снова входит в 03  ·  повтор до конца горизонта  ·  учёт с 01.01.2014  ·  цель — ЧДД своего прогона, не история 2014–2015</text>
  {packet(48, 968, 752, "01  ·  ГРАФИК", "wells_schedule.inc",
          ["Управляющий include под Model Y.", "Собран компилятором. После старта руками не правят.", "CSV — не этот файл. ТЗ §3.2.3, §3.3.2, §3.4."], GOLD)}
  {packet(824, 968, 752, "02  ·  ДЕНЬГИ", "Заявленный ЧДД",
          ["Только CHDD_PYTHON, старт 01.01.2014.", "Таблица из своей полной ГДМ.", "Расхождение с пересчётом организаторов — дисквалификация."], GOLD)}
  {packet(1600, 968, 752, "03  ·  КОД / UI", "Исходники и экран",
          ["Код МАС, документация, инструкция, конфиг.", "Простой UI запуска и разбора. LLM в ядре.", "Docker — только если контейнеризуете."], GOLD)}
  <text class="sans colo" x="64" y="1248">TIMESOIL · ТРЕК 1 · 2400×1350 · 2026-08-17</text>
  <text class="sans colo" x="2336" y="1248" text-anchor="end">LLM = ДЕЙСТВИЯ · CRM = ОТСЕВ · ГДМ = ФИЗИКА МЕСЯЦА · CHDD_PYTHON = ДЕНЬГИ</text>
</svg>
"""
    return HEAD + body


def track2() -> str:
    search = [
        ("01", "Вход Model Z",
         ["Официальный архив Model_Z_final_OPM.zip. Оригинальный deck не менять.",
          "~103 WELSPECS: 92 когда-либо добывающих, 41 когда-либо нагнетательная.",
          "Model_Z_summary.inc пуст — SUMMARY только внешним overlay."],
         [(178, "Model_Z_final_OPM.zip"), (148, "START 01 JUN 1991"), (128, "overlay SUMMARY")],
         GOLD, CARD, LINE, INK),
        ("02", "Свои прогоны OPM",
         ["Дампов организаторов нет. Гидродинамику снимает команда на OPM Flow.",
          "Семейства: replay, импульсы, Sobol, stop/start, границы.",
          "TimesOil — не датасет Model Z."],
         [(96, "OPM Flow"), (128, "свои сценарии"), (96, "нет дампов")],
         GOLD, CARD, LINE, INK),
        ("03", "Суррогат мира",
         ["Гибрид: CRM / матбаланс + обучаемый остаток. Шлюз UQ / OOD.",
          "Метрика поиска — WAPE и черновик ЧДД. Черновик не заявляют.",
          "TimesOil прогнозирует Model Y на 6 месяцев — это другой контур."],
         [(96, "CRM / CRMP"), (78, "UQ / OOD"), (68, "WAPE")],
         GOLD, GOLD_WASH, GOLD, INK),
        ("04", "Агенты на суррогате",
         ["Тот же цикл МАС, но мир = суррогат, не полная ГДМ.",
          "LLM пишет намерения. Официальный CHDD дёшев на каждом плане.",
          "Много полных графиков. Один лучший уходит в сертификацию."],
         [(96, "намерения"), (68, "поиск"), (118, "черновик ЧДД")],
         GOLD, CARD, LINE, INK),
    ]
    certify = [
        ("05", "Один replay OPM",
         ["Не помесячный цикл трека 1. Лучший график гоняется целиком.",
          "Единственная физика, которую можно заявлять.",
          "Выход: помесячная таблица 14 полей из overlay, не из суррогата."],
         [(178, "Model_Z_final_OPM"), (96, "весь график"), (78, "один раз")],
         RUST, GOLD_WASH, GOLD, INK),
        ("06", "CHDD_PYTHON · 2007",
         ["Обязателен флаг --start-year 2007. Без флага берётся 1991.",
          "Число 4418,132 млн руб. со старта 1991 — не контроль трека 2.",
          "Выход: одно заявленное число, млн руб."],
         [(148, "--start-year 2007"), (128, "РАСЧЕТ_ЧДД.py"), (88, "WLPR ≤ 500")],
         RUST, GOLD_WASH, GOLD, INK),
        ("07", "wells_schedule.inc",
         ["Компилятор пишет Eclipse / OPM include из принятого плана.",
          "CSV — вход калькулятора, не формат сдачи.",
          "После генерации руками не правят. INCLUDE только в overlay."],
         [(158, "wells_schedule.inc"), (78, "не CSV"), (128, "overlay INCLUDE")],
         RUST, CARD, LINE, INK),
        ("08", "Организаторы",
         ["Один прогон ГДМ по сданному графику и тот же CHDD_PYTHON.",
          "Расхождение с заявленным ЧДД — дисквалификация. Повтора нет.",
          "Люфт симулятора ~2 %. Защита 7+5 мин. Ранжирование по ЧДД."],
         [(88, "один ГДМ"), (68, "~2 %"), (118, "топ 20 % ≤ 10")],
         RUST, RUST_WASH, RUST, RUST),
    ]
    body = f"""  <title id="title">Трек 2 · от Model Z до пакета организаторов</title>
  <desc id="desc">Поиск на суррогате, один replay OPM, CHDD с 2007. TimesOil не суррогат. Пакет: код, график, ЧДД, метрики.</desc>
  <text class="sans kicker" x="64" y="78">AIOS · ТАТНЕФТЬ · ТРЕК 2 · MODEL Z · 91×102×59</text>
  <text class="serif h1" x="64" y="112">От Model Z до пакета организаторов</text>
  <text class="sans lede" x="64" y="136">Суррогат ищет. Один replay сертифицирует. TimesOil — не этот суррогат. CHDD с 01.01.2007.</text>
  {kpi(1608, "103", "WELSPECS · 92 / 41")}
  {kpi(1828, "01.01.2007", "старт учёта ЧДД")}
  {kpi(2040, "много", "графиков на суррогате")}
  {kpi(2220, "1", "дорогой прогон OPM")}
  <line x1="64" y1="152" x2="2336" y2="152" stroke="{LINE}"/>
  <rect x="48" y="168" width="1146" height="44" rx="2" fill="{GOLD_WASH}" stroke="{GOLD}"/>
  <text class="sans rib" x="64" y="196">01–04  ·  дешёвый поиск  ·  много графиков на суррогате</text>
  <rect x="1212" y="168" width="1140" height="44" rx="2" fill="{RUST_WASH}" stroke="{RUST}"/>
  <text class="sans rib" x="1228" y="196">05–08  ·  один дорогой прогон  ·  один replay сертифицирует</text>
"""
    for i, spec in enumerate(search):
        body += card(XS[i], Y1, spec[0], spec[1], spec[2], spec[3], bar=spec[4], fill=spec[5], stroke=spec[6], badge=spec[7])
    body += f"""
  <rect x="48" y="576" width="2304" height="44" rx="2" fill="{INK}"/>
  <text class="sans rib" x="1200" y="604" text-anchor="middle" fill="{PAPER}">Шлюз  ·  один полный график  ·  04 агенты передают 05 replay OPM  ·  остальные отсекаются  ·  суррогат не сертифицирует</text>
"""
    for i, spec in enumerate(certify):
        body += card(XS[i], Y2, spec[0], spec[1], spec[2], spec[3], bar=spec[4], fill=spec[5], stroke=spec[6], badge=spec[7])
    body += f"""
  {packet(48, 968, 558, "01  ·  КОД / UI", "Исходники",
          ["МАС, инструкция, конфиг.", "Интерфейс запуска и разбора.", "Docker — только если контейнер."], GOLD)}
  {packet(630, 968, 558, "02  ·  ГРАФИК", "wells_schedule.inc",
          ["Управляющий include Model Z.", "Без ручной правки после сборки.", "CSV — вход CHDD, не сдача."], GOLD)}
  {packet(1212, 968, 558, "03  ·  ЧДД", "Заявленное число",
          ["Только CHDD_PYTHON, старт 2007.", "Таблица из своего replay.", "Люфт ~2 %. Повтора нет."], GOLD)}
  {packet(1794, 968, 558, "04  ·  СУРРОГАТ", "Модель и метрики",
          ["Способ воспроизвести и веса.", "WAPE на своих прогонах OPM.", "TimesOil сюда не класть."], GOLD)}
  <text class="sans colo" x="64" y="1248">TIMESOIL · ТРЕК 2 · 2400×1350 · 2026-08-17</text>
  <text class="sans colo" x="2336" y="1248" text-anchor="end">СУРРОГАТ = ПОИСК · OPM = ИСТИНА ПЛАНА · CHDD = ДЕНЬГИ · 2007 ≠ 1991 ≠ 4418</text>
</svg>
"""
    return HEAD + body


def main() -> None:
    (OUT / "track1.svg").write_text(track1(), encoding="utf-8")
    (OUT / "track2.svg").write_text(track2(), encoding="utf-8")
    for name in ("track1.svg", "track2.svg"):
        text = (OUT / name).read_text(encoding="utf-8")
        assert 'viewBox="0 0 2400 1350"' in text
        assert 'y="-40"' not in text
        assert "2240" not in text
        assert "marker-end" not in text
    print("wrote", OUT / "track1.svg", OUT / "track2.svg")


if __name__ == "__main__":
    main()
