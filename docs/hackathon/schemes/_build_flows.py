#!/usr/bin/env python3
"""Landscape swimlane flowcharts. Stdlib only. Arrows stay in 48..2352 × 48..1302."""

from __future__ import annotations

from pathlib import Path

W, H = 2400, 1350
OUT = Path(__file__).resolve().parent
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
SAFE = (48, 48, 2352, 1302)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(title: str, desc: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" lang="ru" xml:lang="ru"
     viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-labelledby="title desc"
     preserveAspectRatio="xMidYMid meet">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(desc)}</desc>
  <defs>
    <style type="text/css"><![CDATA[
      .serif {{ font-family: Georgia, "Times New Roman", Times, serif; }}
      .sans  {{ font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
      .kicker {{ font-size: 11px; letter-spacing: 0.16em; font-weight: 600; fill: {GOLD_INK}; }}
      .h1 {{ font-size: 30px; fill: {INK}; }}
      .lede {{ font-size: 13px; fill: {MUTED}; }}
      .lane {{ font-size: 12px; font-weight: 700; letter-spacing: 0.12em; fill: {GOLD_INK}; }}
      .nt {{ font-size: 15px; fill: {INK}; }}
      .ns {{ font-size: 12px; fill: {MUTED}; }}
      .el {{ font-size: 11px; font-weight: 600; fill: {GOLD_INK}; }}
      .role {{ font-size: 12px; fill: {INK}; }}
      .colo {{ font-size: 11px; fill: {MUTED}; letter-spacing: 0.05em; }}
    ]]></style>
  </defs>
  <rect width="{W}" height="{H}" fill="{PAPER}"/>
  <rect x="48" y="48" width="2304" height="1254" fill="none" stroke="{GOLD}" stroke-width="1.25"/>
"""


def node(x: int, y: int, w: int, h: int, title: str, sub: str, *, fill: str = CARD, bar: str = GOLD) -> str:
    return f"""  <g>
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{LINE}"/>
    <rect x="{x}" y="{y}" width="4" height="{h}" fill="{bar}"/>
    <text class="serif nt" x="{x + 16}" y="{y + 28}">{esc(title)}</text>
    <text class="sans ns" x="{x + 16}" y="{y + 50}">{esc(sub)}</text>
  </g>"""


def lane(y: int, h: int, name: str, wash: str) -> str:
    return (
        f'<rect x="48" y="{y}" width="2304" height="{h}" fill="{wash}"/>'
        f'<rect x="48" y="{y}" width="160" height="{h}" fill="#EBE4D4"/>'
        f'<text class="sans lane" x="64" y="{y + h // 2 + 4}">{esc(name)}</text>'
    )


def h_arrow(x1: int, x2: int, y: int, label: str, color: str = GOLD_INK) -> str:
    tip = x2
    shaft = tip - 10
    return (
        f'<line x1="{x1}" y1="{y}" x2="{shaft}" y2="{y}" stroke="{color}" stroke-width="1.6"/>'
        f'<polygon points="{tip},{y} {shaft},{y - 5} {shaft},{y + 5}" fill="{color}"/>'
        f'<text class="sans el" x="{(x1 + shaft) // 2}" y="{y - 8}" text-anchor="middle">{esc(label)}</text>'
    )


def v_arrow(x: int, y1: int, y2: int, label: str, color: str = GOLD_INK) -> str:
    down = y2 > y1
    tip = y2
    shaft = tip - 10 if down else tip + 10
    if down:
        head_pts = f"{x},{tip} {x - 5},{shaft} {x + 5},{shaft}"
    else:
        head_pts = f"{x},{tip} {x - 5},{shaft} {x + 5},{shaft}"
    lx = x + 10
    ly = (y1 + y2) // 2
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{shaft}" stroke="{color}" stroke-width="1.6"/>'
        f'<polygon points="{head_pts}" fill="{color}"/>'
        f'<text class="sans el" x="{lx}" y="{ly}">{esc(label)}</text>'
    )


def elbow_down_right(x1: int, y1: int, x2: int, y2: int, label: str, color: str = GOLD_INK) -> str:
    tip = x2
    shaft = tip - 10
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{x1},{y1} {x1},{y2} {shaft},{y2}"/>'
        f'<polygon points="{tip},{y2} {shaft},{y2 - 5} {shaft},{y2 + 5}" fill="{color}"/>'
        f'<text class="sans el" x="{x1 + 10}" y="{(y1 + y2) // 2}">{esc(label)}</text>'
    )


def elbow_left_up(x1: int, y1: int, x2: int, y2: int, label: str, color: str = RUST) -> str:
    # from GDM left then up to state
    tip = y2
    shaft = tip + 10
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" stroke-dasharray="6 4" '
        f'points="{x1},{y1} {x2},{y1} {x2},{shaft}"/>'
        f'<polygon points="{x2},{tip} {x2 - 5},{shaft} {x2 + 5},{shaft}" fill="{color}"/>'
        f'<text class="sans el" x="{(x1 + x2) // 2}" y="{y1 - 8}" text-anchor="middle">{esc(label)}</text>'
    )


def role_chip(x: int, y: int, w: int, title: str, body: str, bar: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="52" rx="2" fill="{CARD}" stroke="{LINE}"/>'
        f'<rect x="{x}" y="{y}" width="4" height="52" fill="{bar}"/>'
        f'<text class="sans role" x="{x + 14}" y="{y + 20}" font-weight="700">{esc(title)}</text>'
        f'<text class="sans ns" x="{x + 14}" y="{y + 40}">{esc(body)}</text>'
    )


def track1() -> str:
    # lanes
    lanes = [
        (200, 150, "ДАННЫЕ", "#F7F2E6"),
        (350, 150, "МАС", "#F3EEE4"),
        (500, 150, "ШЛЮЗ", "#F7F2E6"),
        (650, 150, "МИР", "#F4EBD4"),
        (800, 150, "ДЕНЬГИ", "#F7F2E6"),
        (950, 150, "СДАЧА", "#F3E4DF"),
    ]
    # node box 280×72, left of content = 232
    n = {
        "zip": (232, 228),
        "case": (592, 228),
        "state": (952, 228),
        "llm": (952, 378),
        "intent": (1312, 378),
        "lim": (1312, 528),
        "crm": (1672, 528),
        "cmp": (2032, 528),
        "gdm": (2032, 678),
        "tab": (1672, 828),
        "chdd": (2032, 828),
        "pack": (1672, 978),
        "org": (2032, 978),
    }
    nw, nh = 280, 72
    parts = [head(
        "Трек 1 · поток от Model Y до организаторов",
        "Дорожки и стрелки: что передаётся и кто владеет. Полная ГДМ каждый месяц. CHDD с 2014.",
    )]
    parts.append('<text class="sans kicker" x="64" y="76">AIOS · ТАТНЕФТЬ · ТРЕК 1 · СХЕМА ПОТОКА</text>')
    parts.append('<text class="serif h1" x="64" y="108">Что куда идёт · Model Y</text>')
    parts.append('<text class="sans lede" x="64" y="132">Стрелка подписана грузом. Цвет дорожки = роль. Цикл месяца возвращает состояние, не историю Excel.</text>')
    parts.append(role_chip(1288, 72, 250, "LLM / агенты", "пишут действия, не дебиты", GOLD))
    parts.append(role_chip(1550, 72, 250, "CRM", "только отсев кандидатов", GOLD))
    parts.append(role_chip(1812, 72, 250, "ГДМ", "физика месяца, не деньги", GOLD))
    parts.append(role_chip(2074, 72, 250, "CHDD_PYTHON", "деньги с 01.01.2014", RUST))
    for y, h, name, wash in lanes:
        parts.append(lane(y, h, name, wash))
    parts.append(node(*n["zip"], nw, nh, "Model Y", "Model_Y (3).zip · deck", bar=GOLD))
    parts.append(node(*n["case"], nw, nh, "Кейс и лимиты", "то же поле, другие ограничения", bar=GOLD))
    parts.append(node(*n["state"], nw, nh, "Состояние фонда", "33 + 16 · блоки A–E", bar=GOLD))
    parts.append(node(*n["llm"], nw, nh, "Ядро МАС", "LLM обязательна · UI рядом", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["intent"], nw, nh, "Намерения", "стоп / старт / режим / перевод", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["lim"], nw, nh, "Жёсткие лимиты", "нарушитель не идёт в ГДМ", bar=GOLD))
    parts.append(node(*n["crm"], nw, nh, "CRM / Джентил", "отсев, не истина месяца", bar=GOLD))
    parts.append(node(*n["cmp"], nw, nh, "Компилятор", "wells_schedule.inc", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["gdm"], nw, nh, "Полная ГДМ", "1 месяц · tNavigator / OPM", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["tab"], nw, nh, "14 полей", "нефть т · закачка м³", bar=GOLD))
    parts.append(node(*n["chdd"], nw, nh, "CHDD_PYTHON", "--start-year 2014", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["pack"], nw, nh, "Пакет сдачи", "inc + ЧДД + код / UI", fill=RUST_WASH, bar=RUST))
    parts.append(node(*n["org"], nw, nh, "Организаторы", "один прогон графика", fill=RUST_WASH, bar=RUST))

    def mid(key: str, side: str) -> tuple[int, int]:
        x, y = n[key]
        if side == "r":
            return x + nw, y + nh // 2
        if side == "l":
            return x, y + nh // 2
        if side == "b":
            return x + nw // 2, y + nh
        return x + nw // 2, y

    # horizontal in-lane
    a, b = mid("zip", "r"), mid("case", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "архив / deck"))
    a, b = mid("case", "r"), mid("state", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "лимиты кейса"))
    a, b = mid("llm", "r"), mid("intent", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "оценка состояния"))
    a, b = mid("lim", "r"), mid("crm", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "допустимые действия"))
    a, b = mid("crm", "r"), mid("cmp", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "принятый месяц"))
    a, b = mid("tab", "r"), mid("chdd", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "таблица ГДМ"))
    a, b = mid("pack", "r"), mid("org", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "заявка", RUST))

    # vertical drops
    a, b = mid("state", "b"), mid("llm", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "канон. состояние"))
    a, b = mid("intent", "b"), mid("lim", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "ControlIntent"))
    a, b = mid("cmp", "b"), mid("gdm", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "include месяца"))
    a, b = mid("chdd", "b"), mid("org", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "заявленный ЧДД", RUST))
    a, b = mid("tab", "b"), mid("pack", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "в пакет сдачи", RUST))

    # GDM → 14 fields (down-left elbow)
    gx, gy = mid("gdm", "l")
    tx, ty = mid("tab", "r")
    parts.append(
        f'<polyline fill="none" stroke="{GOLD_INK}" stroke-width="1.6" '
        f'points="{gx},{gy} {tx + 10},{gy} {tx + 10},{ty}"/>'
        f'<polygon points="{tx + 10},{ty} {tx + 5},{ty - 10} {tx + 15},{ty - 10}" fill="{GOLD_INK}"/>'
        f'<text class="sans el" x="{(gx + tx) // 2}" y="{gy - 8}" text-anchor="middle">физика месяца</text>'
    )

    # monthly return: gutter x=1272, not the west rail (would cross zip/case)
    parts.append(
        f'<polyline fill="none" stroke="{RUST}" stroke-width="1.6" stroke-dasharray="6 4" '
        f'points="2032,714 1272,714 1272,274"/>'
        f'<polygon points="1272,264 1267,274 1277,274" fill="{RUST}"/>'
        f'<line x1="1272" y1="264" x2="1242" y2="264" stroke="{RUST}" stroke-width="1.6" stroke-dasharray="6 4"/>'
        f'<polygon points="1232,264 1242,259 1242,269" fill="{RUST}"/>'
        f'<text class="sans el" x="1640" y="706">месячный возврат · restart обновляет состояние</text>'
    )

    parts.append('<text class="sans colo" x="64" y="1284">TIMESOIL · ТРЕК 1 · ДОРОЖКИ 2400×1350 · СТРЕЛКИ ВНУТРИ ПОЛЯ</text>')
    parts.append('<text class="sans colo" x="2336" y="1284" text-anchor="end">LLM = ДЕЙСТВИЯ · CRM = ОТСЕВ · ГДМ = ФИЗИКА · CHDD = ДЕНЬГИ</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def track2() -> str:
    lanes = [
        (200, 150, "ДАННЫЕ", "#F7F2E6"),
        (350, 150, "СУРРОГАТ", "#F4EBD4"),
        (500, 150, "МАС", "#F3EEE4"),
        (650, 150, "ШЛЮЗ", "#F7F2E6"),
        (800, 150, "СЕРТИФИКАЦИЯ", "#F4EBD4"),
        (950, 150, "СДАЧА", "#F3E4DF"),
    ]
    n = {
        "zip": (232, 228),
        "opm": (592, 228),
        "train": (592, 378),
        "surr": (952, 378),
        "llm": (952, 528),
        "many": (1312, 528),
        "gate": (1312, 678),
        "rep": (1672, 828),
        "chdd": (2032, 828),
        "inc": (1672, 978),
        "org": (2032, 978),
    }
    nw, nh = 280, 72
    parts = [head(
        "Трек 2 · поток от Model Z до организаторов",
        "Суррогат ищет. Один replay OPM сертифицирует. CHDD с 2007. TimesOil не суррогат.",
    )]
    parts.append('<text class="sans kicker" x="64" y="76">AIOS · ТАТНЕФТЬ · ТРЕК 2 · СХЕМА ПОТОКА</text>')
    parts.append('<text class="serif h1" x="64" y="108">Что куда идёт · Model Z</text>')
    parts.append('<text class="sans lede" x="64" y="132">Много графиков на суррогате. Заявляют только число после одного полного replay. 4418 с 1991 — не контроль.</text>')
    parts.append(role_chip(1288, 72, 250, "Суррогат", "поиск, не заявка ЧДД", GOLD))
    parts.append(role_chip(1550, 72, 250, "LLM / агенты", "намерения, не дебиты", GOLD))
    parts.append(role_chip(1812, 72, 250, "OPM replay", "единственная физика заявки", RUST))
    parts.append(role_chip(2074, 72, 250, "CHDD_PYTHON", "деньги с 01.01.2007", RUST))
    for y, h, name, wash in lanes:
        parts.append(lane(y, h, name, wash))
    parts.append(node(*n["zip"], nw, nh, "Model Z", "Model_Z_final_OPM.zip", bar=GOLD))
    parts.append(node(*n["opm"], nw, nh, "Свои прогоны OPM", "дампов организаторов нет", bar=GOLD))
    parts.append(node(*n["train"], nw, nh, "Обучение", "свои траектории OPM", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["surr"], nw, nh, "Суррогат мира", "CRM + остаток · UQ / OOD", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["llm"], nw, nh, "Ядро МАС", "тот же цикл, мир = суррогат", bar=GOLD))
    parts.append(node(*n["many"], nw, nh, "Много графиков", "черновик ЧДД на каждом", fill=GOLD_WASH, bar=GOLD))
    parts.append(node(*n["gate"], nw, nh, "Один кандидат", "суррогат не сертифицирует", fill=RUST_WASH, bar=RUST))
    parts.append(node(*n["rep"], nw, nh, "Один replay OPM", "весь график · один раз", fill=GOLD_WASH, bar=RUST))
    parts.append(node(*n["chdd"], nw, nh, "CHDD_PYTHON", "--start-year 2007", fill=GOLD_WASH, bar=RUST))
    parts.append(node(*n["inc"], nw, nh, "wells_schedule.inc", "не CSV · без ручной правки", fill=RUST_WASH, bar=RUST))
    parts.append(node(*n["org"], nw, nh, "Организаторы", "один прогон · сход при расхождении", fill=RUST_WASH, bar=RUST))

    def mid(key: str, side: str) -> tuple[int, int]:
        x, y = n[key]
        if side == "r":
            return x + nw, y + nh // 2
        if side == "l":
            return x, y + nh // 2
        if side == "b":
            return x + nw // 2, y + nh
        return x + nw // 2, y

    a, b = mid("zip", "r"), mid("opm", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "неизменяемый deck"))
    a, b = mid("train", "r"), mid("surr", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "признаки / траектории"))
    a, b = mid("llm", "r"), mid("many", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "намерения"))
    a, b = mid("rep", "r"), mid("chdd", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "14 полей overlay"))
    a, b = mid("inc", "r"), mid("org", "l")
    parts.append(h_arrow(a[0], b[0], a[1], "график + число", RUST))

    a, b = mid("opm", "b"), mid("train", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "свои сценарии"))
    a, b = mid("surr", "b"), mid("llm", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "дешёвый мир"))
    a, b = mid("many", "b"), mid("gate", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "лучший график", RUST))
    a, b = mid("chdd", "b"), mid("org", "t")
    parts.append(v_arrow(a[0], a[1], b[1], "заявленный ЧДД", RUST))

    # gate → replay elbow
    gx, gy = mid("gate", "r")
    rx, ry = mid("rep", "l")
    parts.append(
        f'<polyline fill="none" stroke="{RUST}" stroke-width="1.6" '
        f'points="{gx},{gy} {rx - 16},{gy} {rx - 16},{ry} {rx - 10},{ry}"/>'
        f'<polygon points="{rx},{ry} {rx - 10},{ry - 5} {rx - 10},{ry + 5}" fill="{RUST}"/>'
        f'<text class="sans el" x="{gx + 20}" y="{gy - 8}">в сертификацию</text>'
    )
    # replay also writes inc
    parts.append(v_arrow(mid("rep", "b")[0], mid("rep", "b")[1], mid("inc", "t")[1], "принятый план"))

    parts.append('<text class="sans colo" x="64" y="1284">TIMESOIL · ТРЕК 2 · ДОРОЖКИ 2400×1350 · TIMESOIL ≠ СУРРОГАТ</text>')
    parts.append('<text class="sans colo" x="2336" y="1284" text-anchor="end">СУРРОГАТ = ПОИСК · OPM = ИСТИНА · CHDD = ДЕНЬГИ · 2007 ≠ 1991 ≠ 4418</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _bounds_ok(text: str) -> None:
    assert 'viewBox="0 0 2400 1350"' in text
    assert "marker-end" not in text
    assert 'y="-40"' not in text


def main() -> None:
    t1, t2 = track1(), track2()
    _bounds_ok(t1)
    _bounds_ok(t2)
    (OUT / "track1.svg").write_text(t1, encoding="utf-8")
    (OUT / "track2.svg").write_text(t2, encoding="utf-8")
    print("wrote flows", OUT / "track1.svg", OUT / "track2.svg")


if __name__ == "__main__":
    main()
