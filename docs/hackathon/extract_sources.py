#!/usr/bin/env python3
"""One-off extractor for hackathon sources — run from docs/hackathon/."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
EXTRACTED = ROOT / "extracted"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def extract_pptx(pptx_path: Path) -> list[tuple[int, list[str]]]:
    slides: list[tuple[int, list[str]]] = []
    with zipfile.ZipFile(pptx_path) as zf:
        slide_files = sorted(
            (n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
            key=lambda n: int(re.search(r"slide(\d+)", n).group(1)),
        )
        for idx, name in enumerate(slide_files, start=1):
            root = ET.fromstring(zf.read(name))
            chunks: list[str] = []
            for el in root.iter():
                if _local(el.tag) == "t" and el.text:
                    chunks.append(el.text)
                if _local(el.tag) == "t" and el.tail:
                    chunks.append(el.tail)
            text = "".join(chunks)
            lines = _clean_lines(text)
            slides.append((idx, lines))
    return slides


def _clean_lines(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    parts = re.split(r"\n+|\s{2,}|\u00a0+", text)
    lines: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if lines and len(part) <= 3 and part.isupper():
            lines[-1] = f"{lines[-1]} {part}"
            continue
        lines.append(part)
    merged: list[str] = []
    for line in lines:
        if merged and len(line) <= 2 and line.isdigit() and merged[-1][-1].isdigit() is False:
            merged[-1] = f"{merged[-1]}{line}"
            continue
        merged.append(line)
    return merged


def slides_to_md(slides: list[tuple[int, list[str]]], title: str) -> str:
    out = [f"# {title}", ""]
    for num, lines in slides:
        out.append(f"## Слайд {num}")
        out.append("")
        if not lines:
            out.append("_(пустой слайд)_")
        else:
            for line in lines:
                out.append(line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def extract_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        pages.append((i, text))
    return pages


def pages_to_md(pages: list[tuple[int, str]], title: str, page_label: str = "Страница") -> str:
    out = [f"# {title}", ""]
    for num, text in pages:
        out.append(f"## {page_label} {num}")
        out.append("")
        if not text:
            out.append("_(пустая страница)_")
        else:
            out.append(text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    jobs = [
        (
            SOURCES / "AIOS_Онбординг (4).pptx",
            EXTRACTED / "onboarding.md",
            "AIOS — онбординг",
            "pptx",
        ),
        (
            SOURCES / "AIOS_Лекция.pptx",
            EXTRACTED / "lecture_methods.md",
            "AIOS — лекция (методы)",
            "pptx",
        ),
        (
            SOURCES / "3. AIOS_Лекция_Разработка.pdf",
            EXTRACTED / "lecture_development.md",
            "AIOS — лекция по разработке (Алёнкин)",
            "pdf",
        ),
        (
            SOURCES / "AIOS_трекеры_для_участников.pdf",
            EXTRACTED / "trackers.md",
            "AIOS — трекеры для участников",
            "pdf",
        ),
        (
            SOURCES / "Хакатон_AIOS_—_команды_по_трекам.pdf",
            EXTRACTED / "teams.md",
            "Хакатон AIOS — команды по трекам",
            "pdf",
        ),
    ]
    for src, dst, title, kind in jobs:
        if not src.exists():
            print(f"MISSING: {src}", file=sys.stderr)
            continue
        if kind == "pptx":
            content = slides_to_md(extract_pptx(src), title)
        else:
            content = pages_to_md(extract_pdf(src), title)
        dst.write_text(content, encoding="utf-8")
        print(f"WROTE {dst} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
