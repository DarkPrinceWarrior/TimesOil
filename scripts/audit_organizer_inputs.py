"""Inventory supplied archive bytes and deck keywords without changing model inputs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile


def inventory(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    files, counts, locations = [], Counter(), {}
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            files.append({"path": entry.filename, "bytes": entry.file_size})
            if Path(entry.filename).suffix.lower() not in {".data", ".inc", ".sch", ".txt"}:
                continue
            with archive.open(entry) as stream:
                for number, line in enumerate(stream, 1):
                    code = line.decode("utf-8", errors="replace").split("--", 1)[0].strip()
                    if re.fullmatch(r"[A-Z][A-Z0-9_]{1,15}", code):
                        counts[code] += 1
                        locations.setdefault(code, []).append(f"{entry.filename}:{number}")
    return {"archive": str(path), "sha256": digest.hexdigest(), "files": files,
            "keyword_counts": dict(sorted(counts.items())), "keyword_locations": locations,
            "scope": "Lexical inventory of all DATA/INC/SCH/TXT members; keyword presence is not a certification of semantics or OPM support."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    result = [inventory(path) for path in args.archives]
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps([{ "archive": r["archive"], "files": len(r["files"]),
                       "keyword_counts": r["keyword_counts"]} for r in result]))
