#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="/mnt/c/Users/safae/Desktop/docs/hackathon"

mkdir -p "$(dirname "$DST")"
rsync -a --delete "$SRC/" "$DST/"
