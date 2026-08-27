#!/usr/bin/env bash
# Build student.zip from student/ for distribution.
# Usage: instructor/make_student_zip.sh [output.zip]   (run from anywhere; paths are resolved relative to this script)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/student"
OUT="${1:-$ROOT/student.zip}"

[ -d "$SRC" ] || { echo "error: $SRC not found" >&2; exit 1; }

# Refuse to ship anything containing the reference-solution marker.
if grep -rIl --exclude-dir=__pycache__ "REFERENCE SOLUTION" "$SRC" >/dev/null 2>&1; then
    echo "error: the following files under student/ contain 'REFERENCE SOLUTION'; refusing to build the zip:" >&2
    grep -rIl --exclude-dir=__pycache__ "REFERENCE SOLUTION" "$SRC" >&2
    exit 1
fi

rm -f "$OUT"
( cd "$ROOT" && zip -r -q "$OUT" student \
    -x 'student/predictions/*' \
       'student/figures/*' \
       '*/__pycache__/*' '*.pyc' \
       '*.partial' \
       'student/smoke_ok.json' \
       'student/probe.jsonl' \
       'student/run_log.txt' \
       'student/.venv/*' \
       '*/.DS_Store' \
       '*/.pytest_cache/*' )
# keep the (empty) predictions/ and figures/ directories so the CLI and Makefile work out of the box
( cd "$ROOT" && mkdir -p student/predictions student/figures && zip -q "$OUT" student/predictions/ student/figures/ )

echo "wrote $OUT"
unzip -l "$OUT" | tail -1
