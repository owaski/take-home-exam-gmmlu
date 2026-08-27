#!/bin/bash
# Regenerate ALL instructor reference results with the frozen protocol.
# Usage (any machine with a GPU or an Apple-Silicon Mac):  bash instructor/run_reference.sh [work_dir]
# Runs every graded command exactly as students will, then copies predictions/ + run_log.txt to
# instructor/reference_results/ and refreshes student/reference/{en_first20.jsonl,smoke_reference.json}.
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$PKG/instructor/_refrun}"
SID="INSTRUCTOR"
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$PKG/student/." "$WORK/"
cp "$PKG"/instructor/solution/harness/*.py "$WORK/harness/"          # overlay the reference solution
cd "$WORK"
run() { python -m harness run --student-id "$SID" --batch-size 16 "$@"; }

echo "== Part 1: three scorers on en zh sw"
for L in en zh sw; do for S in LETTER CONT GEN; do run --part 1 --model qwen2.5 --lang "$L" --scorer "$S" --variant v1_en; done; done
echo "== Part 2: sweep"
for L in en de zh ar hi id sw yo; do run --part 2 --model qwen2.5 --lang "$L" --scorer LETTER --variant v1_en; done
echo "== Part 3: permutations + native"
for L in en zh hi sw; do for O in ABCD BCDA CDAB DABC; do run --part 3 --model qwen2.5 --lang "$L" --scorer LETTER --variant "perm_$O"; done; done
for L in zh hi sw; do run --part 3 --model qwen2.5 --lang "$L" --scorer LETTER --variant v2_native; done
echo "== Part 5: contrast model"
for L in en de zh ar hi id sw yo; do run --part 5 --model qwen3 --lang "$L" --scorer LETTER --variant v1_en; done
echo "== Stretch S3: SmolLM2"
for L in en de zh ar hi id sw yo; do run --part S3 --model smollm2 --lang "$L" --scorer LETTER --variant v1_en; done
echo "== Stretch S1: Belebele (300 items)"
for L in eng_Latn zho_Hans hin_Deva swh_Latn; do run --part S1 --model qwen2.5 --lang "$L" --scorer LETTER --variant v1_en --benchmark belebele --n 300; done
echo "== LITE subset (Intel-Mac mode) for bands"
mkdir -p predictions_lite
for L in en de zh ar hi id sw yo; do run --part 2 --model qwen2.5 --lang "$L" --scorer LETTER --variant v1_en --lite --out "predictions_lite/p2_qwen2.5_${L}_LETTER_v1_en.jsonl"; done

echo "== reference files"
python "$PKG/instructor/make_reference_files.py" --out "$PKG/student/reference"
mkdir -p "$PKG/instructor/reference_results"
rm -rf "$PKG/instructor/reference_results/predictions" "$PKG/instructor/reference_results/predictions_lite"
cp -r predictions "$PKG/instructor/reference_results/predictions"
cp -r predictions_lite "$PKG/instructor/reference_results/predictions_lite"
cp run_log.txt "$PKG/instructor/reference_results/"
cp "$PKG/instructor/reference_results/tokenizer.csv" . 2>/dev/null || python "$PKG/instructor/solution/tokenizer_stats.py" --out "$PKG/instructor/reference_results/tokenizer.csv"
cp "$PKG/instructor/solution/analysis.py" .
echo "== reference CSVs, bands.json, probe.jsonl (full + LITE)"
python analysis.py --predictions predictions --out "$PKG/instructor/reference_results" --figures "$PKG/instructor/reference_results/figures"
mkdir -p "$PKG/instructor/reference_results/lite"
python analysis.py --predictions predictions_lite --out "$PKG/instructor/reference_results/lite" --figures "$PKG/instructor/reference_results/lite/figures"
python "$PKG/instructor/make_bands.py" "$PKG/instructor/reference_results" "$PKG/instructor/reference_results/lite" "$PKG/instructor/reference_results"
echo "DONE reference run -> $PKG/instructor/reference_results"
