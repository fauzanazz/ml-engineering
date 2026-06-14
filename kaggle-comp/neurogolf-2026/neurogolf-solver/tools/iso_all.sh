#!/bin/zsh
D=${1:-/tmp/best}
OUT=${2:-/tmp/iso_all.jsonl}
: > "$OUT"
for t in $(seq 1 400); do
  uv run python tools/iso_one.py $t "$D" 2>/dev/null >> "$OUT"
  tt=$(printf "%03d" $t)
  if ! grep -q "\"task\": $t," "$OUT" 2>/dev/null && ! tail -1 "$OUT" | grep -q "\"task\": $t"; then :; fi
done
echo "DONE $OUT"
